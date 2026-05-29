import os
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .forecast import run_pipeline, find_sales_data
from .insights import generate_insights_payload
from .schemas import (
    DimensionsResponse,
    ForecastResponse,
    ForecastPoint,
    MetricsResponse,
    SliceMetric,
    PerformanceResponse,
    FeatureImportanceItem,
    InsightsResponse,
    RunForecastResponse
)

app = FastAPI(title="Demand Forecasting API", description="FastAPI Backend for Weekly Demand Forecasting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_status = {"status": "idle", "message": "Ready"}

def get_paths():
    try:
        csv_path = find_sales_data()
        workspace_root = csv_path.parent
        outputs_dir = workspace_root / "data" / "outputs"
        return csv_path, outputs_dir
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def get_model_status():
    csv_path, outputs_dir = get_paths()
    forecast_path = outputs_dir / "blender_forecast_next3m.csv"
    return {
        "pipeline_run":      forecast_path.exists(),
        "training_status":   training_status["status"],
        "training_message":  training_status["message"]
    }

@app.get("/api/dimensions", response_model=DimensionsResponse)
def get_dimensions():
    csv_path, _ = get_paths()
    try:
        df = pd.read_csv(csv_path)
        channels  = sorted(df["Channel"].dropna().unique().tolist())
        segments  = sorted(df["Customer Segment"].dropna().unique().tolist())
        categories = sorted(df["Category"].dropna().unique().tolist())

        # Parse weekly dates → YYYY-MM-DD strings
        df["Week_Parsed"] = pd.to_datetime(df["Week"], format="%d-%b-%Y")
        weeks = sorted(df["Week_Parsed"].dt.strftime("%Y-%m-%d").unique().tolist())

        # Append the 13 forecast weeks
        forecast_weeks = [
            "2025-04-07","2025-04-14","2025-04-21","2025-04-28",
            "2025-05-05","2025-05-12","2025-05-19","2025-05-26",
            "2025-06-02","2025-06-09","2025-06-16","2025-06-23","2025-06-30",
        ]
        all_weeks = sorted(list(set(weeks + forecast_weeks)))

        return DimensionsResponse(
            channels=channels,
            segments=segments,
            categories=categories,
            months=all_weeks      # field name kept for schema compatibility
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dimensions: {str(e)}")

@app.get("/api/forecast", response_model=ForecastResponse)
def get_forecast(
    channels:   Optional[List[str]] = Query(None),
    segments:   Optional[List[str]] = Query(None),
    categories: Optional[List[str]] = Query(None)
):
    csv_path, outputs_dir = get_paths()
    forecast_path  = outputs_dir / "blender_forecast_next3m.csv"
    cv_rows_path   = outputs_dir / "cv_outputs" / "cv_rows_multistep.csv"

    # 1. Historical weekly data
    try:
        df_sales = pd.read_csv(csv_path)
        df_sales["Week_Parsed"] = pd.to_datetime(df_sales["Week"], format="%d-%b-%Y")
        df_sales["Week_Str"]    = df_sales["Week_Parsed"].dt.strftime("%Y-%m-%d")
        df_sales["Quantity"]    = pd.to_numeric(
            df_sales["Quantity"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading historical data: {str(e)}")

    hist_filter = pd.Series(True, index=df_sales.index)
    if channels:
        hist_filter &= df_sales["Channel"].isin(channels)
    if segments:
        hist_filter &= df_sales["Customer Segment"].isin(segments)
    if categories:
        hist_filter &= df_sales["Category"].isin(categories)

    actuals_by_week = df_sales[hist_filter].groupby("Week_Str")["Quantity"].sum().to_dict()

    # 2. Forecast data
    fcst_by_week = {}
    if forecast_path.exists():
        try:
            df_fcst = pd.read_csv(forecast_path)
            fcst_filter = pd.Series(True, index=df_fcst.index)
            if channels:
                fcst_filter &= df_fcst["Channel"].isin(channels)
            if segments:
                fcst_filter &= df_fcst["Customer Segment"].isin(segments)
            if categories:
                fcst_filter &= df_fcst["Category"].isin(categories)

            fcst_by_week = df_fcst[fcst_filter].groupby("Forecast Week")["Forecast"].sum().to_dict()
        except Exception as e:
            print(f"Error loading forecast file: {e}")

    # 3. Confidence intervals from CV residuals
    std_error = 0.0
    if cv_rows_path.exists() and len(fcst_by_week) > 0:
        try:
            df_cv = pd.read_csv(cv_rows_path)
            cv_filter = pd.Series(True, index=df_cv.index)
            if channels:
                cv_filter &= df_cv["Channel"].isin(channels)
            if segments:
                cv_filter &= df_cv["Customer Segment"].isin(segments)
            if categories:
                cv_filter &= df_cv["Category"].isin(categories)

            df_cv_f = df_cv[cv_filter]
            if len(df_cv_f) > 10:
                std_error = float(np.std(df_cv_f["actual"] - df_cv_f["pred"]))
            else:
                std_error = float(np.std(df_cv["actual"] - df_cv["pred"]))
        except Exception as e:
            print(f"Error loading CV errors: {e}")

    if std_error <= 0.0:
        std_error = float(np.std(list(actuals_by_week.values()))) * 0.15 if actuals_by_week else 100.0

    # 4. Assemble response
    all_weeks = sorted(list(set(list(actuals_by_week.keys()) + list(fcst_by_week.keys()))))
    points = []

    for w in all_weeks:
        actual   = actuals_by_week.get(w)
        fcst_val = fcst_by_week.get(w)
        lower_bound = upper_bound = None

        if fcst_val is not None:
            lower_bound = round(max(0.0, fcst_val - 1.96 * std_error), 2)
            upper_bound = round(fcst_val + 1.96 * std_error, 2)
            fcst_val    = round(fcst_val, 2)

        if actual is not None:
            actual = round(actual, 2)

        points.append(ForecastPoint(
            month=w,          # field name kept for schema compatibility
            actual=actual,
            forecast=fcst_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        ))

    return ForecastResponse(data=points)

@app.get("/api/performance", response_model=PerformanceResponse)
def get_performance():
    csv_path, outputs_dir = get_paths()
    cv_dir         = outputs_dir / "cv_outputs"
    importance_path = outputs_dir / "feature_importance.csv"

    if not cv_dir.exists():
        raise HTTPException(status_code=404, detail="Cross-validation metrics not found. Run model training first.")

    try:
        df_overall = pd.read_csv(cv_dir / "metrics_overall_ALL.csv")
        overall_metrics = {
            "MAE":   round(float(df_overall.iloc[0]["MAE"]),   2),
            "MAPE":  round(float(df_overall.iloc[0]["MAPE"]),  2),
            "sMAPE": round(float(df_overall.iloc[0]["sMAPE"]), 2)
        }

        def load_slice(file_name, value_col):
            df_s = pd.read_csv(cv_dir / file_name)
            df_avg = df_s.groupby(value_col)[["MAE", "MAPE", "sMAPE"]].mean().reset_index()
            return [
                SliceMetric(
                    slice_value=str(r[value_col]),
                    mae=round(float(r["MAE"]),   2),
                    mape=round(float(r["MAPE"]),  2),
                    smape=round(float(r["sMAPE"]), 2)
                )
                for _, r in df_avg.iterrows()
            ]

        by_channel  = load_slice("metrics_by_channel_byH.csv",  "Channel")
        by_segment  = load_slice("metrics_by_segment_byH.csv",  "Customer Segment")
        by_category = load_slice("metrics_by_category_byH.csv", "Category")

        importance_list = []
        if importance_path.exists():
            df_imp = pd.read_csv(importance_path)
            for _, r in df_imp.head(15).iterrows():
                importance_list.append(FeatureImportanceItem(
                    feature=str(r["feature"]),
                    importance=float(r["importance"])
                ))

        return PerformanceResponse(
            metrics=MetricsResponse(
                overall=overall_metrics,
                by_channel=by_channel,
                by_segment=by_segment,
                by_category=by_category
            ),
            importance=importance_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading performance metrics: {str(e)}")

@app.get("/api/insights", response_model=InsightsResponse)
def get_insights():
    csv_path, outputs_dir = get_paths()
    try:
        payload = generate_insights_payload(csv_path, outputs_dir)
        return InsightsResponse(**payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

def bg_train_pipeline(csv_path, outputs_dir):
    global training_status
    training_status = {"status": "running", "message": "Pipeline execution in progress..."}
    try:
        run_pipeline(csv_path, outputs_dir)
        training_status = {"status": "success", "message": "Pipeline completed successfully."}
    except Exception as e:
        training_status = {"status": "error", "message": f"Pipeline failed: {str(e)}"}

@app.post("/api/run-forecast", response_model=RunForecastResponse)
def trigger_forecast(background_tasks: BackgroundTasks):
    if training_status["status"] == "running":
        return RunForecastResponse(success=False, message="Pipeline is already running.")
    csv_path, outputs_dir = get_paths()
    background_tasks.add_task(bg_train_pipeline, csv_path, outputs_dir)
    return RunForecastResponse(success=True, message="Weekly pipeline training triggered in the background.")
