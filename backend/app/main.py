import os
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Local imports
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

app = FastAPI(title="Demand Forecasting API", description="FastAPI Backend for Demand Forecasting App")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global status tracking for background training
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
    
    pipeline_run = forecast_path.exists()
    
    return {
        "pipeline_run": pipeline_run,
        "training_status": training_status["status"],
        "training_message": training_status["message"]
    }

@app.get("/api/dimensions", response_model=DimensionsResponse)
def get_dimensions():
    csv_path, _ = get_paths()
    try:
        df = pd.read_csv(csv_path)
        # Get unique values sorted
        channels = sorted(df["Channel"].dropna().unique().tolist())
        segments = sorted(df["Customer Segment"].dropna().unique().tolist())
        categories = sorted(df["Category"].dropna().unique().tolist())
        
        # Parse months to YYYY-MM
        df["Month_Parsed"] = pd.to_datetime(df["Month"], format="%d-%b-%Y")
        months = sorted(df["Month_Parsed"].dt.strftime("%Y-%m").unique().tolist())
        
        # Include forecast months as well
        forecast_months = ["2025-04", "2025-05", "2025-06"]
        all_months = sorted(list(set(months + forecast_months)))
        
        return DimensionsResponse(
            channels=channels,
            segments=segments,
            categories=categories,
            months=all_months
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dimensions: {str(e)}")

@app.get("/api/forecast", response_model=ForecastResponse)
def get_forecast(
    channels: Optional[List[str]] = Query(None),
    segments: Optional[List[str]] = Query(None),
    categories: Optional[List[str]] = Query(None)
):
    csv_path, outputs_dir = get_paths()
    forecast_path = outputs_dir / "blender_forecast_next3m.csv"
    cv_rows_path = outputs_dir / "cv_outputs" / "cv_rows_multistep.csv"
    
    # 1. Read historical data
    try:
        df_sales = pd.read_csv(csv_path)
        df_sales["Month_Parsed"] = pd.to_datetime(df_sales["Month"], format="%d-%b-%Y")
        df_sales["Month_Str"] = df_sales["Month_Parsed"].dt.strftime("%Y-%m")
        df_sales["Quantity"] = pd.to_numeric(
            df_sales["Quantity"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading historical data: {str(e)}")
        
    # Apply filters to historical
    hist_filter = pd.Series(True, index=df_sales.index)
    if channels:
        hist_filter &= df_sales["Channel"].isin(channels)
    if segments:
        hist_filter &= df_sales["Customer Segment"].isin(segments)
    if categories:
        hist_filter &= df_sales["Category"].isin(categories)
        
    df_hist_filtered = df_sales[hist_filter]
    
    # Aggregate actuals by month
    actuals_by_month = df_hist_filtered.groupby("Month_Str")["Quantity"].sum().to_dict()
    
    # 2. Read forecast data (if exists)
    fcst_by_month = {}
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
                
            df_fcst_filtered = df_fcst[fcst_filter]
            fcst_by_month = df_fcst_filtered.groupby("Forecast Month")["Forecast"].sum().to_dict()
        except Exception as e:
            print(f"Error loading forecast file: {e}")
            
    # 3. Calculate Confidence Intervals
    # We estimate forecast uncertainty based on standard deviation of historical backtest absolute errors.
    std_error = 0.0
    if cv_rows_path.exists() and len(fcst_by_month) > 0:
        try:
            df_cv = pd.read_csv(cv_rows_path)
            cv_filter = pd.Series(True, index=df_cv.index)
            if channels:
                cv_filter &= df_cv["Channel"].isin(channels)
            if segments:
                cv_filter &= df_cv["Customer Segment"].isin(segments)
            if categories:
                cv_filter &= df_cv["Category"].isin(categories)
            
            df_cv_filtered = df_cv[cv_filter]
            if len(df_cv_filtered) > 10:
                # Use standard deviation of residuals (actual - pred)
                residuals = df_cv_filtered["actual"] - df_cv_filtered["pred"]
                std_error = float(np.std(residuals))
            else:
                # Fallback to overall standard deviation of errors
                std_error = float(np.std(df_cv["actual"] - df_cv["pred"]))
        except Exception as e:
            print(f"Error loading CV errors: {e}")
            std_error = 0.0

    # Ensure std_error is positive and sensible
    if std_error <= 0.0:
        # Fallback based on historical volume
        std_error = float(np.std(list(actuals_by_month.values()))) * 0.15 if actuals_by_month else 100.0

    # Assemble response points
    points = []
    
    # Sort all months
    all_months = sorted(list(set(list(actuals_by_month.keys()) + list(fcst_by_month.keys()))))
    
    for m in all_months:
        actual = actuals_by_month.get(m)
        fcst_val = fcst_by_month.get(m)
        
        lower_bound = None
        upper_bound = None
        
        if fcst_val is not None:
            # Add dynamic bounds
            lower_bound = max(0.0, fcst_val - 1.96 * std_error)
            upper_bound = fcst_val + 1.96 * std_error
            # Format to rounded floats
            fcst_val = round(fcst_val, 2)
            lower_bound = round(lower_bound, 2)
            upper_bound = round(upper_bound, 2)
            
        if actual is not None:
            actual = round(actual, 2)
            
        points.append(ForecastPoint(
            month=m,
            actual=actual,
            forecast=fcst_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        ))
        
    return ForecastResponse(data=points)

@app.get("/api/performance", response_model=PerformanceResponse)
def get_performance():
    csv_path, outputs_dir = get_paths()
    cv_dir = outputs_dir / "cv_outputs"
    importance_path = outputs_dir / "feature_importance.csv"
    
    if not cv_dir.exists():
        raise HTTPException(status_code=404, detail="Cross-validation metrics not found. Run model training first.")
        
    try:
        # 1. Overall
        df_overall = pd.read_csv(cv_dir / "metrics_overall_ALL.csv")
        overall_metrics = {
            "MAE": round(float(df_overall.iloc[0]["MAE"]), 2),
            "MAPE": round(float(df_overall.iloc[0]["MAPE"]), 2),
            "sMAPE": round(float(df_overall.iloc[0]["sMAPE"]), 2)
        }
        
        # 2. Slices
        def load_slice(file_name, value_col):
            df_slice = pd.read_csv(cv_dir / file_name)
            # Filter for horizon "ALL" or average across horizons
            # In our forecast.py, we saved by H. Let's average across H for visual ease or show overall by H.
            df_slice_avg = df_slice.groupby(value_col)[["MAE", "MAPE", "sMAPE"]].mean().reset_index()
            res = []
            for _, r in df_slice_avg.iterrows():
                res.append(SliceMetric(
                    slice_value=str(r[value_col]),
                    mae=round(float(r["MAE"]), 2),
                    mape=round(float(r["MAPE"]), 2),
                    smape=round(float(r["sMAPE"]), 2)
                ))
            return res
            
        by_channel = load_slice("metrics_by_channel_byH.csv", "Channel")
        by_segment = load_slice("metrics_by_segment_byH.csv", "Customer Segment")
        by_category = load_slice("metrics_by_category_byH.csv", "Category")
        
        # 3. Feature Importance
        importance_list = []
        if importance_path.exists():
            df_imp = pd.read_csv(importance_path)
            for _, r in df_imp.head(15).iterrows(): # Show top 15 features
                importance_list.append(FeatureImportanceItem(
                    feature=str(r["feature"]),
                    importance=float(r["importance"])
                ))
                
        metrics = MetricsResponse(
            overall=overall_metrics,
            by_channel=by_channel,
            by_segment=by_segment,
            by_category=by_category
        )
        
        return PerformanceResponse(
            metrics=metrics,
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
    training_status["status"] = "running"
    training_status["message"] = "Pipeline execution in progress..."
    try:
        run_pipeline(csv_path, outputs_dir)
        training_status["status"] = "success"
        training_status["message"] = "Pipeline completed successfully."
    except Exception as e:
        training_status["status"] = "error"
        training_status["message"] = f"Pipeline failed: {str(e)}"

@app.post("/api/run-forecast", response_model=RunForecastResponse)
def trigger_forecast(background_tasks: BackgroundTasks):
    global training_status
    if training_status["status"] == "running":
        return RunForecastResponse(success=False, message="Pipeline is already running.")
        
    csv_path, outputs_dir = get_paths()
    background_tasks.add_task(bg_train_pipeline, csv_path, outputs_dir)
    return RunForecastResponse(success=True, message="Pipeline training triggered in the background.")
