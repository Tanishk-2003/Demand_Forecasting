import os
import pandas as pd
import numpy as np
from pathlib import Path
import httpx

from .forecast import find_sales_data

def generate_insights_payload(csv_path: str = None, outputs_dir: str = None) -> dict:
    if csv_path is None:
        csv_path = find_sales_data()
    workspace_root = Path(csv_path).parent

    if outputs_dir is None:
        outputs_dir = workspace_root / "data" / "outputs"
    else:
        outputs_dir = Path(outputs_dir)

    forecast_path    = outputs_dir / "blender_forecast_next3m.csv"
    cv_overall_path  = outputs_dir / "cv_outputs" / "metrics_overall_ALL.csv"
    cv_channel_path  = outputs_dir / "cv_outputs" / "metrics_by_channel_byH.csv"

    if not forecast_path.exists() or not cv_overall_path.exists():
        return {
            "ready": False,
            "summary": "Forecasting model pipeline hasn't been run yet. Please trigger model execution to compute AI Insights.",
            "metrics": {},
            "bullet_points": []
        }

    # Load weekly sales data
    df_sales = pd.read_csv(csv_path)
    df_sales["Week"] = pd.to_datetime(df_sales["Week"], format="%d-%b-%Y")

    df_sales["Quantity"] = pd.to_numeric(
        df_sales["Quantity"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0.0)

    df_fcst = pd.read_csv(forecast_path)

    # Accuracy metrics
    df_metrics_all = pd.read_csv(cv_overall_path)
    overall_mape  = float(df_metrics_all.iloc[0]["MAPE"])
    overall_smape = float(df_metrics_all.iloc[0]["sMAPE"])

    # Historical averages — last 12 weeks (~3 months)
    max_hist_date      = df_sales["Week"].max()
    hist_last_12w_start = max_hist_date - pd.Timedelta(weeks=11)
    df_hist_12w        = df_sales[df_sales["Week"] >= hist_last_12w_start]

    total_hist_12w_quantity  = df_hist_12w["Quantity"].sum()
    avg_hist_weekly_quantity = total_hist_12w_quantity / 12.0

    # Forecast totals (13 weeks)
    total_fcst_quantity      = df_fcst["Forecast"].sum()
    avg_fcst_weekly_quantity = total_fcst_quantity / 13.0

    # Growth vs historical weekly average
    growth_pct = 0.0
    if avg_hist_weekly_quantity > 0:
        growth_pct = ((avg_fcst_weekly_quantity - avg_hist_weekly_quantity) / avg_hist_weekly_quantity) * 100.0

    # Top category & channel in forecast
    cat_summary    = df_fcst.groupby("Category")["Forecast"].sum().sort_values(ascending=False)
    top_category   = cat_summary.index[0] if len(cat_summary) > 0 else "N/A"
    top_cat_share  = (cat_summary.iloc[0] / total_fcst_quantity * 100.0) if total_fcst_quantity > 0 else 0.0

    ch_summary     = df_fcst.groupby("Channel")["Forecast"].sum().sort_values(ascending=False)
    top_channel    = ch_summary.index[0] if len(ch_summary) > 0 else "N/A"
    top_ch_share   = (ch_summary.iloc[0] / total_fcst_quantity * 100.0) if total_fcst_quantity > 0 else 0.0

    # Invoice decline warnings — last 12 weeks by channel
    df_sales["Invoices"] = pd.to_numeric(
        df_sales["Invoices"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0.0)

    inv_by_week = df_sales.groupby(["Week", "Channel"])["Invoices"].sum().unstack().fillna(0)
    warnings = []

    if len(inv_by_week) >= 4:
        last_4w = inv_by_week.tail(4)
        for ch in last_4w.columns:
            v = last_4w[ch].values
            if v[0] > v[1] > v[2] > v[3] and v[0] > 0:
                pct_decline = ((v[0] - v[3]) / v[0]) * 100.0
                warnings.append(
                    f"Invoices in the '{ch}' channel have declined by {pct_decline:.1f}% "
                    f"over the last 4 weeks, indicating a potential slowdown."
                )

    # High MAPE channel warnings
    if cv_channel_path.exists():
        df_ch_metrics     = pd.read_csv(cv_channel_path)
        ch_avg_mape       = df_ch_metrics.groupby("Channel")["MAPE"].mean()
        high_error_channels = ch_avg_mape[ch_avg_mape > 25.0]
        for ch, err in high_error_channels.items():
            warnings.append(
                f"Forecast for '{ch}' channel has higher model volatility "
                f"(MAPE of {err:.1f}% in validation), proceed with caution on supply planning."
            )

    bullet_points = [
        f"**Accuracy Assessment:** Model validation shows strong predictive performance with an overall MAPE of {overall_mape:.1f}% ({100-overall_mape:.1f}% accuracy). sMAPE is stable at {overall_smape:.1f}%.",
        f"**Aggregate Projection:** The 13-week forecast projects a total demand of {total_fcst_quantity:,.0f} units, averaging {avg_fcst_weekly_quantity:,.0f} units/week. This is a **{growth_pct:+.1f}%** shift vs the recent 12-week historical average ({avg_hist_weekly_quantity:,.0f} units/week).",
        f"**Product Concentration:** '{top_category}' is the primary volume driver, capturing **{top_cat_share:.1f}%** of forecasted demand ({cat_summary.iloc[0]:,.0f} units total).",
        f"**Channel Concentration:** '{top_channel}' is the most active distribution channel, accounting for **{top_ch_share:.1f}%** of all projected weekly transactions.",
    ]

    if warnings:
        bullet_points.extend([f"**Alert:** {w}" for w in warnings[:3]])
    else:
        bullet_points.append("**Risk Profile:** Weekly transaction volume is steady with no immediate warnings detected in invoice momentum.")

    summary_text = (
        f"Overall forecasting accuracy is solid at {100-overall_mape:.1f}% (MAPE of {overall_mape:.1f}%). "
        f"The model projects a weekly average demand of {avg_fcst_weekly_quantity:,.0f} units over the next 13 weeks, "
        f"representing a {growth_pct:+.1f}% change from the recent baseline. "
        f"Volume is concentrated in the '{top_channel}' channel and '{top_category}' category."
    )

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        print("GEMINI_API_KEY detected. Requesting live LLM-powered narrative refinement...")
        refined_summary, refined_bullets = get_gemini_insights(gemini_key, summary_text, bullet_points)
        if refined_summary and refined_bullets:
            summary_text   = refined_summary
            bullet_points  = refined_bullets

    return {
        "ready": True,
        "summary": summary_text,
        "metrics": {
            "overall_mape":      overall_mape,
            "overall_smape":     overall_smape,
            "total_forecast":    total_fcst_quantity,
            "growth_percentage": growth_pct,
            "top_category":      top_category,
            "top_category_share": top_cat_share,
            "top_channel":       top_channel,
            "top_channel_share": top_ch_share
        },
        "bullet_points": bullet_points
    }


def get_gemini_insights(api_key: str, default_summary: str, default_bullets: list) -> tuple:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    prompt = f"""
    You are an expert demand forecasting AI analyst.
    Below is a summary of statistical metrics and findings generated from our machine learning model (LightGBM) forecasting weekly sales:

    Baseline Summary: {default_summary}
    Bullet points:
    {" ".join(default_bullets)}

    Please refine this information into a highly professional executive briefing.
    Format your response EXACTLY as a JSON object with two keys:
    1. "summary": A concise 2-3 sentence paragraph giving a premium overview of accuracy and weekly trend.
    2. "bullet_points": An array of 4-5 key bullet points (Markdown format) summarizing highlights, weekly drivers, and risk factors.

    Return ONLY valid raw JSON. Do not wrap in markdown code blocks.
    """
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        if response.status_code == 200:
            import json
            res_json  = response.json()
            text_resp = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text_resp.startswith("```json"):
                text_resp = text_resp.replace("```json", "", 1)
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3].strip()
            parsed = json.loads(text_resp)
            return parsed.get("summary"), parsed.get("bullet_points")
    except Exception as e:
        print(f"Error calling Gemini API: {e}. Falling back to default statistical generator.")
    return None, None
