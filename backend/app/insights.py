import os
import pandas as pd
import numpy as np
from pathlib import Path
import httpx

# Find sales data helper
from .forecast import find_sales_data

def generate_insights_payload(csv_path: str = None, outputs_dir: str = None) -> dict:
    if csv_path is None:
        csv_path = find_sales_data()
    workspace_root = Path(csv_path).parent
    
    if outputs_dir is None:
        outputs_dir = workspace_root / "data" / "outputs"
    else:
        outputs_dir = Path(outputs_dir)
        
    forecast_path = outputs_dir / "blender_forecast_next3m.csv"
    cv_overall_path = outputs_dir / "cv_outputs" / "metrics_overall_ALL.csv"
    cv_channel_path = outputs_dir / "cv_outputs" / "metrics_by_channel_byH.csv"
    
    # 1. Fallback / check if outputs exist. If not, return a placeholder until model runs.
    if not forecast_path.exists() or not cv_overall_path.exists():
        return {
            "ready": False,
            "summary": "Forecasting model pipeline hasn't been run yet. Please trigger model execution to compute AI Insights.",
            "metrics": {},
            "bullet_points": []
        }

    # Load data
    df_sales = pd.read_csv(csv_path)
    df_sales["Month"] = pd.to_datetime(df_sales["Month"], format="%d-%b-%Y")
    
    # Clean Quantity in historical data
    df_sales["Quantity"] = pd.to_numeric(
        df_sales["Quantity"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0.0)
    
    df_fcst = pd.read_csv(forecast_path)
    
    # Load Overall accuracy metrics
    df_metrics_all = pd.read_csv(cv_overall_path)
    overall_mape = float(df_metrics_all.iloc[0]["MAPE"])
    overall_smape = float(df_metrics_all.iloc[0]["sMAPE"])
    
    # Calculate historical averages (last 3 months)
    max_hist_date = df_sales["Month"].max()
    hist_last_3m_start = max_hist_date - pd.offsets.MonthBegin(2)
    df_hist_3m = df_sales[df_sales["Month"] >= hist_last_3m_start]
    
    total_hist_3m_quantity = df_hist_3m["Quantity"].sum()
    avg_hist_monthly_quantity = total_hist_3m_quantity / 3.0
    
    # Calculate forecast totals
    total_fcst_quantity = df_fcst["Forecast"].sum()
    avg_fcst_monthly_quantity = total_fcst_quantity / 3.0
    
    # Growth percent
    growth_pct = 0.0
    if avg_hist_monthly_quantity > 0:
        growth_pct = ((avg_fcst_monthly_quantity - avg_hist_monthly_quantity) / avg_hist_monthly_quantity) * 100.0
        
    # Find top categories & channels in forecast
    cat_summary = df_fcst.groupby("Category")["Forecast"].sum().sort_values(ascending=False)
    top_category = cat_summary.index[0] if len(cat_summary) > 0 else "N/A"
    top_category_share = (cat_summary.iloc[0] / total_fcst_quantity * 100.0) if total_fcst_quantity > 0 else 0.0
    
    ch_summary = df_fcst.groupby("Channel")["Forecast"].sum().sort_values(ascending=False)
    top_channel = ch_summary.index[0] if len(ch_summary) > 0 else "N/A"
    top_channel_share = (ch_summary.iloc[0] / total_fcst_quantity * 100.0) if total_fcst_quantity > 0 else 0.0
    
    # Slices with poor accuracy or drop in invoice volume
    # Find channels with falling invoice volume in last 3 months
    df_sales["Invoices"] = pd.to_numeric(
        df_sales["Invoices"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0.0)
    
    inv_by_month = df_sales.groupby(["Month", "Channel"])["Invoices"].sum().unstack().fillna(0)
    warnings = []
    
    if len(inv_by_month) >= 3:
        last_3_months_inv = inv_by_month.tail(3)
        for ch in last_3_months_inv.columns:
            v = last_3_months_inv[ch].values
            if v[0] > v[1] > v[2] and v[0] > 0:  # declining invoice count month-over-month
                pct_decline = ((v[0] - v[2]) / v[0]) * 100.0
                warnings.append(f"Invoices in the '{ch}' channel have declined by {pct_decline:.1f}% over the last 3 historical months, indicating a potential slowdown.")

    # High MAPE warnings (load by channel accuracy)
    if cv_channel_path.exists():
        df_ch_metrics = pd.read_csv(cv_channel_path)
        # Average MAPE across horizons per channel
        ch_avg_mape = df_ch_metrics.groupby("Channel")["MAPE"].mean()
        high_error_channels = ch_avg_mape[ch_avg_mape > 25.0]
        for ch, err in high_error_channels.items():
            warnings.append(f"Forecast for '{ch}' channel has higher model volatility (MAPE of {err:.1f}% in validation), proceed with caution on supply planning.")

    # Local narrative generation
    bullet_points = [
        f"**Accuracy Assessment:** Model validation shows strong predictive performance with an overall MAPE of {overall_mape:.1f}% ({100-overall_mape:.1f}% accuracy). sMAPE is stable at {overall_smape:.1f}%.",
        f"**Aggregate Projection:** The next 3-month forecast projects a total demand of {total_fcst_quantity:,.0f} units, representing a monthly average of {avg_fcst_monthly_quantity:,.0f} units. This is a **{growth_pct:+.1f}%** shift compared to the recent historical average ({avg_hist_monthly_quantity:,.0f} units/month).",
        f"**Product Concentration:** '{top_category}' stands out as the primary volume driver, capturing **{top_category_share:.1f}%** of the forecasted demand ({cat_summary.iloc[0]:,.0f} units total).",
        f"**Channel Concentration:** '{top_channel}' is the most active distribution channel, accounting for **{top_channel_share:.1f}%** of all projected transactions.",
    ]
    
    # Add warnings or custom rules
    if warnings:
        bullet_points.extend([f"**Alert:** {w}" for w in warnings[:3]])
    else:
        bullet_points.append("**Risk Profile:** Overall transaction volume is steady, and there are no immediate warnings detected in invoice momentum.")
        
    summary_text = (
        f"Overall forecasting accuracy is solid at {100-overall_mape:.1f}% (MAPE of {overall_mape:.1f}%). "
        f"We project a monthly average demand of {avg_fcst_monthly_quantity:,.0f} units over the next quarter, "
        f"representing a {growth_pct:+.1f}% change from the baseline. "
        f"Volume is heavily concentrated in the '{top_channel}' channel and '{top_category}' category."
    )
    
    # Check for Gemini API key and run live LLM polish if configured
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        print("GEMINI_API_KEY detected. Requesting live LLM-powered narrative refinement...")
        refined_summary, refined_bullets = get_gemini_insights(gemini_key, summary_text, bullet_points)
        if refined_summary and refined_bullets:
            summary_text = refined_summary
            bullet_points = refined_bullets

    return {
        "ready": True,
        "summary": summary_text,
        "metrics": {
            "overall_mape": overall_mape,
            "overall_smape": overall_smape,
            "total_forecast": total_fcst_quantity,
            "growth_percentage": growth_pct,
            "top_category": top_category,
            "top_category_share": top_category_share,
            "top_channel": top_channel,
            "top_channel_share": top_channel_share
        },
        "bullet_points": bullet_points
    }

def get_gemini_insights(api_key: str, default_summary: str, default_bullets: list) -> tuple:
    # Query Gemini API via REST
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    prompt = f"""
    You are an expert demand forecasting AI analyst.
    Below is a summary of statistical metrics and findings generated from our machine learning model (LightGBM) forecasting sales:
    
    Baseline Summary: {default_summary}
    Bullet points:
    {" ".join(default_bullets)}
    
    Please refine this information into a highly professional executive briefing.
    Format your response EXACTLY as a JSON object with two keys:
    1. "summary": A concise 2-3 sentence paragraph that gives a premium overview of accuracy and trend.
    2. "bullet_points": An array of 4-5 key bullet points (Markdown format) summarizing the highlights, drivers, and potential risk factors.
    
    Return ONLY valid raw JSON. Do not wrap in markdown code blocks.
    """
    
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        if response.status_code == 200:
            import json
            res_json = response.json()
            text_resp = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Clean possible markdown wrapping if any remains
            if text_resp.startswith("```json"):
                text_resp = text_resp.replace("```json", "", 1)
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3].strip()
            parsed = json.loads(text_resp)
            return parsed.get("summary"), parsed.get("bullet_points")
    except Exception as e:
        print(f"Error calling Gemini API: {e}. Falling back to default statistical generator.")
    return None, None
