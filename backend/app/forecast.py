import os
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# --- CONFIG ---
TARGET_COL = "Quantity"
DATE_COL = "Month"
KEY_COLS = ["Channel", "Category", "Customer Segment"]
FORECAST_MONTHS = ["2025-04", "2025-05", "2025-06"]
CV_LAST_N_ORIGINS = 6
HORIZON = 3

# --------- Helpers ----------
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

def add_lags_rolls(g, target):
    g = g.copy()
    for L in [1, 2, 3]:
        g[f"{target}_lag{L}"] = g[target].shift(L)
    g[f"{target}_rmean3"] = g[target].shift(1).rolling(3).mean()
    g[f"{target}_rmean6"] = g[target].shift(1).rolling(6).mean()
    return g

def find_sales_data():
    # Helper to find sales_data.csv in parent directories
    paths_to_check = [
        Path("sales_data.csv"),
        Path("../sales_data.csv"),
        Path("../../sales_data.csv"),
        Path(os.getcwd()) / "sales_data.csv",
        Path(__file__).parent.parent.parent / "sales_data.csv"
    ]
    for p in paths_to_check:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not locate sales_data.csv in workspace.")

def run_pipeline(csv_path: str = None, output_dir: str = None):
    if csv_path is None:
        csv_path = str(find_sales_data())
    
    if output_dir is None:
        # Default to a data/outputs folder under workspace root
        workspace_root = Path(csv_path).parent
        output_dir = workspace_root / "data" / "outputs"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {csv_path}")
    print(f"Outputs will be written to {output_dir}")

    # --- LOAD & PREP ---
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d-%b-%Y")
    df.sort_values([*KEY_COLS, DATE_COL], inplace=True)

    # Clean numeric cols
    num_cols_raw = ["ASP", "Customers", "Cities Billed", "Invoices", "Quantity"]
    for c in num_cols_raw:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # IDs & features
    df["series_id"] = df[KEY_COLS].astype(str).agg("|".join, axis=1)
    ohe_cols = [c for c in df.columns if c.startswith("OHE_")]
    num_predictors = [c for c in num_cols_raw if c != TARGET_COL]

    # Lags/rolls (native GroupBy operations to avoid apply index issues in newer pandas versions)
    for L in [1, 2, 3]:
        df[f"{TARGET_COL}_lag{L}"] = df.groupby("series_id")[TARGET_COL].shift(L)
    df[f"{TARGET_COL}_rmean3"] = df.groupby("series_id")[TARGET_COL].transform(lambda x: x.shift(1).rolling(3).mean())
    df[f"{TARGET_COL}_rmean6"] = df.groupby("series_id")[TARGET_COL].transform(lambda x: x.shift(1).rolling(6).mean())

    # ffill regressors per series
    for c in num_predictors:
        df[c] = df.groupby("series_id")[c].transform(lambda s: s.ffill())

    # time feats
    df["month_num"] = df[DATE_COL].dt.month
    df["quarter"]   = df[DATE_COL].dt.quarter
    df["year"]      = df[DATE_COL].dt.year

    # modeling frame (need lags)
    df_model = df.dropna(subset=[f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3"]).copy()

    feature_cols = (
        ohe_cols
        + num_predictors
        + [f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
           f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6", "month_num", "quarter", "year"]
    )

    # --------- MULTI-STEP ROLLING BACKTEST ---------
    print("Running multi-step rolling cross-validation backtest...")
    
    lgb_params = dict(
        objective="poisson",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1
    )

    months_sorted = sorted(df_model[DATE_COL].drop_duplicates().tolist())
    valid_origins = []
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    for i, m in enumerate(months_sorted):
        if i + HORIZON - 1 < len(months_sorted):
            valid_origins.append(m)
            
    origins = valid_origins[-CV_LAST_N_ORIGINS:] if CV_LAST_N_ORIGINS <= len(valid_origins) else valid_origins
    rows_all = []

    for origin in origins:
        train_df = df_model[df_model[DATE_COL] < origin].copy()
        if train_df.empty:
            continue

        X_tr = train_df[feature_cols].copy()
        y_tr = train_df[TARGET_COL].copy()

        val_mask = train_df[DATE_COL] == train_df[DATE_COL].max()
        if val_mask.sum() < 50:
            X_tr_in, y_tr_in = X_tr, y_tr
            X_val, y_val = X_tr.tail(min(200, len(X_tr))), y_tr.tail(min(200, len(y_tr)))
        else:
            X_tr_in, y_tr_in = X_tr[~val_mask], y_tr[~val_mask]
            X_val, y_val = X_tr[val_mask], y_tr[val_mask]

        for X_ in (X_tr_in, X_val):
            for c in [f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
                      f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6"]:
                if c in X_.columns:
                    X_[c] = X_[c].fillna(0.0)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr_in,
            y_tr_in,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )

        static_cols = KEY_COLS + [c for c in feature_cols if c.startswith("OHE_")] + [c for c in num_predictors]
        latest_static = (
            df_model[df_model[DATE_COL] < origin]
            .sort_values(["series_id", DATE_COL])
            .groupby("series_id")
            .tail(1)
            .set_index("series_id")[static_cols]
        )

        work = df_model[df_model[DATE_COL] < origin][[DATE_COL, "series_id", TARGET_COL]].copy()

        for h in range(HORIZON):
            m = months_sorted[month_to_idx[origin] + h]
            fut = latest_static.reset_index().copy()
            fut[DATE_COL] = m
            fut["month_num"] = m.month
            fut["quarter"]   = pd.Timestamp(m).quarter
            fut["year"]      = m.year

            lags = []
            for sid in fut["series_id"]:
                hist = work.loc[work["series_id"] == sid, [DATE_COL, TARGET_COL]].sort_values(DATE_COL)
                hist_ex = hist.set_index(DATE_COL).sort_index()
                hist_ex = hist_ex.loc[hist_ex.index < m]
                lag1 = hist_ex[TARGET_COL].iloc[-1] if len(hist_ex) >= 1 else np.nan
                lag2 = hist_ex[TARGET_COL].iloc[-2] if len(hist_ex) >= 2 else np.nan
                lag3 = hist_ex[TARGET_COL].iloc[-3] if len(hist_ex) >= 3 else np.nan
                r3  = hist_ex[TARGET_COL].rolling(3).mean().iloc[-1] if len(hist_ex) >= 3 else np.nan
                r6  = hist_ex[TARGET_COL].rolling(6).mean().iloc[-1] if len(hist_ex) >= 6 else np.nan
                lags.append((sid, lag1, lag2, lag3, r3, r6))
                
            lags_df = pd.DataFrame(lags, columns=["series_id",
                                                  f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
                                                  f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6"])
            fut = fut.merge(lags_df, on="series_id", how="left")

            X_fut = fut[feature_cols].copy()
            for c in [f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
                      f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6"]:
                X_fut[c] = X_fut[c].fillna(0.0)

            y_hat = model.predict(X_fut, num_iteration=model.best_iteration_)
            y_hat = np.clip(y_hat, 0, None)

            actual_df = df_model[df_model[DATE_COL] == m][["series_id", TARGET_COL, *KEY_COLS]].copy()
            out = fut[["series_id", *KEY_COLS]].copy()
            out[DATE_COL] = m
            out["origin_month"] = origin
            out["horizon"] = h + 1
            out["pred"] = y_hat

            out = out.merge(actual_df.rename(columns={TARGET_COL: "actual"}), on=["series_id", *KEY_COLS], how="left")
            rows_all.append(out)

            add_back = fut[[DATE_COL, "series_id"]].copy()
            add_back[TARGET_COL] = y_hat
            work = pd.concat([work, add_back], ignore_index=True)

    cv_rows = pd.concat(rows_all, ignore_index=True)
    cv_rows = cv_rows.dropna(subset=["actual"])

    # Metrics computation
    cv_rows["abs_err"] = (cv_rows["actual"] - cv_rows["pred"]).abs()
    cv_rows["APE"] = cv_rows["abs_err"] / np.maximum(cv_rows["actual"].abs(), 1e-6)

    overall = cv_rows.groupby("horizon").apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        )),
        include_groups=False
    ).reset_index().rename(columns={"horizon": "H"})

    overall_all = pd.DataFrame({
        "H": ["ALL"],
        "MAE": [cv_rows["abs_err"].mean()],
        "MAPE": [mape(cv_rows["actual"], cv_rows["pred"])],
        "sMAPE": [smape(cv_rows["actual"], cv_rows["pred"])]
    })

    by_channel = cv_rows.groupby(["horizon", "Channel"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        )),
        include_groups=False
    ).reset_index().rename(columns={"horizon": "H"})

    by_segment = cv_rows.groupby(["horizon", "Customer Segment"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        )),
        include_groups=False
    ).reset_index().rename(columns={"horizon": "H"})

    by_category = cv_rows.groupby(["horizon", "Category"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        )),
        include_groups=False
    ).reset_index().rename(columns={"horizon": "H"})

    by_series = cv_rows.groupby(["horizon", "series_id"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        )),
        include_groups=False
    ).reset_index().rename(columns={"horizon": "H"}).sort_values(["H", "MAPE"])

    # Save CV outputs
    cv_dir = output_dir / "cv_outputs"
    cv_dir.mkdir(exist_ok=True)
    cv_rows.to_csv(cv_dir / "cv_rows_multistep.csv", index=False)
    overall.to_csv(cv_dir / "metrics_overall_byH.csv", index=False)
    overall_all.to_csv(cv_dir / "metrics_overall_ALL.csv", index=False)
    by_channel.to_csv(cv_dir / "metrics_by_channel_byH.csv", index=False)
    by_segment.to_csv(cv_dir / "metrics_by_segment_byH.csv", index=False)
    by_category.to_csv(cv_dir / "metrics_by_category_byH.csv", index=False)
    by_series.to_csv(cv_dir / "metrics_by_series_byH.csv", index=False)

    # ---- Train FINAL model and forecast Apr–Jun 2025 ----
    print("Training final model on all historical data...")
    last_date = df_model[DATE_COL].max()
    val_cut = (last_date - pd.offsets.MonthBegin(2))
    train_mask = df_model[DATE_COL] < val_cut
    valid_mask = df_model[DATE_COL] >= val_cut

    X_train = df_model.loc[train_mask, feature_cols].copy()
    y_train = df_model.loc[train_mask, TARGET_COL]
    X_valid = df_model.loc[valid_mask, feature_cols].copy()
    y_valid = df_model.loc[valid_mask, TARGET_COL]

    for c in [f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
              f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6"]:
        X_train[c] = X_train[c].fillna(0.0)
        X_valid[c] = X_valid[c].fillna(0.0)

    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
    )

    # LGBM Feature Importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    print("Generating recursive forecast for next 3 months...")
    future_months = pd.to_datetime(pd.Index(FORECAST_MONTHS), format="%Y-%m")
    static_cols = KEY_COLS + ohe_cols + num_predictors
    latest_static = (
        df.sort_values(["series_id", DATE_COL])
          .groupby("series_id")
          .tail(1)
          .set_index("series_id")[static_cols]
    )

    work = df.copy()
    all_future_preds = []

    for m in future_months:
        fut = latest_static.reset_index().copy()
        fut[DATE_COL] = m
        fut["month_num"] = m.month
        fut["quarter"]   = pd.Timestamp(m).quarter
        fut["year"]      = m.year

        lags = []
        for sid in fut["series_id"]:
            hist = work.loc[work["series_id"] == sid, [DATE_COL, TARGET_COL]].sort_values(DATE_COL)
            hist_ex = hist.set_index(DATE_COL).sort_index()
            hist_ex = hist_ex.loc[hist_ex.index < m]
            lag1 = hist_ex[TARGET_COL].iloc[-1] if len(hist_ex) >= 1 else np.nan
            lag2 = hist_ex[TARGET_COL].iloc[-2] if len(hist_ex) >= 2 else np.nan
            lag3 = hist_ex[TARGET_COL].iloc[-3] if len(hist_ex) >= 3 else np.nan
            r3  = hist_ex[TARGET_COL].rolling(3).mean().iloc[-1] if len(hist_ex) >= 3 else np.nan
            r6  = hist_ex[TARGET_COL].rolling(6).mean().iloc[-1] if len(hist_ex) >= 6 else np.nan
            lags.append((sid, lag1, lag2, lag3, r3, r6))
            
        lags_df = pd.DataFrame(lags, columns=["series_id",
                                              f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
                                              f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6"])
        fut = fut.merge(lags_df, on="series_id", how="left")

        X_fut = fut[feature_cols].copy()
        for c in [f"{TARGET_COL}_lag1", f"{TARGET_COL}_lag2", f"{TARGET_COL}_lag3",
                  f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6"]:
            X_fut[c] = X_fut[c].fillna(0.0)

        y_hat = final_model.predict(X_fut, num_iteration=final_model.best_iteration_)
        y_hat = np.clip(y_hat, 0, None)

        fut["Forecast Month"] = m.strftime("%Y-%m")
        fut["Forecast"] = y_hat
        all_future_preds.append(fut[["series_id", *KEY_COLS, "Forecast Month", "Forecast"]])

        add_back = fut[[DATE_COL, "series_id"]].copy()
        add_back[TARGET_COL] = y_hat
        work = pd.concat([work, add_back], ignore_index=True)

    forecast_df = pd.concat(all_future_preds, ignore_index=True).sort_values(["Forecast Month", "Channel", "Customer Segment", "Category"])
    
    # Save final forecast
    forecast_df.to_csv(output_dir / "blender_forecast_next3m.csv", index=False)
    print("Pipeline complete. Saved outputs to outputs/ folder.")
    return True

if __name__ == "__main__":
    run_pipeline()
