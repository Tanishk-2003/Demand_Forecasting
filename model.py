# --- CONFIG ---
CSV_PATH = "sales_data.csv"  # <-- your data path
TARGET_COL = "Quantity"                   # or "Tax Base Amount"
DATE_COL = "Month"
KEY_COLS = ["Channel", "Category", "Customer Segment"]
FORECAST_MONTHS = ["2025-04", "2025-05", "2025-06"]     # final forecast
CV_LAST_N_ORIGINS = 6    # number of rolling CV origins (each origin predicts 3 months)
HORIZON = 3              # multi-step horizon

# --- LIBS ---
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

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

# --- LOAD & PREP ---
df = pd.read_csv("sales_data.csv")
df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d-%b-%Y")
df.sort_values([*KEY_COLS, DATE_COL], inplace=True)

# Clean numeric cols
num_cols_raw = ["ASP","Customers","Cities Billed","Invoices","Quantity"]
for c in num_cols_raw:
    if c in df.columns:
        df[c] = (df[c].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False))
        df[c] = pd.to_numeric(df[c], errors="coerce")

# IDs & features
df["series_id"] = df[KEY_COLS].astype(str).agg("|".join, axis=1)
ohe_cols = [c for c in df.columns if c.startswith("OHE_")]
num_predictors = [c for c in num_cols_raw if c != TARGET_COL]

# Lags/rolls
df = df.groupby("series_id", group_keys=False).apply(
    lambda x: add_lags_rolls(x, target=TARGET_COL)
)


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
       f"{TARGET_COL}_rmean3", f"{TARGET_COL}_rmean6", "month_num","quarter","year"]
)

# --------- MULTI-STEP ROLLING BACKTEST ---------
def multistep_rolling_backtest_lgbm(
    data, feature_cols, target_col, date_col, key_cols,
    horizon=3, last_n_origins=6, lgb_params=None, early_stopping_rounds=300
):
    if lgb_params is None:
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
            random_state=42
        )

    months_sorted = sorted(data[date_col].drop_duplicates().tolist())
    # All origin months such that origin..origin+horizon-1 exist in data
    valid_origins = []
    month_to_idx = {m:i for i,m in enumerate(months_sorted)}
    for i, m in enumerate(months_sorted):
        if i + horizon - 1 < len(months_sorted):
            valid_origins.append(m)
    # Keep only the last_n_origins from the end
    origins = valid_origins[-last_n_origins:] if last_n_origins <= len(valid_origins) else valid_origins

    rows_all = []  # store predictions with horizons

    for origin in origins:
        # Train on all data strictly before origin
        train_df = data[data[date_col] < origin].copy()
        if train_df.empty:
            continue

        # inner val: last month of the training period
        split_cut = train_df[date_col].max() - pd.offsets.MonthBegin(0)
        X_tr = train_df[feature_cols].copy()
        y_tr = train_df[target_col].copy()

        # simple inner val = last ~1 month of train (fallback to tail 200)
        val_mask = train_df[date_col] == train_df[date_col].max()
        if val_mask.sum() < 50:
            X_tr_in, y_tr_in = X_tr, y_tr
            X_val, y_val = X_tr.tail(min(200, len(X_tr))), y_tr.tail(min(200, len(y_tr)))
        else:
            X_tr_in, y_tr_in = X_tr[~val_mask], y_tr[~val_mask]
            X_val,   y_val   = X_tr[val_mask],  y_tr[val_mask]

        # fill NA lags
        for X_ in (X_tr_in, X_val):
            for c in [f"{target_col}_lag1", f"{target_col}_lag2", f"{target_col}_lag3",
                      f"{target_col}_rmean3", f"{target_col}_rmean6"]:
                if c in X_.columns:
                    X_[c] = X_[c].fillna(0.0)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr_in,
            y_tr_in,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        )


        # Prepare recursive forecasting starting from origin for H steps
        # Latest static attrs before origin
        static_cols = key_cols + [c for c in feature_cols if c.startswith("OHE_")] + [c for c in num_predictors]
        latest_static = (
            data[data[date_col] < origin]
            .sort_values(["series_id", date_col])
            .groupby("series_id")
            .tail(1)
            .set_index("series_id")[static_cols]
        )

        # Working frame that contains all actuals up to origin-1
        work = data[data[date_col] < origin][[date_col, "series_id", target_col]].copy()

        for h in range(horizon):
            m = months_sorted[month_to_idx[origin] + h]  # month to evaluate (exists in data)
            # Build future features for all known series
            fut = latest_static.reset_index().copy()
            fut[date_col] = m
            fut["month_num"] = m.month
            fut["quarter"]   = pd.Timestamp(m).quarter
            fut["year"]      = m.year

            # Lags from 'work'
            lags = []
            for sid in fut["series_id"]:
                hist = work.loc[work["series_id"] == sid, [date_col, target_col]].sort_values(date_col)
                hist_ex = hist.set_index(date_col).sort_index()
                # compute lags prior to month m
                hist_ex = hist_ex.loc[hist_ex.index < m]
                lag1 = hist_ex[target_col].iloc[-1] if len(hist_ex)>=1 else np.nan
                lag2 = hist_ex[target_col].iloc[-2] if len(hist_ex)>=2 else np.nan
                lag3 = hist_ex[target_col].iloc[-3] if len(hist_ex)>=3 else np.nan
                r3  = hist_ex[target_col].rolling(3).mean().iloc[-1] if len(hist_ex)>=3 else np.nan
                r6  = hist_ex[target_col].rolling(6).mean().iloc[-1] if len(hist_ex)>=6 else np.nan
                lags.append((sid, lag1, lag2, lag3, r3, r6))
            lags_df = pd.DataFrame(lags, columns=["series_id",
                                                  f"{target_col}_lag1", f"{target_col}_lag2", f"{target_col}_lag3",
                                                  f"{target_col}_rmean3", f"{target_col}_rmean6"])
            fut = fut.merge(lags_df, on="series_id", how="left")

            X_fut = fut[feature_cols].copy()
            for c in [f"{target_col}_lag1", f"{target_col}_lag2", f"{target_col}_lag3",
                      f"{target_col}_rmean3", f"{target_col}_rmean6"]:
                X_fut[c] = X_fut[c].fillna(0.0)

            y_hat = model.predict(X_fut, num_iteration=model.best_iteration_)
            y_hat = np.clip(y_hat, 0, None)

            # actuals for this month (if exist)
            actual_df = data[data[date_col] == m][["series_id", target_col, *key_cols]].copy()
            out = fut[["series_id", *key_cols]].copy()
            out[date_col] = m
            out["origin_month"] = origin
            out["horizon"] = h+1
            out["pred"] = y_hat

            out = out.merge(actual_df.rename(columns={target_col:"actual"}), on=["series_id", *key_cols], how="left")
            rows_all.append(out)

            # append predictions to work for next step lags
            add_back = fut[[date_col, "series_id"]].copy()
            add_back[target_col] = y_hat
            work = pd.concat([work, add_back], ignore_index=True)

    cv_rows = pd.concat(rows_all, ignore_index=True)
    # Drop rows where actual is NaN (outside available data)
    cv_rows = cv_rows.dropna(subset=["actual"])

    # Metrics
    cv_rows["abs_err"] = (cv_rows["actual"] - cv_rows["pred"]).abs()
    cv_rows["APE"] = cv_rows["abs_err"] / np.maximum(cv_rows["actual"].abs(), 1e-6)

    overall = cv_rows.groupby("horizon").apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        ))
    ).reset_index().rename(columns={"horizon":"H"})

    overall_all = pd.DataFrame({
        "H":["ALL"],
        "MAE":[cv_rows["abs_err"].mean()],
        "MAPE":[mape(cv_rows["actual"], cv_rows["pred"])],
        "sMAPE":[smape(cv_rows["actual"], cv_rows["pred"])]
    })

    by_channel = cv_rows.groupby(["horizon","Channel"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        ))
    ).reset_index().rename(columns={"horizon":"H"})

    by_segment = cv_rows.groupby(["horizon","Customer Segment"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        ))
    ).reset_index().rename(columns={"horizon":"H"})

    by_category = cv_rows.groupby(["horizon","Category"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        ))
    ).reset_index().rename(columns={"horizon":"H"})

    by_series = cv_rows.groupby(["horizon","series_id"]).apply(
        lambda g: pd.Series(dict(
            MAE=g["abs_err"].mean(),
            MAPE=mape(g["actual"], g["pred"]),
            sMAPE=smape(g["actual"], g["pred"])
        ))
    ).reset_index().rename(columns={"horizon":"H"}).sort_values(["H","MAPE"])

    # Save
    out_dir = Path("cv_outputs_multistep")
    out_dir.mkdir(exist_ok=True)
    cv_rows.to_csv(out_dir / "cv_rows_multistep.csv", index=False)
    overall.to_csv(out_dir / "metrics_overall_byH.csv", index=False)
    overall_all.to_csv(out_dir / "metrics_overall_ALL.csv", index=False)
    by_channel.to_csv(out_dir / "metrics_by_channel_byH.csv", index=False)
    by_segment.to_csv(out_dir / "metrics_by_segment_byH.csv", index=False)
    by_category.to_csv(out_dir / "metrics_by_category_byH.csv", index=False)
    by_series.to_csv(out_dir / "metrics_by_series_byH.csv", index=False)

    print("Multi-step CV complete. Per-horizon metrics:")
    print(overall)
    return cv_rows, overall, by_channel, by_segment, by_category, by_series

# ---- Run multi-step CV (3-month horizon) on modeling frame ----
cv_rows, cv_overall, cv_ch, cv_seg, cv_cat, cv_series = multistep_rolling_backtest_lgbm(
    df_model, feature_cols, TARGET_COL, DATE_COL, KEY_COLS,
    horizon=HORIZON, last_n_origins=CV_LAST_N_ORIGINS
)

# ---- Train FINAL model on all data up to Mar-2025 and forecast Apr–Jun 2025 ----
from sklearn.metrics import mean_absolute_error

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

final_params = dict(
    objective="poisson",
    boosting_type="gbdt",
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    callbacks=[
        lgb.early_stopping(stopping_rounds=300),
        lgb.log_evaluation(0)
    ]
)

print("Final fit valid MAE:", mean_absolute_error(y_valid, final_model.predict(X_valid, num_iteration=final_model.best_iteration_)))

# ---- Recursive forecast for Apr–Jun 2025 ----
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
        lag1 = hist_ex[TARGET_COL].iloc[-1] if len(hist_ex)>=1 else np.nan
        lag2 = hist_ex[TARGET_COL].iloc[-2] if len(hist_ex)>=2 else np.nan
        lag3 = hist_ex[TARGET_COL].iloc[-3] if len(hist_ex)>=3 else np.nan
        r3  = hist_ex[TARGET_COL].rolling(3).mean().iloc[-1] if len(hist_ex)>=3 else np.nan
        r6  = hist_ex[TARGET_COL].rolling(6).mean().iloc[-1] if len(hist_ex)>=6 else np.nan
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

forecast_df = pd.concat(all_future_preds, ignore_index=True).sort_values(["Forecast Month","Channel","Customer Segment","Category"])
forecast_df.to_csv("blender_forecast_next3m.csv", index=False)
print("Saved: blender_forecast_next3m.csv")
