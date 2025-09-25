# streamlit_app.py — Smart Energy Forecaster (Prophet vs LGBM) + LLM advice + Overlay + LGBM insights
import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

# ------------------------------------------------------------
# Auto dataset source (URL -> local hourly -> upload -> raw UCI)
# ------------------------------------------------------------
GITHUB_DATA_URL = os.getenv(
    "DATA_URL",
    # Example after you push the repo:
    # "https://raw.githubusercontent.com/<user>/<repo>/main/data/hourly_power.csv"
    ""
)

@st.cache_data(show_spinner=False)
def load_hourly_from_url(url: str) -> pd.DataFrame | None:
    if not url:
        return None
    try:
        df = pd.read_csv(url, parse_dates=["datetime"])
        df = df.set_index("datetime")
        if "Global_active_power" not in df.columns:
            return None
        return df[["Global_active_power"]]
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_hourly_local() -> pd.DataFrame | None:
    p = "data/hourly_power.csv"
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime")
        return df[["Global_active_power"]]
    return None

# ---------------- Page / UI ----------------
st.set_page_config(page_title="⚡ Smart Energy Forecaster", page_icon="⚡", layout="wide")
st.title("⚡ Smart Energy Forecaster")
st.caption("Forecast next hours of household energy (kW), detect peaks, and suggest eco-advice. MIT License.")

DEFAULT_LOCAL = "household_power_consumption.txt"   # UCI/Kaggle original (semicolon)
FALLBACK_LOCAL = "household_power_consumption.csv"  # if teammates converted it

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Forecasting model:", ["Prophet", "LightGBM (lags)"], index=0)
    compare_overlay = st.checkbox("Overlay both models (next horizon)", value=False)
    horizon_hours = st.number_input("Forecast horizon (hours)", min_value=6, max_value=72, value=24, step=6)
    peak_quantile = st.slider("Peak threshold (quantile)", 0.60, 0.95, 0.80, 0.01)

    # Prophet-only knobs
    cps = st.slider("Prophet: changepoint prior scale", 0.01, 0.50, 0.10, 0.01)
    show_components = st.checkbox("Show Prophet components", value=False)

    # LGBM insights
    show_lgbm_insights = st.checkbox("Show LightGBM insights", value=False)

    st.markdown("---")
    st.subheader("Advice settings")
    use_llm = st.checkbox("Use LLM to rewrite tips (Mistral via HF Inference)", value=False)
    advice_lang = st.selectbox("Advice language", ["en", "my", "fr"], index=0)

    st.markdown("---")
    st.subheader("Upload data (optional)")
    up = st.file_uploader(
        "UCI/Kaggle file (semicolon CSV/TXT). Columns: Date;Time;Global_active_power;…",
        type=["csv", "txt"]
    )

# ---------------- Data loading / prep ----------------
@st.cache_data(show_spinner=False)
def load_uci_like(file: io.BytesIO | str) -> pd.DataFrame:
    """Parse raw UCI/Kaggle minute data -> hourly mean single-column frame."""
    df = pd.read_csv(
        file,
        sep=";",
        na_values=["?", "nan", ""],
        parse_dates={"datetime": ["Date", "Time"]},
        infer_datetime_format=True,
        low_memory=False,  # keep C engine
    )
    df = df.set_index("datetime")
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    hourly = df["Global_active_power"].resample("1H").mean()
    hourly = hourly.fillna(method="ffill", limit=3).dropna()

    q_low, q_hi = hourly.quantile([0.001, 0.999])  # mild outlier clip
    hourly = hourly.clip(q_low, q_hi)
    return hourly.to_frame()

def try_autoload_local_raw() -> pd.DataFrame | None:
    if os.path.exists(DEFAULT_LOCAL):
        return load_uci_like(DEFAULT_LOCAL)
    if os.path.exists(FALLBACK_LOCAL):
        return load_uci_like(FALLBACK_LOCAL)
    return None

# -------- Preferred order: URL -> local hourly -> uploaded -> local raw UCI --------
data_hourly = load_hourly_from_url(GITHUB_DATA_URL)
if data_hourly is None:
    data_hourly = load_hourly_local()
if data_hourly is None and up is not None:
    data_hourly = load_uci_like(up)
if data_hourly is None:
    data_hourly = try_autoload_local_raw()
if data_hourly is None:
    st.warning("No dataset found. Provide DATA_URL env var, commit data/hourly_power.csv, or upload a file.")
    st.stop()

st.success(
    f"Data loaded: {len(data_hourly):,} hourly points "
    f"from {data_hourly.index.min().date()} to {data_hourly.index.max().date()}."
)

@st.cache_data(show_spinner=False)
def make_train_test(df_hourly: pd.DataFrame, test_hours: int = 24*7):
    if len(df_hourly) <= test_hours + 48:
        test_hours = max(24, int(len(df_hourly) * 0.15))
    train = df_hourly.iloc[:-test_hours]
    test = df_hourly.iloc[-test_hours:]
    return train, test

train_df, test_df = make_train_test(data_hourly, test_hours=24*7)
y_train = train_df["Global_active_power"]
y_test  = test_df["Global_active_power"]

# ---------------- Prophet helpers ----------------
def to_prophet(df_onecol: pd.DataFrame) -> pd.DataFrame:
    return df_onecol.reset_index().rename(columns={"datetime": "ds", "Global_active_power": "y"})

@st.cache_resource(show_spinner=False)
def fit_prophet(train_prophet: pd.DataFrame, changepoint_prior_scale: float):
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    m.fit(train_prophet)
    return m

def prophet_backtest_and_future(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon_hours: int, cps: float):
    """Backtest on train->test; then fit on full data for the live next-horizon forecast."""
    # Backtest
    train_p, test_p = to_prophet(train_df), to_prophet(test_df)
    m_train = fit_prophet(train_p, cps)
    horizon_for_test = len(test_p)
    future_all_train = m_train.make_future_dataframe(periods=horizon_for_test, freq="H")
    fcst_all_train = m_train.predict(future_all_train)
    fcst_test = fcst_all_train.set_index("ds").loc[test_p["ds"]][["yhat"]].rename(columns={"yhat": "pred"})
    y_true = test_p.set_index("ds")["y"]
    mae = float(mean_absolute_error(y_true, fcst_test["pred"]))
    mape = float((np.abs((y_true - fcst_test["pred"]) / np.clip(y_true, 1e-6, None))).mean() * 100.0)

    # Live forecast after last timestamp (fit on FULL data)
    full_df = pd.concat([train_df, test_df])
    full_p = to_prophet(full_df)
    model_full = fit_prophet(full_p, cps)
    fcst_all_full = model_full.predict(full_p[["ds"]])
    future_full = model_full.make_future_dataframe(periods=horizon_hours, freq="H")
    fcst_future = model_full.predict(future_full).tail(horizon_hours)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return model_full, fcst_all_full, fcst_test, mae, mape, fcst_future

# ---------------- LightGBM (lags) helpers ----------------
@st.cache_data(show_spinner=False)
def build_lagged_frame(
    series_hourly: pd.Series,
    max_lag: int = 24,
    add_rolling: bool = True,
    rolling_windows: tuple[int, ...] = (24, 168),
    add_calendar: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": series_hourly.astype(float)})
    for l in range(1, max_lag + 1):
        df[f"lag_{l}"] = df["y"].shift(l)
    if add_rolling:
        for w in rolling_windows:
            df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(window=w, min_periods=int(w*0.6)).mean()
            df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(window=w, min_periods=int(w*0.6)).std()
    if add_calendar:
        idx = series_hourly.index
        df["hour"] = idx.hour
        df["dow"] = idx.dayofweek
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df.dropna()

def lgbm_backtest_and_future(train_series: pd.Series, test_series: pd.Series, horizon_hours: int, max_lag: int = 24):
    full_series = pd.concat([train_series, test_series])
    feat_full = build_lagged_frame(full_series, max_lag=max_lag)

    split_time = test_series.index[0]
    feat_train = feat_full.loc[feat_full.index < split_time].copy()
    feat_test  = feat_full.loc[feat_full.index >= split_time].copy()

    X_train = feat_train.drop(columns=["y"]); y_train = feat_train["y"]
    X_test  = feat_test.drop(columns=["y"]);  y_test  = feat_test["y"]

    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=-1, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test_pred = pd.Series(model.predict(X_test), index=X_test.index, name="pred")
    mae = float(mean_absolute_error(y_test, test_pred))
    mape = float((np.abs((y_test - test_pred) / np.clip(y_test, 1e-6, None))).mean() * 100.0)
    lgbm_test = pd.DataFrame(test_pred)

    # Recursive next-horizon forecast
    freq = pd.infer_freq(full_series.index) or "H"
    last_time = full_series.index[-1]
    current_series = full_series.copy()
    future_times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=horizon_hours, freq=freq)
    preds = []
    for t in future_times:
        row = build_lagged_frame(current_series, max_lag=max_lag).iloc[-1:].drop(columns=["y"])
        yhat = float(model.predict(row)[0])
        preds.append((t, yhat))
        current_series = pd.concat([current_series, pd.Series([yhat], index=[t])])
    future_df = pd.DataFrame(preds, columns=["ds", "yhat"])
    return model, X_train.columns.tolist(), lgbm_test, mae, mape, future_df

# ---------------- Peaks & Advice ----------------
def detect_peaks(fcst_future: pd.DataFrame, q: float = 0.80, top_n: int = 3):
    thr = fcst_future["yhat"].quantile(q)
    peaks = fcst_future[fcst_future["yhat"] >= thr].copy().sort_values("yhat", ascending=False).head(top_n)
    return float(thr), peaks

def rule_based_advice(fcst_future: pd.DataFrame, thr_value: float, history_hourly: pd.DataFrame):
    tips = []
    soon = fcst_future.iloc[:3]
    if (soon["yhat"] >= thr_value).any():
        tips.append("Delay high-load chores (laundry/dishwasher) by ~2–3 hours to avoid the upcoming peak.")
    overnight = fcst_future[fcst_future["ds"].dt.hour.isin([0,1,2,3,4,5])]
    if not overnight.empty and overnight["yhat"].median() <= 0.7 * fcst_future["yhat"].median():
        tips.append("Schedule appliances after 22:00 or early morning to leverage off-peak hours.")
    past_med = float(history_hourly["Global_active_power"].tail(24*7).median()) if len(history_hourly) >= 24*7 else float(history_hourly.median())
    fut_med  = float(fcst_future["yhat"].median())
    if fut_med > 1.2 * past_med:
        tips.append("Baseline looks elevated — unplug idle chargers or use smart plugs for standby devices.")
    if not tips:
        tips.append("No major peaks detected — keep using off-peak hours for heavy loads.")
    return tips

# ---------------- LLM advice (HF Inference API) ----------------
def format_advice_json(peaks_df: pd.DataFrame, future_df: pd.DataFrame, tips: list[str]) -> dict:
    peak_hours = [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M") for t in peaks_df["ds"]] if not peaks_df.empty else []
    return {
        "peak_hours": peak_hours,
        "top_peak_kw": float(peaks_df["yhat"].max()) if not peaks_df.empty else None,
        "low_kw": float(future_df["yhat"].min()) if not future_df.empty else None,
        "median_kw": float(future_df["yhat"].median()) if not future_df.empty else None,
        "rule_based_tips": tips[:4],
    }

def llm_paraphrase_advice(summary: dict, language: str = "en") -> list[str]:
    hf_token = os.getenv("HF_TOKEN")
    model_id = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    if not hf_token:
        return summary.get("rule_based_tips", [])
    prompt = (
        "You are an energy-efficiency assistant. Rewrite the following JSON into 2–3 concise, user-friendly "
        "energy-saving tips for the next 24 hours. Be specific with times if peak_hours exist. "
        "Avoid technical jargon. Output plain bullet points only.\n\n"
        f"LANGUAGE: {language}\n"
        f"DATA:\n{json.dumps(summary, ensure_ascii=False)}\n\n"
        "OUTPUT:\n- "
    )
    endpoint = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 160, "temperature": 0.4, "return_full_text": False}}
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        text = r.json()[0]["generated_text"].strip()
        bullets = [line.strip("-• ").strip() for line in text.split("\n") if line.strip()]
        return [b for b in bullets if b][:3] or summary.get("rule_based_tips", [])
    except Exception:
        return summary.get("rule_based_tips", [])

# ---------------- Run chosen model ----------------
if model_choice == "Prophet":
    with st.spinner("Training Prophet…"):
        model_p, fcst_all_p, fcst_test_p, mae, mape, future_df = prophet_backtest_and_future(
            train_df, test_df, horizon_hours, cps
        )
    chosen_name = "Prophet"
    has_uncertainty = True
    model_lgbm = None
    lgbm_feat_names = None
else:
    with st.spinner("Training LightGBM (lags)…"):
        model_lgbm, lgbm_feat_names, lgbm_test, mae, mape, lgbm_future = lgbm_backtest_and_future(
            y_train, y_test, horizon_hours, max_lag=24
        )
    chosen_name = "LightGBM (lags)"
    future_df = lgbm_future[["ds", "yhat"]].copy()
    has_uncertainty = False

# ---------------- Optional overlay ----------------
overlay_df = None
overlay_label = None
if compare_overlay:
    if model_choice == "Prophet":
        _, _, _, _, _, lgbm_future_overlay = lgbm_backtest_and_future(y_train, y_test, horizon_hours, max_lag=24)
        overlay_df = lgbm_future_overlay.rename(columns={"yhat": "yhat_overlay"})
        overlay_label = "LightGBM overlay"
    else:
        model_full_overlay, _, _, _, _, fcst_future_overlay = prophet_backtest_and_future(train_df, test_df, horizon_hours, cps)
        overlay_df = fcst_future_overlay.rename(columns={"yhat": "yhat_overlay"})
        overlay_label = "Prophet overlay"

# ---------------- Metrics / peaks / advice ----------------
k1, k2, k3 = st.columns(3)
k1.metric("Validation MAE (kW)", f"{mae:.3f}")
k2.metric("Validation MAPE (%)", f"{mape:.1f}")
k3.metric("Peak threshold (quantile)", f"{peak_quantile:.2f}")

thr, peak_rows = detect_peaks(future_df, q=peak_quantile, top_n=3)
advice = rule_based_advice(future_df, thr, data_hourly)
summary = format_advice_json(peak_rows, future_df, advice)
final_tips = llm_paraphrase_advice(summary, advice_lang) if use_llm else advice

# ---------------- Plot ----------------
st.subheader(f"History (last 7 days) + Next {horizon_hours}h Forecast ({chosen_name})")
hist_tail = data_hourly.tail(24*7).copy()
fig = go.Figure()

# History
fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail["Global_active_power"], mode="lines", name="History (kW)"))

# Forecast line
fig.add_trace(go.Scatter(x=future_df["ds"], y=future_df["yhat"], mode="lines+markers",
                         name=f"{chosen_name} forecast (kW)"))

# Uncertainty (Prophet only)
if has_uncertainty and set(["yhat_lower", "yhat_upper"]).issubset(future_df.columns):
    fig.add_trace(go.Scatter(
        x=pd.concat([future_df["ds"], future_df["ds"][::-1]]),
        y=pd.concat([future_df["yhat_upper"], future_df["yhat_lower"][::-1]]),
        fill="toself", line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty", opacity=0.2, showlegend=True
    ))

# Overlay line
if overlay_df is not None:
    fig.add_trace(go.Scatter(x=overlay_df["ds"], y=overlay_df["yhat_overlay"],
                             mode="lines", name=overlay_label))

# Peak markers
if not peak_rows.empty:
    fig.add_trace(go.Scatter(x=peak_rows["ds"], y=peak_rows["yhat"], mode="markers",
                             marker=dict(size=10, symbol="diamond"), name="Detected peaks"))

fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Time", yaxis_title="Power (kW)")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Tables / Advice ----------------
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Detected Peak Hours")
    if peak_rows.empty:
        st.info("No strong peaks detected in the forecast window.")
    else:
        display_peaks = peak_rows.copy()
        display_peaks["Hour"] = display_peaks["ds"].dt.strftime("%Y-%m-%d %H:%M")
        display_peaks = display_peaks[["Hour", "yhat"]].rename(columns={"yhat": "Forecast (kW)"})
        st.dataframe(display_peaks, use_container_width=True)

with c2:
    st.subheader("💡 Eco-Advice")
    for tip in final_tips:
        st.success(tip)
    if use_llm and not os.getenv("HF_TOKEN"):
        st.info("LLM disabled: set HF_TOKEN (and optional MODEL_ID) in environment to enable Mistral paraphrasing.")

# ---------------- Components / Insights ----------------
if model_choice == "Prophet" and show_components:
    st.subheader("Prophet Components")
    comp_fig = model_p.plot_components(fcst_all_p)  # matplotlib fig from cached fit
    st.pyplot(comp_fig, clear_figure=True)

if model_choice == "LightGBM (lags)" and show_lgbm_insights and model_lgbm is not None:
    st.subheader("LightGBM Insights")
    # Feature importances
    importances = pd.DataFrame({
        "feature": lgbm_feat_names,
        "importance": model_lgbm.feature_importances_.astype(float)
    }).sort_values("importance", ascending=False).head(12)
    fig_imp = px.bar(importances, x="importance", y="feature", orientation="h", title="Top Feature Importances")
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

    # Partial dependence for hour & day-of-week (if available)
    feat_full = build_lagged_frame(pd.concat([y_train, y_test]), max_lag=24)
    X_last = feat_full.drop(columns=["y"]).iloc[-1:].copy()

    def pdp_for(feature: str, values: list[int]) -> pd.DataFrame | None:
        if feature not in X_last.columns:
            return None
        rows = []
        for v in values:
            x = X_last.copy()
            x[feature] = v
            yhat = float(model_lgbm.predict(x)[0])
            rows.append((v, yhat))
        return pd.DataFrame(rows, columns=[feature, "yhat"])

    # Hour-of-day PDP
    df_hour = pdp_for("hour", list(range(24)))
    if df_hour is not None:
        fig_h = px.line(df_hour, x="hour", y="yhat", markers=True, title="Model response vs Hour of day")
        st.plotly_chart(fig_h, use_container_width=True)

    # Day-of-week PDP (0=Mon)
    df_dow = pdp_for("dow", list(range(7)))
    if df_dow is not None:
        fig_d = px.line(df_dow, x="dow", y="yhat", markers=True, title="Model response vs Day of week (0=Mon)")
        st.plotly_chart(fig_d, use_container_width=True)

st.caption("Notes: Forecasts are indicative; accuracy depends on data quality and household usage patterns.")
st.caption("Data source: UCI Machine Learning Repository — Household Power Consumption Dataset.")
st.caption("Developed by Seon for FTL Machine Learning Bootcamp for Myanmar by UNDP.")
