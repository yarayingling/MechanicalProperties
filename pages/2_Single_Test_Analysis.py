# pages/2_Single_Test_Analysis.py
# Single Test Analysis (robust E, 0.2% offset yield, UTS, toughness, CIs)
import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import bootstrap
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, LinearRegression
import plotly.graph_objects as go

st.title("Single Test Analysis")

# -------------------------
# Guards & load session data
# -------------------------
if "trimmed_df" not in st.session_state:
    st.info("Go to **Upload & QA/QC** first to load data.")
    st.stop()

df = st.session_state["trimmed_df"].copy()

# Expect columns eng_strain & eng_stress (created on Upload page)
if not {"eng_strain", "eng_stress"}.issubset(df.columns):
    st.error("Expected columns 'eng_strain' and 'eng_stress' not found.")
    st.stop()

# Clean data (finite only)
mfin = np.isfinite(df["eng_strain"].values) & np.isfinite(df["eng_stress"].values)
df = df.loc[mfin].copy()

# Ensure nondecreasing strain to avoid negative integration steps
df["eng_strain"] = np.maximum.accumulate(df["eng_strain"].values)

x_eng = df["eng_strain"].values.astype(float)
y_eng = df["eng_stress"].values.astype(float)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Analysis settings")

curve_type = st.sidebar.selectbox("Curve type", ["Engineering", "True"], index=0,
                                  help="True stress/strain uses ln(1+ε) and σ_true = σ_eng(1+ε).")
smooth = st.sidebar.toggle("Savitzky–Golay smoothing", value=True)
win = st.sidebar.slider("Savgol window", 7, 301, 41, step=2)
poly = st.sidebar.slider("Savgol polyorder", 1, 5, 3)

elastic_max = st.sidebar.slider("Elastic fit upper strain ε_max (engineering scale)", 0.002, 0.05, 0.02, step=0.001,
                                help="Upper bound of linear elastic region for modulus fit.")
offset = st.sidebar.number_input("Yield offset ε₀ (engineering scale)", 0.000, 0.050, 0.002, step=0.001, format="%.3f",
                                 help="0.002 = 0.2% proof stress (engineering).")

do_boot = st.sidebar.toggle("Compute bootstrap CIs", value=True,
                            help="95% confidence intervals via SciPy bootstrap (paired resampling).")
B = st.sidebar.slider("Bootstrap resamples", 200, 5000, 1000, step=100)

# -------------------------
# Choose working curve (engineering vs true)
# -------------------------
if curve_type == "True":
    # True strain/stress approximations
    x_work = np.log1p(x_eng)                  # ε_true = ln(1+ε_eng)
    y_work = y_eng * (1.0 + x_eng)            # σ_true ≈ σ_eng (1+ε_eng) up to necking
else:
    x_work = x_eng.copy()
    y_work = y_eng.copy()

# Optional smoothing on the working stress
if smooth:
    try:
        y_s = signal.savgol_filter(y_work, window_length=min(win, (len(y_work) // 2) * 2 - 1 if len(y_work) > 5 else win), polyorder=poly)
    except Exception:
        y_s = y_work.copy()
else:
    y_s = y_work.copy()

# -------------------------
# Elastic modulus (robust) — evaluated in the *engineering* small-strain window
# -------------------------
# Elastic window is defined in engineering strain to match lab practice
mask_el = x_eng <= float(elastic_max)
if mask_el.sum() < 5:
    st.error("Not enough points in the elastic window. Increase ε_max or check data.")
    st.stop()

# For modulus, use the same curve_type selected (small difference near origin)
X_el = (x_work[mask_el]).reshape(-1, 1)
Y_el = (y_s[mask_el])

E = None
intercept = 0.0
try:
    ransac = RANSACRegressor(LinearRegression(),
                             residual_threshold=max(1e-6, float(np.nanstd(Y_el)) * 0.5),
                             random_state=0)
    ransac.fit(X_el, Y_el)
    E = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)
except Exception:
    pass

if E is None or not np.isfinite(E):
    try:
        ts = TheilSenRegressor().fit(X_el, Y_el)
        E = float(ts.coef_[0]); intercept = float(ts.intercept_)
    except Exception:
        slope, intercept = np.polyfit(X_el.ravel(), Y_el, 1)
        E = float(slope)

# -------------------------
# Yield (0.2% offset), UTS, Toughness
# -------------------------
# Yield proof stress defined in *engineering* space: σ = E (ε_eng - ε0) (+ intercept if nonzero)
# We need an offset line on the *working* x-scale to intersect with y_s.
# Map engineering offset line to working x-scale:
# - If Engineering curve: x_work == x_eng → use directly.
# - If True curve: we still use ε_eng for offset, but the x_work is ln(1+ε_eng); we build yline in y-space only.

# Build offset line in terms of the *engineering* strain grid, then evaluate over x_work grid:
yline_eng = E * (x_eng - offset) + intercept

# To compare with y_s which is in working stress space:
# If curve is Engineering, yline_work == yline_eng
# If curve is True, map engineering offset line to true stress: σ_true ≈ σ_eng (1+ε_eng)
if curve_type == "True":
    yline_work = yline_eng * (1.0 + x_eng)
else:
    yline_work = yline_eng

# Intersection (closest point)
iy = int(np.argmin(np.abs(y_s - yline_work)))
sigma_y = float(y_s[iy]); eps_y_work = float(x_work[iy])  # in working scale
eps_y_eng = float(x_eng[iy])                              # also record engineering yield strain

# UTS on working curve
iuts = int(np.argmax(y_s))
uts = float(y_s[iuts]); eps_uts_work = float(x_work[iuts]); eps_uts_eng = float(x_eng[iuts])

# Toughness = area under selected curve (working stress vs working strain)
# Clean NaNs and enforce nondecreasing strain
mvalid = np.isfinite(x_work) & np.isfinite(y_s)
x_c = np.maximum.accumulate(x_work[mvalid])  # monotonic
y_c = y_s[mvalid]
toughness = float(np.trapz(y_c, x_c))  # units: (MPa * strain_unit); for True/Engineering both are "MPa·strain"

# -------------------------
# Bootstrap CIs (SciPy) — paired resampling
# -------------------------
ciE_lo = ciE_hi = np.nan
ciY_lo = ciY_hi = np.nan

def _safe_slope(xa, ya):
    if len(xa) < 5 or np.allclose(xa.min(), xa.max()):
        return np.nan
    s, _ = np.polyfit(xa, ya, 1)
    return float(s)

if do_boot:
    # E CI in elastic window on working curve
    resE = bootstrap(
        data=(X_el.ravel(), Y_el),
        statistic=lambda a, b: _safe_slope(np.asarray(a), np.asarray(b)),
        paired=True,
        vectorized=False,
        confidence_level=0.95,
        n_resamples=B,
        method="BCa",
        random_state=0,
    )
    ciE_lo, ciE_hi = float(resE.confidence_interval.low), float(resE.confidence_interval.high)

    # σy CI: recompute slope in elastic window for each resample and re-find intersection
    def stat_sigma_y(x_all, y_all, x_eng_all, E_off=offset):
        x_all = np.asarray(x_all); y_all = np.asarray(y_all); x_eng_all = np.asarray(x_eng_all)
        m_el = x_eng_all <= elastic_max
        if m_el.sum() < 5:
            return np.nan
        slope, intercept_b = np.polyfit(x_all[m_el], y_all[m_el], 1)
        yline_eng_b = slope * (x_eng_all - E_off) + intercept_b
        if curve_type == "True":
            yline_work_b = yline_eng_b * (1.0 + x_eng_all)
        else:
            yline_work_b = yline_eng_b
        idx = np.argmin(np.abs(y_all - yline_work_b))
        return float(y_all[idx])

    resY = bootstrap(
        data=(x_work, y_s, x_eng),
        statistic=lambda a, b, c: stat_sigma_y(a, b, c, E_off=offset),
        paired=True,
        vectorized=False,
        confidence_level=0.95,
        n_resamples=B,
        method="BCa",
        random_state=1,
    )
    ciY_lo, ciY_hi = float(resY.confidence_interval.low), float(resY.confidence_interval.high)

# -------------------------
# Metrics UI
# -------------------------
st.subheader("Key metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"E ({'MPa' if curve_type=='Engineering' else 'MPa (true)'} )",
          f"{E:,.0f}", f"95% CI: {ciE_lo:,.0f}–{ciE_hi:,.0f}" if do_boot else None)
m2.metric("σᵧ (MPa)", f"{sigma_y:,.0f}", f"{offset*100:.1f}% offset (eng.)")
m3.metric("UTS (MPa)", f"{uts:,.0f}")
m4.metric("Toughness (MPa·strain)", f"{toughness:,.0f}")
st.caption(f"Yield strain: εᵧ (eng.) = {eps_y_eng:.4f} | εᵧ ({curve_type.lower()}) = {eps_y_work:.4f}")

# -------------------------
# Plots
# -------------------------
fig = go.Figure()
# Always show engineering raw for reference?
if curve_type == "Engineering":
    fig.add_scatter(x=x_work, y=y_work, name="Raw", mode="lines")
else:
    fig.add_scatter(x=x_work, y=y_work, name="True (raw)", mode="lines")
if smooth:
    fig.add_scatter(x=x_work, y=y_s, name="Smoothed", mode="lines")
fig.add_scatter(x=x_work, y=yline_work, name=f"Offset line ({offset*100:.1f}%, eng.)", mode="lines")
fig.add_scatter(x=[eps_y_work], y=[sigma_y], name="Yield", mode="markers")
fig.add_scatter(x=[eps_uts_work], y=[uts], name="UTS", mode="markers")
fig.update_layout(
    xaxis_title=f"{'True' if curve_type=='True' else 'Engineering'} strain (–)",
    yaxis_title=f"{'True' if curve_type=='True' else 'Engineering'} stress (MPa)",
    legend=dict(orientation="h"),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

# Diagnostics expander
with st.expander("Diagnostics: dσ/dε and elastic residuals"):
    dsdE = np.gradient(y_s, x_work, edge_order=2)
    fig_d = go.Figure()
    fig_d.add_scatter(x=x_work, y=dsdE, mode="lines", name="dσ/dε")
    fig_d.update_layout(xaxis_title="Strain", yaxis_title="dσ/dε (MPa)", height=300)
    st.plotly_chart(fig_d, use_container_width=True)

    y_fit_el = E * X_el.ravel() + intercept
    res = Y_el - y_fit_el
    fig_r = go.Figure()
    fig_r.add_scatter(x=X_el.ravel(), y=res, mode="markers", name="Residuals (elastic)")
    fig_r.update_layout(xaxis_title="Strain (elastic window)", yaxis_title="Residual (MPa)", height=300)
    st.plotly_chart(fig_r, use_container_width=True)

# -------------------------
# Downloads
# -------------------------
out_metrics = pd.DataFrame([{
    "curve_type": curve_type.lower(),
    "E_MPa": E,
    "E_CI_low": ciE_lo, "E_CI_high": ciE_hi,
    "sigma_y_MPa": sigma_y,
    "sigma_y_CI_low": ciY_lo, "sigma_y_CI_high": ciY_hi,
    "UTS_MPa": uts,
    "eps_y_eng": eps_y_eng, "eps_y_work": eps_y_work,
    "eps_UTS_eng": eps_uts_eng, "eps_UTS_work": eps_uts_work,
    "toughness_MPa_strain": toughness,
    "offset_eng": offset,
    "elastic_max_eng": float(elastic_max),
    "smoothed": bool(smooth),
    "savgol_window": int(win),
    "savgol_poly": int(poly),
}])

st.download_button(
    "⬇️ Metrics CSV",
    out_metrics.to_csv(index=False),
    file_name="metrics.csv",
    mime="text/csv"
)

curves = pd.DataFrame({
    "strain_work": x_work,
    "stress_work_raw": y_work,
    "stress_work_smoothed": y_s,
    "offset_line_work": yline_work,
    "strain_eng": x_eng,
    "stress_eng": y_eng
})
st.download_button(
    "⬇️ Curves CSV",
    curves.to_csv(index=False),
    file_name="curves.csv",
    mime="text/csv"
)

st.caption("Include metrics and curves CSV in your lab report for reproducibility.")
