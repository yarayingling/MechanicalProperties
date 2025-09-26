# pages/2_Single_Test_Analysis.py
# Stress–strain single-test analysis with SciPy bootstrap CIs
import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal, integrate
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
x = df["eng_strain"].values.astype(float)   # engineering strain
y = df["eng_stress"].values.astype(float)   # engineering stress (MPa)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Analysis settings")

smooth = st.sidebar.toggle("Savitzky–Golay smoothing", value=True)
win = st.sidebar.slider("Savgol window", 7, 301, 41, step=2)
poly = st.sidebar.slider("Savgol polyorder", 1, 5, 3)

elastic_max = st.sidebar.slider("Elastic fit upper strain (εₘₐₓ)", 0.002, 0.05, 0.02, step=0.001, help="Upper bound of linear region for E")
offset = st.sidebar.number_input("Yield offset (ε₀)", 0.000, 0.050, 0.002, step=0.001, format="%.3f",
                                 help="0.002 = 0.2% proof stress")

do_boot = st.sidebar.toggle("Compute bootstrap CIs", value=True,
                            help="95% confidence intervals via SciPy bootstrap (paired resampling)")
B = st.sidebar.slider("Bootstrap resamples", 200, 5000, 1000, step=100)

# -------------------------
# Preprocess (smoothing)
# -------------------------
if smooth:
    try:
        y_s = signal.savgol_filter(y, window_length=win, polyorder=poly)
    except ValueError:
        st.warning("Savgol window too large for dataset — using raw curve instead.")
        y_s = y.copy()
else:
    y_s = y.copy()

# -------------------------
# Elastic modulus (robust)
# -------------------------
mask = x <= float(elastic_max)
if mask.sum() < 5:
    st.error("Not enough points in the elastic window. Increase εₘₐₓ or check data.")
    st.stop()

X_el = x[mask].reshape(-1, 1)
Y_el = y_s[mask]

E = None
intercept = 0.0
# Try robust RANSAC, then Theil-Sen, then simple linear regression
try:
    ransac = RANSACRegressor(LinearRegression(), residual_threshold=max(1e-6, Y_el.std() * 0.5), random_state=0)
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
        # Fallback to simple LR
        slope, intercept = np.polyfit(X_el.ravel(), Y_el, 1)
        E = float(slope)

# -------------------------
# Yield (0.2% offset), UTS, Toughness
# -------------------------
offset_line = E * (x - offset) + intercept
iy = int(np.argmin(np.abs(y_s - offset_line)))
sigma_y = float(y_s[iy]); eps_y = float(x[iy])

iuts = int(np.argmax(y_s))
uts = float(y_s[iuts]); eps_uts = float(x[iuts])

toughness = float(integrate.trapz(y_s, x))  # MPa·strain

# -------------------------
# Bootstrap CIs (SciPy)
# -------------------------
ciE_lo = ciE_hi = np.nan
ciY_lo = ciY_hi = np.nan

if do_boot:
    # E CI: paired resampling over the elastic window
    def stat_E(x_el, y_el):
        x_el = np.asarray(x_el); y_el = np.asarray(y_el)
        # simple slope on resampled elastic window (fast & stable)
        if x_el.size < 5:
            return np.nan
        slope, _ = np.polyfit(x_el, y_el, 1)
        return slope

    resE = bootstrap(
        data=(X_el.ravel(), Y_el),
        statistic=lambda a, b: stat_E(a, b),
        paired=True,
        vectorized=False,
        confidence_level=0.95,
        n_resamples=B,
        method="BCa",
        random_state=0,
    )
    ciE_lo, ciE_hi = float(resE.confidence_interval.low), float(resE.confidence_interval.high)

    # Yield CI: paired resampling on the whole curve; recompute E within each sample
    def stat_sigma_y(x_all, y_all):
        x_all = np.asarray(x_all); y_all = np.asarray(y_all)
        m = x_all <= elastic_max
        if m.sum() < 5:
            return np.nan
        # quick E on this bootstrap sample
        slope, intercept_b = np.polyfit(x_all[m], y_all[m], 1)
        yline = slope * (x_all - offset) + intercept_b
        idx = np.argmin(np.abs(y_all - yline))
        return float(y_all[idx])

    resY = bootstrap(
        data=(x, y_s),
        statistic=lambda a, b: stat_sigma_y(a, b),
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
m1.metric("E (MPa)", f"{E:,.0f}", f"95% CI: {ciE_lo:,.0f}–{ciE_hi:,.0f}" if do_boot else None)
m2.metric("σᵧ (MPa)", f"{sigma_y:,.0f}", f"{offset*100:.1f}% offset")
m3.metric("UTS (MPa)", f"{uts:,.0f}")
m4.metric("Toughness (MPa·strain)", f"{toughness:,.0f}")

if do_boot:
    st.caption(f"σᵧ 95% CI: {ciY_lo:,.0f} – {ciY_hi:,.0f} MPa")

# -------------------------
# Plots
# -------------------------
fig = go.Figure()
fig.add_scatter(x=x, y=y, name="Raw", mode="lines")
if smooth:
    fig.add_scatter(x=x, y=y_s, name="Smoothed", mode="lines")
fig.add_scatter(x=x, y=offset_line, name=f"Offset line ({offset*100:.1f}%)", mode="lines")
fig.add_scatter(x=[eps_y], y=[sigma_y], name="Yield", mode="markers")
fig.add_scatter(x=[eps_uts], y=[uts], name="UTS", mode="markers")
fig.update_layout(
    xaxis_title="Engineering strain (–)",
    yaxis_title="Engineering stress (MPa)",
    legend=dict(orientation="h"),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

# Optional derivative/residuals quick-look
with st.expander("Diagnostics: dσ/dε and residuals"):
    # Derivative (finite diff on smoothed curve)
    dsdE = np.gradient(y_s, x, edge_order=2)
    fig_d = go.Figure()
    fig_d.add_scatter(x=x, y=dsdE, mode="lines", name="dσ/dε")
    fig_d.update_layout(xaxis_title="Strain", yaxis_title="dσ/dε (MPa)", height=300)
    st.plotly_chart(fig_d, use_container_width=True)

    # Residuals in elastic window vs. fitted line
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
    "E_MPa": E,
    "E_CI_low": ciE_lo, "E_CI_high": ciE_hi,
    "sigma_y_MPa": sigma_y,
    "sigma_y_CI_low": ciY_lo, "sigma_y_CI_high": ciY_hi,
    "UTS_MPa": uts,
    "eps_y": eps_y,
    "eps_UTS": eps_uts,
    "toughness_MPa_strain": toughness,
    "offset": offset,
    "elastic_max": float(elastic_max),
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
    "strain": x,
    "stress_raw": y,
    "stress_smoothed": y_s,
    "offset_line": offset_line
})
st.download_button(
    "⬇️ Curves CSV",
    curves.to_csv(index=False),
    file_name="curves.csv",
    mime="text/csv"
)

st.caption("Tip: For reproducibility, include this CSV and metrics file in your lab notebook or report.")
