# pages/2_Single_Test_Analysis.py
import streamlit as st, pandas as pd, numpy as np
from scipy import signal, integrate
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, LinearRegression
import plotly.graph_objects as go
from statsmodels.stats.bootstrap import bootstrap as sm_bootstrap

st.title("Single Test Analysis")

if "trimmed_df" not in st.session_state:
    st.info("Go to 'Upload & QA/QC' first."); st.stop()

df = st.session_state["trimmed_df"].copy()
st.sidebar.header("Analysis settings")
smooth = st.sidebar.toggle("Savitzky–Golay Smoothing", value=True)
win = st.sidebar.slider("Savgol window", 7, 301, 41, step=2)
poly = st.sidebar.slider("Savgol polyorder", 1, 5, 3)
elastic_max = st.sidebar.slider("Elastic fit max strain", 0.002, 0.05, 0.02, step=0.001)
offset = st.sidebar.number_input("Offset for yield (ε₀)", 0.000, 0.050, 0.002, step=0.001, format="%.3f")
do_boot = st.sidebar.toggle("Bootstrap CIs", value=True)
B = st.sidebar.slider("Bootstrap samples", 200, 3000, 800, step=100)

x = df["eng_strain"].values
y = df["eng_stress"].values

if smooth:
    y_s = signal.savgol_filter(y, window_length=win, polyorder=poly)
else:
    y_s = y.copy()

# Elastic region mask
mask = x <= elastic_max
X = x[mask].reshape(-1,1); Y = y_s[mask]

# Robust modulus
try:
    ransac = RANSACRegressor(LinearRegression(), residual_threshold=Y.std()*0.5, random_state=0)
    ransac.fit(X, Y)
    E = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)
except Exception:
    ts = TheilSenRegressor().fit(X, Y)
    E = float(ts.coef_[0]); intercept = float(ts.intercept_)

# 0.2% offset yield (closest intersection)
offset_line = E*(x - offset) + intercept
iy = np.argmin(np.abs(y_s - offset_line))
sigma_y, eps_y = y_s[iy], x[iy]

# UTS
iuts = int(np.argmax(y_s)); uts, eps_uts = y_s[iuts], x[iuts]

# Toughness (area)
toughness = integrate.trapz(y_s, x)

# Ramberg–Osgood (optional fit)
def ro_model(eps, E, K, n, sigma0):
    # epsilon = sigma/E + (sigma/K)**n ; invert numerically later if needed
    # Here we fit sigma(eps) using simple search (for demo)
    # We'll do a coarse numeric inversion per point
    out = []
    for e in eps:
        # solve for sigma via Newton
        s = E*e
        for _ in range(25):
            f = e - (s/E) - (s/max(K,1e-6))**n
            df = -1/E - n*(s/max(K,1e-6))**(n-1)/max(K,1e-6)
            s -= f/df
        out.append(s)
    return np.array(out)

st.subheader("Key metrics")
m1,m2,m3,m4 = st.columns(4)
m1.metric("E (MPa)", f"{E:,.0f}")
m2.metric("σᵧ (MPa)", f"{sigma_y:,.0f}", f"{offset*100:.1f}% offset")
m3.metric("UTS (MPa)", f"{uts:,.0f}")
m4.metric("Toughness (MPa·strain)", f"{toughness:,.0f}")

# Bootstrap CIs
def boot_stat(func):
    idx = np.arange(len(x))
    stats = []
    rng = np.random.default_rng(0)
    for _ in range(B):
        b = rng.choice(idx, size=len(idx), replace=True)
        xb, yb = x[b], y_s[b]
        # recompute quick E and yield
        mb = xb <= elastic_max
        if mb.sum() < 5: continue
        Eb = np.polyfit(xb[mb], yb[mb], 1)[0]
        yline = Eb*(xb - offset)
        ib = np.argmin(np.abs(yb - yline))
        stats.append(func(Eb, yb[ib], (xb[ib], yb[ib])))
    arr = np.array(stats)
    return np.nanpercentile(arr, [2.5, 97.5])

ciE_lo, ciE_hi = boot_stat(lambda E, sy, p: E) if do_boot else (np.nan, np.nan)
ciY_lo, ciY_hi = boot_stat(lambda E, sy, p: sy) if do_boot else (np.nan, np.nan)

if do_boot:
    st.caption(f"E 95% CI: {ciE_lo:,.0f} – {ciE_hi:,.0f} MPa | σᵧ 95% CI: {ciY_lo:,.0f} – {ciY_hi:,.0f} MPa")

# Plot
fig = go.Figure()
fig.add_scatter(x=x, y=y, name="Raw", mode="lines")
if smooth: fig.add_scatter(x=x, y=y_s, name="Smoothed", mode="lines")
fig.add_scatter(x=x, y=offset_line, name=f"Offset line ({offset*100:.1f}%)", mode="lines")
fig.add_scatter(x=[eps_y], y=[sigma_y], name="Yield", mode="markers")
fig.add_scatter(x=[eps_uts], y=[uts], name="UTS", mode="markers")
fig.update_layout(xaxis_title="Engineering strain (–)", yaxis_title="Engineering stress (MPa)",
                  legend=dict(orientation="h"), height=500)
st.plotly_chart(fig, use_container_width=True)

# Downloads
out_metrics = pd.DataFrame([{
    "E_MPa":E, "sigma_y_MPa":sigma_y, "UTS_MPa":uts,
    "eps_y":eps_y, "eps_UTS":eps_uts, "toughness":toughness,
    "offset":offset, "elastic_max":elastic_max, "smoothed":smooth
}])
st.download_button("⬇️ Metrics CSV", out_metrics.to_csv(index=False), "metrics.csv", "text/csv")
curves = pd.DataFrame({"strain":x, "stress_raw":y, "stress_smoothed":y_s, "offset_line":offset_line})
st.download_button("⬇️ Curves CSV", curves.to_csv(index=False), "curves.csv", "text/csv")
