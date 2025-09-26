# pages/4_Methods_&_Assumptions.py
import streamlit as st
st.title("Methods & Assumptions")

st.markdown("""
### Calculations
- **Engineering stress**: σ = F / A₀ (A₀ = width × thickness).
- **Engineering strain**: ε = ΔL / L₀.
- **Young’s modulus (E)**: robust linear fit (RANSAC; fallback Theil–Sen) on ε ∈ [0, ε_max].
- **Yield stress (σᵧ)**: intersection of stress–strain curve with **offset line**: σ = E(ε − ε₀).
- **UTS**: max(σ).
- **Toughness**: ∫ σ dε (trapezoidal).
- **Smoothing**: Savitzky–Golay (window, polyorder) with raw data always available.

### Assumptions
- Force is in newtons (N); geometry in mm → stress in MPa.
- Preload trimming removes slack where F < 1%F_max.
- Elastic region upper bound (ε_max) must be within the linear regime and **before any yielding**.
- Bootstrap resamples assume i.i.d. residuals (approximation).

### Notes
- Prefer consistent sampling rate; avoid missing time stamps.
- For high necking, engineering curves understate true stress—consider true stress/strain for post-uniform region.
""")
