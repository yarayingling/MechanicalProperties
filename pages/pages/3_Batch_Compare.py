# pages/3_Batch_Compare.py
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go

st.title("Batch Comparison")

st.write("Upload multiple cleaned CSV files to overlay and summarize.")
ups = st.file_uploader("Cleaned CSVs (from page 1) — multiple", type=["csv"], accept_multiple_files=True)

if ups:
    dfs = []
    for f in ups:
        d = pd.read_csv(f)
        if {"eng_strain","eng_stress"}.issubset(d.columns):
            d["name"] = f.name
            dfs.append(d[["eng_strain","eng_stress","name"]])
    if not dfs: st.error("No valid files."); st.stop()
    big = pd.concat(dfs, ignore_index=True)

    fig = px.line(big, x="eng_strain", y="eng_stress", color="name")
    fig.update_layout(xaxis_title="Strain", yaxis_title="Stress (MPa)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Simple summary at selected strain grid
    grid = np.linspace(0, big["eng_strain"].max(), 200)
    stats = []
    for g in grid:
        ys = []
        for n, df_ in big.groupby("name"):
            # nearest neighbor
            idx = (df_["eng_strain"]-g).abs().idxmin()
            ys.append(df_.loc[idx, "eng_stress"])
        stats.append({"strain":g, "mean":np.mean(ys), "sd":np.std(ys)})
    sm = pd.DataFrame(stats)
    st.area_chart(sm.set_index("strain")[["mean"]])
