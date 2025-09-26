# Home.py
import streamlit as st
st.set_page_config(page_title="MSE Mechanical App", page_icon="🧪", layout="wide")

st.title("🧪 Materials Mechanics: Stress–Strain")
st.write("Upload raw tensile data → robust metrics → uncertainty → batch comparison → reports.")

with st.expander("What this app does"):
    st.markdown("""
- Calculates E, σᵧ (0.2% offset), UTS, elongations, toughness
- Robust fitting (RANSAC/Theil–Sen), bootstrap CIs
- Batch overlays and statistical summaries
- Download cleaned data, metrics, parameters, and an HTML report
""")
