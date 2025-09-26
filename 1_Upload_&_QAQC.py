# pages/1_Upload_&_QAQC.py
import streamlit as st, pandas as pd, numpy as np
from pydantic import BaseModel, Field, ValidationError
from io import StringIO
import json

st.title("Upload & QA/QC")

class Specimen(BaseModel):
    width_mm: float = Field(gt=0)
    thickness_mm: float = Field(gt=0)
    gauge_len_mm: float = Field(gt=0)
    unit_stress: str = "MPa"    # MPa or ksi

class Columns(BaseModel):
    force: str
    extension: str
    time: str | None = None
    stress: str | None = None
    strain: str | None = None

st.markdown("### 1) Upload raw CSV")
up = st.file_uploader("CSV with headers", type=["csv"])

st.markdown("### 2) Map columns (if needed)")
cform = st.form("colmap")
force_col = cform.text_input("Force column name", "Force")
ext_col   = cform.text_input("Extension column name", "Extension")
time_col  = cform.text_input("Time column (optional)", "Time")
stress_col= cform.text_input("Stress column (optional)", "")
strain_col= cform.text_input("Strain column (optional)", "")
submitted = cform.form_submit_button("Save mapping")

st.markdown("### 3) Specimen geometry")
g1,g2,g3,g4 = st.columns(4)
with g1: width = st.number_input("Width (mm)", 0.01, None, 6.0)
with g2: thick = st.number_input("Thickness (mm)", 0.01, None, 2.0)
with g3: gl    = st.number_input("Gauge length (mm)", 1.0, None, 25.0)
with g4: unit  = st.selectbox("Stress unit", ["MPa","ksi"], index=0)

if up and submitted:
    df = pd.read_csv(up)
    cols = Columns(force=force_col, extension=ext_col,
                   time=time_col if time_col else None,
                   stress=stress_col if stress_col else None,
                   strain=strain_col if strain_col else None)
    spec = Specimen(width_mm=width, thickness_mm=thick, gauge_len_mm=gl, unit_stress=unit)

    # Basic QA/QC
    for c in [cols.force, cols.extension]:
        if c not in df.columns: st.error(f"Missing column: {c}"); st.stop()

    # Engineering stress/strain if not present
    A = (spec.width_mm * spec.thickness_mm) * 1e-6  # mm^2 -> m^2; MPa uses N/mm^2 but we'll stay consistent
    # Here keep MPa convention: stress_MPa = Force_N / A_mm2
    A_mm2 = spec.width_mm * spec.thickness_mm
    df["eng_stress"] = df[cols.force] / A_mm2  # N / mm^2 -> MPa if Force is N
    df["eng_strain"] = df[cols.extension] / spec.gauge_len_mm

    # Preload trim: remove initial slack where force < 1% of max
    fmax = df[cols.force].max()
    trimmed = df[df[cols.force] >= 0.01 * fmax].copy()

    st.session_state["raw_df"] = df
    st.session_state["trimmed_df"] = trimmed
    st.session_state["mapping"] = cols.model_dump()
    st.session_state["specimen"] = spec.model_dump()

    st.success(f"Loaded {len(trimmed)} rows after preload trimming.")
    st.dataframe(trimmed.head(10), use_container_width=True)

    st.download_button("⬇️ Download cleaned CSV",
        trimmed.to_csv(index=False).encode(),
        "cleaned_stress_strain.csv", "text/csv")

    st.download_button("⬇️ Download parameters JSON",
        json.dumps({"columns": st.session_state["mapping"], "specimen": st.session_state["specimen"]}, indent=2),
        "parameters.json", "application/json")
