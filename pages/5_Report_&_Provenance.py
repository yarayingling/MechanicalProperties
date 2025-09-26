# pages/5_Report_&_Provenance.py
import streamlit as st, pandas as pd, platform, sys, datetime as dt, json
st.title("Report & Provenance")

if "trimmed_df" not in st.session_state:
    st.info("Analyze a test first."); st.stop()

params = {
    "timestamp_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "settings": {k: st.session_state.get(k) for k in ["mapping","specimen"]}
}
st.json(params)

html = f"""
<h2>Mechanical Test Report</h2>
<p><b>Generated:</b> {params['timestamp_utc']} UTC</p>
<p><b>Environment:</b> Python {params['python']} on {params['platform']}</p>
<pre>{json.dumps(params['settings'], indent=2)}</pre>
"""
st.download_button("⬇️ Download HTML report", html, "report.html", "text/html")
