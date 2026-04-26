import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Click directly on the plot to select reference peak**")

# Session State
if "ref_value" not in st.session_state:
    st.session_state.ref_value = None
if "clicked_x" not in st.session_state:
    st.session_state.clicked_x = None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_files = st.file_uploader("Upload CSV or Excel files", 
                                    type=["csv", "xlsx", "xls"], 
                                    accept_multiple_files=True)

    spectra_type = st.selectbox("Spectra Type", ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others"])

    normalization_mode = st.radio("Normalization Mode", 
                                ["Stack & Normalize Together", "Individual Normalization"])

    baseline_mode = st.radio("Baseline Correction", ["Auto (Minimum)", "Fixed Value"])
    manual_baseline = st.number_input("Fixed Baseline", value=0.0) if baseline_mode == "Fixed Value" else 0.0

    smooth = st.checkbox("Savitzky-Golay Smoothing", False)
    if smooth:
        window = st.slider("Window Length", 5, 51, 11, step=2)
        poly = st.slider("Polynomial Order", 1, 5, 2)

# ====================== LOAD DATA ======================
data_dict = {}
if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2: continue
            x_col = numeric_cols[0]
            for y_col in numeric_cols[1:]:
                name = f"{file.name} - {y_col}"
                temp = df[[x_col, y_col]].dropna().copy()
                temp.columns = ["x", "y"]
                data_dict[name] = temp
        except:
            pass

if not data_dict:
    st.info("👆 Upload your files to begin")
    st.stop()

# ====================== REFERENCE SELECTION ======================
st.subheader("🎯 Click on Plot to Select Reference Peak")

ref_plot = st.selectbox("Select Reference Spectrum", list(data_dict.keys()))
ref_df = data_dict[ref_plot]

# Plot for clicking
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ref_df["x"], 
    y=ref_df["y"], 
    mode="lines+markers",
    name=ref_plot,
    line=dict(width=3),
    marker=dict(size=6)
))
fig.update_layout(title="Click on any peak in this plot to set it as reference", height=550)

clicked = plotly_events(fig, click_event=True, override_width="100%")

if clicked:
    cx = clicked[0]["x"]
    idx = np.argmin(np.abs(ref_df["x"] - cx))
    st.session_state.clicked_x = float(ref_df["x"].iloc[idx])
    st.session_state.ref_value = float(ref_df["y"].iloc[idx])
    st.success(f"✅ Peak Selected → X = {st.session_state.clicked_x:.4f} | Y = {st.session_state.ref_value:.4f}")

# ====================== NORMALIZATION ======================
normalized_data = {}
for name, df in data_dict.items():
    x = df["x"].values
    y = df["y"].values

    baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
    y_base = np.maximum(y - baseline, 0)

    if smooth and 'window' in locals() and len(y_base) > window:
        y_base = savgol_filter(y_base, window, poly)

    if normalization_mode == "Individual Normalization":
        norm_factor = y_base.max() or 1.0
    else:
        if st.session_state.ref_value is None:
            st.warning("Please click on a peak in the plot above first")
            st.stop()
        norm_factor = st.session_state.ref_value

    y_norm = y_base / norm_factor if norm_factor > 0 else y_base
    normalized_data[name] = pd.DataFrame({"x": x, "y_normalized": y_norm})

# ====================== FINAL PLOT ======================
st.header("📈 Normalized Stacked Spectra")
fig_final = go.Figure()
for i, (name, dfn) in enumerate(normalized_data.items()):
    fig_final.add_trace(go.Scatter(x=dfn["x"], y=dfn["y_normalized"], name=name))

fig_final.update_layout(
    height=650,
    hovermode="x unified",
    xaxis_title="X Axis",
    yaxis_title="Normalized Intensity (0 - 1)"
)
st.plotly_chart(fig_final, use_container_width=True)