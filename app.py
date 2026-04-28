import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Upload Data")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    spectra_type = st.selectbox(
        "Spectra Type",
        ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others"]
    )

    normalization_mode = st.radio(
        "Normalization Mode",
        ["Stack & Normalize Together", "Individual Normalization"]
    )

    baseline_mode = st.radio(
        "Baseline Correction",
        ["Auto (Minimum)", "Fixed Value"]
    )

    manual_baseline = 0.0
    if baseline_mode == "Fixed Value":
        manual_baseline = st.number_input("Fixed Baseline", value=0.0)

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
            if len(numeric_cols) < 2:
                continue

            x_col = numeric_cols[0]

            for y_col in numeric_cols[1:]:
                name = f"{file.name} - {y_col}"
                temp = df[[x_col, y_col]].dropna().copy()
                temp.columns = ["x", "y"]
                data_dict[name] = temp

        except:
            pass

# ====================== MAIN ======================
if data_dict:

    st.header("🎯 Reference Peak Selection")

    ref_plot = st.selectbox("Select Reference Spectrum", list(data_dict.keys()))
    ref_df = data_dict[ref_plot]

    picker_mode = st.radio(
        "Reference Selection Mode",
        ["Auto (Global Max)", "Peak Slider", "Manual Value"]
    )

    x = ref_df["x"].values
    y = ref_df["y"].values

    # ====================== BASELINE ======================
    baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
    y_base = np.maximum(y - baseline, 0)

    if smooth and len(y_base) > window:
        y_base = savgol_filter(y_base, window, poly)

    # ====================== PEAK DETECTION ======================
    peaks, _ = find_peaks(y_base)

    # ====================== SELECT REFERENCE ======================
    if picker_mode == "Auto (Global Max)":
        ref_value = y_base.max()

    elif picker_mode == "Peak Slider":
        if len(peaks) == 0:
            st.warning("No peaks detected")
            st.stop()

        peak_index = st.slider(
            "Select Peak Index",
            0,
            len(peaks) - 1,
            0
        )

        idx = peaks[peak_index]
        ref_value = y_base[idx]

    else:
        ref_value = st.number_input("Enter Reference Value", value=float(y_base.max()))

    # ====================== PLOT ======================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y_base,
        mode="lines",
        name="Spectrum"
    ))

    # Highlight selected peak
    if picker_mode == "Peak Slider" and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=[x[idx]],
            y=[y_base[idx]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Selected Peak"
        ))

    fig.update_layout(height=500, title="Reference Spectrum")

    st.plotly_chart(fig, use_container_width=True)

# ====================== NORMALIZATION ======================
if data_dict:

    normalized_data = {}

    for name, df in data_dict.items():

        x = df["x"].values
        y = df["y"].values

        baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
        y_base = np.maximum(y - baseline, 0)

        if smooth and len(y_base) > window:
            y_base = savgol_filter(y_base, window, poly)

        if normalization_mode == "Individual Normalization":
            norm_factor = y_base.max() or 1.0
        else:
            norm_factor = ref_value if ref_value > 0 else 1.0

        y_norm = y_base / norm_factor

        normalized_data[name] = pd.DataFrame({
            "x": x,
            "y": y_norm
        })

    # ====================== FINAL PLOT ======================
    st.header("📈 Normalized Spectra")

    fig = go.Figure()

    for name, dfn in normalized_data.items():
        fig.add_trace(go.Scatter(
            x=dfn["x"],
            y=dfn["y"],
            mode="lines",
            name=name
        ))

    fig.update_layout(
        height=650,
        hovermode="x unified",
        xaxis_title="X Axis",
        yaxis_title="Normalized Intensity"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📂 Upload files to start")