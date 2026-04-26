import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import os

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Multi-file stacking • Reference normalization • Peak detection**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_files = st.file_uploader("Upload CSV or Excel files", 
                                    type=["csv", "xlsx", "xls"], 
                                    accept_multiple_files=True)

    spectra_type = st.selectbox("Spectra Type", 
        ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others"])

    st.divider()
    normalization_mode = st.radio("Normalization Mode", 
                                ["Stack & Normalize Together", "Individual Normalization"])

    baseline_mode = st.radio("Baseline Correction", 
                           ["Auto (Minimum)", "Fixed Value"])

    if baseline_mode == "Fixed Value":
        manual_baseline = st.number_input("Fixed Baseline Value", value=0.0)

    smooth = st.checkbox("Savitzky-Golay Smoothing", value=False)
    if smooth:
        window = st.slider("Window Length", 5, 51, 11, step=2)
        poly = st.slider("Polynomial Order", 1, 5, 2)

    detect_peaks = st.checkbox("Enable Peak Detection", value=True)

# ====================== LOAD DATA ======================
data_dict = {}

if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                continue

            x_col = numeric_cols[0]
            y_col = numeric_cols[1]

            df = df[[x_col, y_col]].dropna().copy()
            df.columns = ['x', 'y']
            data_dict[file.name] = df
        except:
            pass

# ====================== PROCESSING ======================
if data_dict:
    filenames = list(data_dict.keys())
    
    ref_file = None
    if normalization_mode == "Stack & Normalize Together" and len(filenames) > 1:
        ref_file = st.selectbox("🔑 Select Reference File (its max will be used for all)", 
                              filenames, index=0)

    normalized_data = {}

    for name, df in data_dict.items():
        x = df['x'].values
        y = df['y'].values

        # Baseline
        baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
        y_base = np.maximum(y - baseline, 0)

        # Smoothing
        if smooth and len(y_base) > window:
            y_base = savgol_filter(y_base, window, poly)

        max_val = y_base.max() if y_base.max() > 0 else 1.0

        # Normalization using reference
        if ref_file and normalization_mode == "Stack & Normalize Together":
            ref_y = data_dict[ref_file]['y'].values
            ref_baseline = ref_y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
            ref_max = max(ref_y - ref_baseline)
            y_norm = y_base / ref_max if ref_max > 0 else y_base
        else:
            y_norm = y_base / max_val

        normalized_data[name] = pd.DataFrame({'x': x, 'y_normalized': y_norm})

    # ====================== PLOTTING ======================
    st.header("Stacked Normalized Spectra")

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, dfn) in enumerate(normalized_data.items()):
        fig.add_trace(go.Scatter(x=dfn['x'], y=dfn['y_normalized'],
                                mode='lines', name=name,
                                line=dict(width=2.5)))

    fig.update_layout(
        title="Normalized & Stacked Spectra",
        xaxis_title="X Axis",
        yaxis_title="Normalized Intensity",
        hovermode="x unified",
        height=650,
        legend=dict(orientation="h", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload multiple files to see stacking and reference selection")