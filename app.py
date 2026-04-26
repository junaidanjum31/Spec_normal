import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Multi-column files • Stacking • Reference Normalization**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files (multi-column supported)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    spectra_type = st.selectbox("Spectra Type", 
        ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others"])

    normalization_mode = st.radio("Normalization Mode", 
                                ["Stack & Normalize Together", "Individual Normalization"])

    baseline_mode = st.radio("Baseline Correction", ["Auto (Minimum)", "Fixed Value"])
    if baseline_mode == "Fixed Value":
        manual_baseline = st.number_input("Fixed Baseline Value", value=0.0)

    smooth = st.checkbox("Savitzky-Golay Smoothing", False)
    if smooth:
        window = st.slider("Window Length", 5, 51, 11, step=2)
        poly = st.slider("Polynomial Order", 1, 5, 2)

    detect_peaks = st.checkbox("Enable Peak Detection", True)

# ====================== LOAD MULTI-COLUMN DATA ======================
data_dict = {}  # key = "filename - ColumnName"

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
            y_cols = numeric_cols[1:]

            for y_col in y_cols:
                plot_name = f"{file.name} - {y_col}"
                temp_df = df[[x_col, y_col]].dropna().copy()
                temp_df.columns = ['x', 'y']
                data_dict[plot_name] = temp_df
        except Exception as e:
            st.error(f"Error with {file.name}: {e}")

# ====================== REFERENCE SELECTION (NOW MORE RELIABLE) ======================
ref_plot = None
if normalization_mode == "Stack & Normalize Together" and len(data_dict) > 1:
    ref_plot = st.selectbox(
        "🔑 Select Reference Plot/Column (its max will normalize all others)",
        list(data_dict.keys()),
        index=0
    )

# ====================== PROCESSING ======================
if data_dict:
    normalized_data = {}

    for name, df in data_dict.items():
        x = df['x'].values
        y = df['y'].values

        baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
        y_base = np.maximum(y - baseline, 0)

        if smooth and len(y_base) > window:
            y_base = savgol_filter(y_base, window, poly)

        max_val = y_base.max() or 1.0

        # Normalization
        if ref_plot:
            ref_y = data_dict[ref_plot]['y'].values
            ref_baseline = ref_y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
            ref_max = np.max(ref_y - ref_baseline) or 1.0
            y_norm = y_base / ref_max
        else:
            y_norm = y_base / max_val

        normalized_data[name] = pd.DataFrame({'x': x, 'y_normalized': y_norm})

    # ====================== PLOT ======================
    st.header("📈 Stacked Normalized Spectra")

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (name, dfn) in enumerate(normalized_data.items()):
        fig.add_trace(go.Scatter(
            x=dfn['x'], 
            y=dfn['y_normalized'],
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))

    fig.update_layout(
        title=f"Normalized {spectra_type} Spectra",
        xaxis_title="X Axis",
        yaxis_title="Normalized Intensity (0 - 1)",
        hovermode="x unified",
        height=700,
        legend=dict(orientation="h", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload your file to see all columns as separate plots")