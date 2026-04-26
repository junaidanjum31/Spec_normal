import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Multi-column Support • Peak Reference Normalization**")

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

# ====================== LOAD MULTI-COLUMN DATA ======================
data_dict = {}

if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                continue
                
            x_col = numeric_cols[0]
            for y_col in numeric_cols[1:]:
                plot_name = f"{file.name} - {y_col}"
                temp_df = df[[x_col, y_col]].dropna().copy()
                temp_df.columns = ['x', 'y']
                data_dict[plot_name] = temp_df
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

# ====================== REFERENCE SELECTION ======================
ref_plot = None
ref_value = None

if normalization_mode == "Stack & Normalize Together" and len(data_dict) > 1:
    ref_plot = st.selectbox("Select Reference Spectrum", list(data_dict.keys()))

    ref_method = st.radio("How to get Reference Value?", 
                        ["Global Maximum", "Specific Detected Peak", "Manual Value"])

    if ref_method == "Specific Detected Peak" and ref_plot:
        ref_df = data_dict[ref_plot]
        y = ref_df['y'].values
        baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
        y_base = np.maximum(y - baseline, 0)

        peaks, _ = find_peaks(y_base, prominence=0.03, distance=10)
        
        if len(peaks) > 0:
            peak_options = []
            for p in peaks:
                peak_options.append(f"X = {ref_df['x'].iloc[p]:.4f} | Y = {y_base[p]:.4f}")
            
            selected_peak = st.selectbox("Choose Reference Peak", peak_options)
            # Extract Y value
            ref_value = float(selected_peak.split("Y = ")[1])
            st.success(f"✅ Reference Peak selected: {selected_peak}")
        else:
            st.warning("No strong peaks detected. Using Global Max instead.")
            ref_value = y_base.max()

    elif ref_method == "Manual Value":
        ref_value = st.number_input("Enter Reference Value", value=1.0, step=0.01)

# ====================== PROCESS & PLOT ======================
if data_dict:
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

        # Normalization
        if normalization_mode == "Individual Normalization":
            norm_factor = y_base.max() or 1.0
        else:
            if ref_method == "Global Maximum":
                ref_y = data_dict[ref_plot]['y'].values
                ref_base = ref_y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
                norm_factor = (ref_y.max() - ref_base) or 1.0
            else:
                norm_factor = ref_value or (y_base.max() or 1.0)

        y_norm = y_base / norm_factor if norm_factor > 0 else y_base
        normalized_data[name] = pd.DataFrame({'x': x, 'y_normalized': y_norm})

    # Plot
    st.header("📈 Normalized Stacked Spectra")
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
    st.info("👆 Please upload your file(s)")