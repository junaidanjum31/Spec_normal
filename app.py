import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Click on Plot to Select Reference Peak**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_files = st.file_uploader("Upload files", 
                                    type=["csv", "xlsx", "xls"], 
                                    accept_multiple_files=True)

    spectra_type = st.selectbox("Spectra Type", ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others"])

    normalization_mode = st.radio("Normalization Mode", 
                                ["Stack & Normalize Together", "Individual Normalization"])

    baseline_mode = st.radio("Baseline Correction", ["Auto (Minimum)", "Fixed Value"])
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
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2: continue
            x_col = numeric_cols[0]
            for y_col in numeric_cols[1:]:
                name = f"{file.name} - {y_col}"
                temp = df[[x_col, y_col]].dropna().copy()
                temp.columns = ['x', 'y']
                data_dict[name] = temp
        except:
            pass

# ====================== REFERENCE PICKER ======================
ref_plot = None
ref_value = None

if normalization_mode == "Stack & Normalize Together" and data_dict:
    ref_plot = st.selectbox("Select Reference Spectrum", list(data_dict.keys()))
    
    picker_mode = st.radio("Reference Value Select کرنے کا طریقہ", 
                         ["Global Maximum", "Click on Plot to Pick Peak"])

    if picker_mode == "Click on Plot to Pick Peak" and ref_plot:
        st.info("**نیچے والے گراف پر جس پیک پر کلک کریں گے، وہ ریفرنس بن جائے گا**")
        
        ref_df = data_dict[ref_plot]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ref_df['x'], y=ref_df['y'], mode='lines+markers',
                               name=ref_plot, line=dict(width=3), marker=dict(size=4)))
        fig.update_layout(title="Click on any Peak", height=500)
        
        clicked = st.plotly_chart(fig, use_container_width=True, key="picker")

        # Click handling
        if clicked and "points" in clicked and len(clicked["points"]) > 0:
            point = clicked["points"][0]
            clicked_x = point["x"]
            idx = np.argmin(np.abs(ref_df['x'] - clicked_x))
            ref_value = float(ref_df['y'].iloc[idx])
            st.success(f"✅ Peak Selected Successfully!\nX = {clicked_x:.4f} | Y = {ref_value:.4f}")
            st.rerun()

# ====================== NORMALIZATION ======================
if data_dict:
    normalized_data = {}

    for name, df in data_dict.items():
        x = df['x'].values
        y = df['y'].values

        baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
        y_base = np.maximum(y - baseline, 0)

        if smooth and len(y_base) > window:
            y_base = savgol_filter(y_base, window, poly)

        if normalization_mode == "Individual Normalization":
            norm_factor = y_base.max() or 1.0
        else:
            if picker_mode == "Global Maximum":
                ref_y = data_dict[ref_plot]['y'].values
                ref_base = ref_y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
                norm_factor = (ref_y.max() - ref_base) or 1.0
            else:
                norm_factor = ref_value or (y_base.max() or 1.0)

        y_norm = y_base / norm_factor if norm_factor > 0 else y_base
        normalized_data[name] = pd.DataFrame({'x': x, 'y_normalized': y_norm})

    # Final Plot
    st.header("📈 Normalized Stacked Spectra")
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, dfn) in enumerate(normalized_data.items()):
        fig.add_trace(go.Scatter(x=dfn['x'], y=dfn['y_normalized'],
                               mode='lines', name=name,
                               line=dict(color=colors[i % len(colors)], width=2.5)))

    fig.update_layout(
        title=f"Normalized {spectra_type} Spectra",
        xaxis_title="X Axis",
        yaxis_title="Normalized Intensity (0 - 1)",
        hovermode="x unified",
        height=650,
        legend=dict(orientation="h", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("فائل اپ لوڈ کریں")