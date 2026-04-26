import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Peak Reference Normalization**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_files = st.file_uploader("Upload CSV/Excel files", 
                                    type=["csv", "xlsx", "xls"], 
                                    accept_multiple_files=True)

    spectra_type = st.selectbox("Spectra Type", ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others"])

    normalization_mode = st.radio("Normalization Mode", 
                                ["Stack & Normalize Together", "Individual Normalization"])

    baseline_mode = st.radio("Baseline", ["Auto (Minimum)", "Fixed Value"])
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
    st.info("Upload your file to begin")
    st.stop()

# ====================== REFERENCE SELECTION ======================
st.subheader("🎯 Reference Selection")

ref_plot = st.selectbox("Select Reference Spectrum", list(data_dict.keys()))
ref_df = data_dict[ref_plot]

# Show Reference Plot
st.plotly_chart(go.Figure(data=go.Scatter(x=ref_df["x"], y=ref_df["y"], mode="lines")).update_layout(
    title="Reference Spectrum - Visual Only", height=400), use_container_width=True)

# Peak Selection (Reliable way)
ref_method = st.radio("Reference Value", ["Global Max", "Select from Detected Peaks", "Manual Value"])

ref_value = None
if ref_method == "Select from Detected Peaks":
    y = ref_df["y"].values
    baseline = y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
    y_base = np.maximum(y - baseline, 0)
    peaks, _ = find_peaks(y_base, prominence=0.05, distance=10)
    
    if len(peaks) > 0:
        options = [f"X={ref_df['x'].iloc[p]:.4f} | Y={y_base[p]:.4f}" for p in peaks]
        selected = st.selectbox("Choose Peak", options)
        ref_value = float(selected.split("Y=")[1])
    else:
        st.warning("No peaks detected")

elif ref_method == "Manual Value":
    ref_value = st.number_input("Enter Reference Value", value=1.0)

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
        norm_factor = ref_value if ref_value is not None else (y_base.max() or 1.0)

    y_norm = y_base / norm_factor if norm_factor > 0 else y_base
    normalized_data[name] = pd.DataFrame({"x": x, "y_normalized": y_norm})

# Final Plot
st.header("📈 Normalized Stacked Spectra")
fig = go.Figure()
for name, dfn in normalized_data.items():
    fig.add_trace(go.Scatter(x=dfn["x"], y=dfn["y_normalized"], name=name))

fig.update_layout(height=650, hovermode="x unified", yaxis_title="Normalized Intensity")
st.plotly_chart(fig, use_container_width=True)