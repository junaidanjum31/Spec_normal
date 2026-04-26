import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import os
st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

# ================== THEME ==================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.theme == "Dark")
if dark_mode:
    st.session_state.theme = "Dark"
    st._config.set_option("theme.base", "dark")
else:
    st.session_state.theme = "Light"
    st._config.set_option("theme.base", "light")

st.title("📊 Spectrum / Peak Data Normalizer Pro")
st.markdown("**Normalize • Smooth • Peak Detect • Multi-Spectra Support**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Data Input")
    upload_mode = st.radio("Upload Method", ["Multiple Files", "Load Folder (Local)"])
    
    if upload_mode == "Multiple Files":
        uploaded_files = st.file_uploader("CSV or Excel files", 
                                         type=["csv", "xlsx", "xls"], 
                                         accept_multiple_files=True)
    else:
        folder_path = st.text_input("Folder path (local only)", "")
        uploaded_files = []
        if folder_path and os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                    full = os.path.join(folder_path, f)
                    uploaded_files.append(type('obj', (object,), {
                        'name': f, 
                        'read': lambda p=full: open(p, 'rb').read()
                    })())

    st.divider()
    spectra_type = st.selectbox("Spectra Type", 
        ["XPS", "Raman", "FTIR", "UV-Vis", "XRD", "Others (Custom)"])
    
    # Auto axis labels
    x_label = {"XPS": "Binding Energy (eV)", 
               "Raman": "Raman Shift (cm⁻¹)", 
               "FTIR": "Wavenumber (cm⁻¹)", 
               "UV-Vis": "Wavelength (nm)", 
               "XRD": "2θ (degrees)", 
               "Others (Custom)": "X"}[spectra_type]
    
    y_label = {"XPS": "Intensity (a.u.)", 
               "Raman": "Intensity (a.u.)", 
               "FTIR": "Absorbance", 
               "UV-Vis": "Absorbance", 
               "XRD": "Intensity (counts)", 
               "Others (Custom)": "Y"}[spectra_type]

    st.divider()
    baseline_mode = st.radio("Baseline Correction", 
                           ["Auto (Minimum)", "Manual Range", "Fixed Value"])
    
    if baseline_mode == "Manual Range":
        baseline_x_min = st.number_input("Baseline Start X", value=0.0)
        baseline_x_max = st.number_input("Baseline End X", value=100.0)
    elif baseline_mode == "Fixed Value":
        manual_baseline = st.number_input("Fixed Baseline Y", value=0.0)

    mode = st.radio("Normalization Mode", 
                   ["Stack & Normalize Together", "Individual Normalization"])

    smooth = st.checkbox("✨ Savitzky-Golay Smoothing", value=False)
    if smooth:
        window = st.slider("Window Length (odd number)", 3, 51, 11, step=2)
        poly = st.slider("Polynomial Order", 1, 5, 2)

    detect_peaks = st.checkbox("🔍 Peak Detection", value=True)
    if detect_peaks:
        prominence = st.slider("Prominence", 0.01, 0.5, 0.05)
        min_height = st.slider("Min Peak Height (norm)", 0.05, 1.0, 0.15)

    overlay_raw = st.checkbox("📍 Overlay Raw Data (scaled)", value=False)

# ====================== LOAD & PROCESS ======================
data_dict = {}

if uploaded_files:
    for file in uploaded_files:
        try:
            if hasattr(file, 'name') and file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                continue
            
            x_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            
            # Column selector (only once)
            if not data_dict:
                c1, c2 = st.columns(2)
                with c1:
                    x_col = st.selectbox("X Column", numeric_cols, index=0, key="xcol")
                with c2:
                    y_col = st.selectbox("Y Column", numeric_cols, index=1, key="ycol")
            
            df = df[[x_col, y_col]].dropna().copy()
            df.columns = ['x', 'y']
            data_dict[file.name if hasattr(file,'name') else f"Data_{len(data_dict)}"] = df
        except:
            pass

# ====================== NORMALIZATION & PROCESSING ======================
if data_dict:
    filenames = list(data_dict.keys())
    normalized_data = {}
    raw_scaled = {}
    peaks_dict = {}

    ref_file = st.selectbox("Reference file for scaling (max=1)", filenames) if mode == "Stack & Normalize Together" else None

    for name, df in data_dict.items():
        x = df['x'].values
        y = df['y'].values

        # Baseline
        if baseline_mode == "Auto (Minimum)":
            baseline = y.min()
        elif baseline_mode == "Fixed Value":
            baseline = manual_baseline
        else:  # Manual Range
            mask = (x >= baseline_x_min) & (x <= baseline_x_max)
            baseline = y[mask].mean() if np.any(mask) else y.min()

        y_base = y - baseline
        y_base = np.maximum(y_base, 0)

        # Smoothing
        if smooth and len(y_base) > window:
            y_base = savgol_filter(y_base, window, poly)

        max_val = y_base.max() or 1.0

        # Normalization
        if ref_file and mode == "Stack & Normalize Together":
            ref_max = max((d['y'].max() - (d['y'].min() if baseline_mode=="Auto (Minimum)" else baseline)) 
                         for d in data_dict.values())
            y_norm = y_base / ref_max
        else:
            y_norm = y_base / max_val

        normalized_data[name] = pd.DataFrame({'x': x, 'y_normalized': y_norm})
        raw_scaled[name] = pd.DataFrame({'x': x, 'y_raw_scaled': y / y.max()})  # for overlay

        # Peaks
        if detect_peaks:
            peaks, props = find_peaks(y_norm, prominence=prominence, height=min_height)
            peaks_dict[name] = (x[peaks], y_norm[peaks])

    # ====================== PLOT ======================
    st.header(f"Interactive Plot — {spectra_type} Spectra")

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, name in enumerate(normalized_data):
        dfn = normalized_data[name]
        fig.add_trace(go.Scatter(
            x=dfn['x'], y=dfn['y_normalized'],
            mode='lines', name=f"{name} (Norm)",
            line=dict(color=colors[i % len(colors)], width=2.5),
            hovertemplate=f'<b>{name} Norm</b><br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<extra></extra>'
        ))

        if overlay_raw:
            dfr = raw_scaled[name]
            fig.add_trace(go.Scatter(
                x=dfr['x'], y=dfr['y_raw_scaled'],
                mode='lines', name=f"{name} (Raw)",
                line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'),
                opacity=0.6,
                hovertemplate=f'<b>{name} Raw</b><br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<extra></extra>'
            ))

        if detect_peaks and name in peaks_dict:
            px, py = peaks_dict[name]
            fig.add_trace(go.Scatter(
                x=px, y=py, mode='markers+text',
                name=f"{name} peaks",
                marker=dict(color='red', size=9, symbol='x'),
                text=[f"{p:.3f}" for p in py],
                textposition="top center"
            ))

    fig.update_layout(
        title=f"Normalized {spectra_type} Spectra",
        xaxis_title=x_label,
        yaxis_title="Normalized Intensity (0–1)",
        template="plotly_dark" if dark_mode else "plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        height=680
    )
    fig.update_yaxes(range=[-0.05, 1.1])

    st.plotly_chart(fig, use_container_width=True)

    # Peak Table
    if detect_peaks:
        st.subheader("Detected Peaks")
        for name in peaks_dict:
            px, py = peaks_dict[name]
            st.write(f"**{name}**")
            st.dataframe(pd.DataFrame({"X": px, "Norm Y": py}).round(4))

    # ====================== DOWNLOADS ======================
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        for name, dfn in normalized_data.items():
            csv = dfn.to_csv(index=False)
            st.download_button(f"↓ {name} Normalized", csv, f"{name}_normalized.csv", "text/csv")
    with c2:
        st.download_button("📸 Download Plot (PNG)", fig.to_image("png"), "spectrum_plot.png", "image/png")
    with c3:
        st.success(f"Ready for {spectra_type} analysis!")

else:
    st.info("👆 Upload files or enter a folder path to begin")
