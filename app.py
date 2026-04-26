import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from streamlit_plotly_events import plotly_events
import io

# ====================== CONFIG & STATE ======================
st.set_page_config(page_title="Spectrum Pro", layout="wide")

if "ref_value" not in st.session_state:
    st.session_state.update({"ref_value": None, "clicked_x": None})

def reset_selection():
    st.session_state.update({"ref_value": None, "clicked_x": None})

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Configuration")
    uploaded_files = st.file_uploader("Upload Spectra", type=["csv", "xlsx"], accept_multiple_files=True)
    
    st.divider()
    norm_mode = st.radio("Normalization Strategy", ["Individual (0 to 1)", "Reference Peak Scaling"])
    base_mode = st.radio("Baseline", ["Auto (Subtract Min)", "Fixed Value"])
    manual_baseline = st.number_input("Baseline Value", value=0.0) if base_mode == "Fixed Value" else 0.0
    
    st.divider()
    smooth = st.toggle("Apply Savitzky-Golay Smoothing")
    if smooth:
        win = st.slider("Window", 5, 51, 11, step=2)
        poly = st.slider("Order", 1, 5, 2)

    if st.button("Reset Selection", on_click=reset_selection):
        st.rerun()

# ====================== DATA PROCESSING ======================
data_dict = {}
if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            cols = df.select_dtypes(include=[np.number]).columns
            if len(cols) >= 2:
                for y_col in cols[1:]:
                    name = f"{file.name} ({y_col})"
                    data_dict[name] = df[[cols[0], y_col]].dropna().rename(columns={cols[0]: "x", y_col: "y"})
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

# ====================== MAIN INTERFACE ======================
if not data_dict:
    st.info("👋 Upload CSV or Excel files to begin.")
    st.stop()

# 1. Reference Selection
st.subheader("🎯 Reference Selection")
ref_key = st.selectbox("Select reference spectrum", list(data_dict.keys()))
ref_df = data_dict[ref_key]

fig_ref = go.Figure()
fig_ref.add_trace(go.Scatter(x=ref_df["x"], y=ref_df["y"], name=ref_key, line=dict(color="#1f77b4")))

if st.session_state["clicked_x"]:
    fig_ref.add_vline(x=st.session_state["clicked_x"], line_dash="dash", line_color="red")
    fig_ref.add_trace(go.Scatter(x=[st.session_state["clicked_x"]], y=[st.session_state["ref_value"]], 
                                 mode="markers", marker=dict(color="red", size=10), name="Selected Peak"))

clicked = plotly_events(fig_ref, click_event=True)

if clicked:
    cx = clicked[0]["x"]
    # Snap to nearest actual data point
    idx = (ref_df["x"] - cx).abs().idxmin()
    st.session_state.update({"clicked_x": ref_df.loc[idx, "x"], "ref_value": ref_df.loc[idx, "y"]})
    st.rerun()

# 2. Calculation
normalized_dfs = {}
for name, df in data_dict.items():
    y_proc = df["y"].values
    
    # Baseline
    b_val = y_proc.min() if base_mode == "Auto (Subtract Min)" else manual_baseline
    y_proc = np.maximum(y_proc - b_val, 0)
    
    # Smoothing
    if smooth and len(y_proc) > win:
        y_proc = savgol_filter(y_proc, win, poly)
        
    # Scaling
    if norm_mode == "Individual (0 to 1)":
        scale = y_proc.max() if y_proc.max() != 0 else 1
    else:
        scale = st.session_state["ref_value"] if st.session_state["ref_value"] else 1
    
    normalized_dfs[name] = pd.DataFrame({"x": df["x"], "y": y_proc / scale})

# 3. Final Visualization
st.subheader("📈 Normalized Results")
fig_res = go.Figure()
for name, ndf in normalized_dfs.items():
    fig_res.add_trace(go.Scatter(x=ndf["x"], y=ndf["y"], name=name))

fig_res.update_layout(height=600, hovermode="x", xaxis_title="Wavelength / Energy", yaxis_title="Normalized Intensity")
st.plotly_chart(fig_res, use_container_width=True)

# 4. Export
if normalized_dfs:
    # Merge all for download
    export_df = pd.DataFrame({"x": next(iter(normalized_dfs.values()))["x"]})
    for name, ndf in normalized_dfs.items():
        export_df[name] = ndf["y"]
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Normalized Data (CSV)", data=csv, file_name="normalized_spectra.csv", mime="text/csv")