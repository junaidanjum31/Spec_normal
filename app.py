import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Spectrum Normalizer Pro", layout="wide")

st.title("📊 Spectrum Normalizer Pro")
st.markdown("**Click directly on the plot to select reference peak**")

# ====================== SESSION STATE ======================
if "ref_value" not in st.session_state:
    st.session_state["ref_value"] = None

if "clicked_x" not in st.session_state:
    st.session_state["clicked_x"] = None

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

# ====================== MAIN UI ======================
if data_dict:

    st.header("🎯 Reference Peak Selection")

    ref_plot = st.selectbox("Select Reference Spectrum", list(data_dict.keys()))
    ref_df = data_dict[ref_plot]

    picker_mode = st.radio(
        "Reference Selection Mode",
        ["Auto (Global Max)", "Click from Plot", "Manual / Click Hybrid"]
    )

    # ====================== PLOT ======================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ref_df["x"],
        y=ref_df["y"],
        mode="lines+markers",
        name=ref_plot,
        line=dict(width=3),
        marker=dict(size=4)
    ))

    # show selected peak
    if st.session_state["ref_value"] is not None:
        fig.add_trace(go.Scatter(
            x=[st.session_state["clicked_x"]],
            y=[st.session_state["ref_value"]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Selected Peak"
        ))

    fig.update_layout(height=500, title="Click on Peak")

    clicked_points = plotly_events(fig, click_event=True)

    # ====================== CLICK HANDLER ======================
    if clicked_points:
        point = clicked_points[0]
        clicked_x = point["x"]

        peaks, _ = find_peaks(ref_df["y"].values)

        if len(peaks) > 0:
            peak_xs = ref_df["x"].iloc[peaks]
            closest_peak_idx = np.argmin(np.abs(peak_xs - clicked_x))
            idx = peaks[closest_peak_idx]
        else:
            idx = np.argmin(np.abs(ref_df["x"] - clicked_x))

        st.session_state["ref_value"] = float(ref_df["y"].iloc[idx])
        st.session_state["clicked_x"] = float(ref_df["x"].iloc[idx])

    # ====================== MANUAL INPUT ======================
    if picker_mode == "Manual / Click Hybrid":

        manual_input = st.number_input(
            "Reference Value",
            value=float(st.session_state["ref_value"]) if st.session_state["ref_value"] else 0.0
        )

        if manual_input != 0.0:
            st.session_state["ref_value"] = manual_input

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
            if picker_mode == "Auto (Global Max)":
                ref_y = data_dict[ref_plot]["y"].values
                ref_base = ref_y.min() if baseline_mode == "Auto (Minimum)" else manual_baseline
                norm_factor = (ref_y.max() - ref_base) or 1.0
            else:
                ref_value = st.session_state["ref_value"]

                if ref_value is None:
                    st.warning("⚠️ Select or enter reference peak value")
                    st.stop()

                norm_factor = ref_value

        y_norm = y_base / norm_factor if norm_factor > 0 else y_base

        normalized_data[name] = pd.DataFrame({
            "x": x,
            "y_normalized": y_norm
        })

    # ====================== FINAL PLOT ======================
    st.header("📈 Normalized Spectra")

    fig = go.Figure()

    for i, (name, dfn) in enumerate(normalized_data.items()):
        fig.add_trace(go.Scatter(
            x=dfn["x"],
            y=dfn["y_normalized"],
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