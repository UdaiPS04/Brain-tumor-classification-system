import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "brain_tumor_model.keras"
)

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMG_SIZE = (224, 224)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)

model = load_ai_model()

# ==========================================================
# THEME
# ==========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

LIGHT = {
    "bg": "#edf4fb",
    "card": "#ffffff",
    "text": "#17324d",
    "subtext": "#64748b",
    "sidebar": "#0d4d88",
    "border": "#dbe8f5",
    "button": "#0d6efd"
}

DARK = {
    "bg": "#111827",
    "card": "#1f2937",
    "text": "#f8fafc",
    "subtext": "#cbd5e1",
    "sidebar": "#0b1120",
    "border": "#334155",
    "button": "#2563eb"
}

theme = LIGHT if st.session_state.theme == "Light" else DARK

# ==========================================================
# CSS
# ==========================================================

st.markdown(f"""
<style>

#MainMenu {{
visibility:hidden;
}}

footer {{
visibility:hidden;
}}

header {{
visibility:hidden;
}}

html,
body,
.stApp {{

background:{theme['bg']};

font-family:
Inter,
Segoe UI,
SF Pro Display,
Helvetica,
Arial,
sans-serif;

}}

.block-container {{

padding-top:25px;
padding-left:32px;
padding-right:32px;
padding-bottom:25px;

}}

[data-testid="stSidebar"] {{

background:{theme['sidebar']};

}}

[data-testid="stSidebar"] * {{

color:white;

}}

.card {{

background:{theme['card']};

border:1px solid {theme['border']};

border-radius:22px;

padding:25px;

box-shadow:0 8px 24px rgba(0,0,0,.08);

margin-bottom:18px;

}}

.metric-card {{

background:{theme['card']};

border:1px solid {theme['border']};

border-radius:18px;

padding:18px;

text-align:center;

transition:.25s;

}}

.metric-card:hover {{

transform:translateY(-4px);

}}

.title {{

font-size:42px;

font-weight:700;

color:{theme['text']};

}}

.subtitle {{

font-size:18px;

color:{theme['subtext']};

}}

.section-title {{

font-size:25px;

font-weight:600;

color:{theme['text']};

}}

.stButton>button {{

width:100%;

height:52px;

border:none;

border-radius:12px;

background:{theme['button']};

color:white;

font-weight:600;

font-size:17px;

}}

.stButton>button:hover {{

background:#0b5ed7;

}}

[data-testid="stMetricValue"] {{

color:{theme['text']};

}}

[data-testid="stMetricLabel"] {{

color:{theme['subtext']};

}}

[data-testid="stMarkdownContainer"] {{

color:{theme['text']};

}}

.stFileUploader {{

background:{theme['card']};

}}

.stProgress > div > div > div {{

background:{theme['button']};

}}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:


    st.markdown(
    "<h1 style='text-align:center;font-size:90px;'>🧠</h1>",
    unsafe_allow_html=True
)

    st.title("Brain Tumor\nClassification")

    st.caption("AI Powered MRI Diagnosis")

    st.divider()

    st.subheader("Dashboard")

    st.button("Home", use_container_width=True)

    st.button("Upload MRI Scan", use_container_width=True)

    st.button("AI Prediction", use_container_width=True)

    st.button("Analytics", use_container_width=True)

    st.button("Classification Report", use_container_width=True)

    st.divider()

    st.subheader("Model Information")

    st.metric("Architecture", "CNN")

    st.metric("Framework", "TensorFlow")

    st.metric("Input Size", "224×224")

    st.metric("Classes", "4")

    st.divider()

    st.subheader("Tumor Classes")

    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• No Tumor")
    st.write("• Pituitary")

    st.divider()

    st.info("Version 1.0")

"""
Deep Learning Based MRI Classification
"""
    
# ==========================================================
# HEADER
# ==========================================================

left, right = st.columns([5,1])

with left:

    st.markdown(f"""
    <div class="card">

    <div class="title">

    🧠 Brain Tumor Classification

    </div>

    <div class="subtitle">

    Deep Learning Based MRI Brain Scan Classification System

    </div>

    </div>
    """, unsafe_allow_html=True)

with right:

    st.write("")

    if st.toggle(
        "Dark Mode",
        value=(st.session_state.theme == "Dark")
    ):

        if st.session_state.theme != "Dark":

            st.session_state.theme = "Dark"

            st.rerun()

    else:

        if st.session_state.theme != "Light":

            st.session_state.theme = "Light"

            st.rerun()

# ==========================================================
# DASHBOARD METRICS
# ==========================================================

m1, m2, m3, m4, m5 = st.columns(5)

with m1:

    st.markdown("""
    <div class="metric-card">
    <h4>Model</h4>
    <h2>CNN</h2>
    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown("""
    <div class="metric-card">
    <h4>Framework</h4>
    <h2>TensorFlow</h2>
    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown("""
    <div class="metric-card">
    <h4>Classes</h4>
    <h2>4</h2>
    </div>
    """, unsafe_allow_html=True)

with m4:

    st.markdown("""
    <div class="metric-card">
    <h4>Input</h4>
    <h2>224×224</h2>
    </div>
    """, unsafe_allow_html=True)

with m5:

    st.markdown("""
    <div class="metric-card">
    <h4>Status</h4>
    <h2>Ready</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================================
# UPLOAD SECTION
# ==========================================================

st.markdown(
    f"""
    <h2 style="color:{theme['text']}; margin-top:10px;">
    Upload MRI Scan
    </h2>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1.35, 1])

# ==========================================================
# LEFT PANEL
# ==========================================================

with left:

    with st.container(border=True):

        st.subheader("MRI Image Upload")

        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.caption(
            "Supported formats: JPG, JPEG, PNG"
        )

        st.write("")

        st.info(
            "Use a clear MRI brain scan for the most accurate prediction."
        )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    with st.container(border=True):

        st.subheader("Scan Guidelines")

        st.markdown("""
**Recommended**

- High-quality MRI scan
- Complete brain visible
- JPG / PNG format
- Proper orientation
- Good contrast

---

**Supported Classes**

- Glioma
- Meningioma
- No Tumor
- Pituitary
""")

st.write("")

# ==========================================================
# MRI PREVIEW
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    preview_col, info_col = st.columns([1.15, 1])

    with preview_col:

        with st.container(border=True):

            st.subheader("MRI Scan Preview")

            st.image(
                image,
                use_container_width=True
            )

    with info_col:

        with st.container(border=True):

            st.subheader("Scan Information")

            st.metric(
                "Image Width",
                image.size[0]
            )

            st.metric(
                "Image Height",
                image.size[1]
            )

            st.metric(
                "Image Mode",
                image.mode
            )

            st.metric(
                "Model Input",
                "224 × 224"
            )

else:

    st.info("Upload an MRI image to display its preview.")

st.write("")

# ==========================================================
# AI ANALYSIS
# ==========================================================

if uploaded_file is not None:

    st.write("")

    analyze_col1, analyze_col2 = st.columns([1.1, 1])

    with analyze_col1:

        analyze = st.button(
            "Analyze MRI Scan",
            use_container_width=True
        )

    with analyze_col2:

        st.success("AI Model Ready")

    if analyze:

        with st.spinner("Analyzing MRI scan..."):

            img = image.resize(IMG_SIZE)

            img_array = np.array(img)

            img_array = img_array.astype("float32") / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)

            predicted_index = np.argmax(prediction)

            predicted_class = CLASSES[predicted_index]

            confidence = float(prediction[0][predicted_index]) * 100

            st.session_state["prediction"] = prediction[0]
            st.session_state["predicted_class"] = predicted_class
            st.session_state["confidence"] = confidence

# ==========================================================
# RESULT PANEL
# ==========================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    predicted_class = st.session_state["predicted_class"]

    confidence = st.session_state["confidence"]

    st.write("")

    left, right = st.columns([1.2, 1])

    # ------------------------------------------------------
    # RESULT CARD
    # ------------------------------------------------------

    with left:

        with st.container(border=True):

            st.subheader("Classification Result")

            if predicted_class == "No Tumor":

                st.success("No Tumor Detected")

            else:

                st.error(predicted_class)

            st.write("")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(confidence / 100)

    # ------------------------------------------------------
    # MODEL SUMMARY
    # ------------------------------------------------------

    with right:

        with st.container(border=True):

            st.subheader("Prediction Summary")

            st.metric(
                "Predicted Class",
                predicted_class
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.metric(
                "Total Classes",
                len(CLASSES)
            )

            st.metric(
                "Framework",
                "TensorFlow"
            )

st.write("")

# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================

if "prediction" in st.session_state:

    import pandas as pd
    import plotly.express as px

    prediction = st.session_state["prediction"]
    predicted_class = st.session_state["predicted_class"]
    confidence = st.session_state["confidence"]

    st.markdown(
        f"""
        <h2 style="color:{theme['text']};">
        Prediction Analytics
        </h2>
        """,
        unsafe_allow_html=True
    )

    chart_col, summary_col = st.columns([1.7, 1])

    # =====================================================
    # PROBABILITY CHART
    # =====================================================

    with chart_col:

        with st.container(border=True):

            st.subheader("Class Probability Distribution")

            df = pd.DataFrame({
                "Tumor Type": CLASSES,
                "Confidence": prediction * 100
            })

            fig = px.bar(
                df,
                x="Confidence",
                y="Tumor Type",
                orientation="h",
                text="Confidence"
            )

            fig.update_traces(
                texttemplate="%{x:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                height=360,
                template="plotly_white" if st.session_state.theme == "Light" else "plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Confidence (%)",
                yaxis_title="",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    with summary_col:

        with st.container(border=True):

            st.subheader("Prediction Summary")

            st.metric(
                "Prediction",
                predicted_class
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.metric(
                "Classes",
                len(CLASSES)
            )

            st.metric(
                "Framework",
                "TensorFlow"
            )

    st.write("")

    # =====================================================
    # CLASS CONFIDENCE TABLE
    # =====================================================

    with st.container(border=True):

        st.subheader("Confidence Scores")

        table = pd.DataFrame({
            "Tumor Type": CLASSES,
            "Confidence (%)": np.round(prediction * 100, 2)
        })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    st.write("")

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = f"""
===============================
Brain Tumor Classification Report
===============================

Prediction:
{predicted_class}

Confidence:
{confidence:.2f} %

--------------------------------

Glioma      : {prediction[0]*100:.2f} %

Meningioma  : {prediction[1]*100:.2f} %

No Tumor    : {prediction[2]*100:.2f} %

Pituitary   : {prediction[3]*100:.2f} %

--------------------------------

Generated using
Brain Tumor Classification
Deep Learning Dashboard
"""

    st.download_button(

        "Download Classification Report",

        report,

        file_name="Brain_Tumor_Report.txt",

        mime="text/plain",

        use_container_width=True

    )

st.write("")

# ==========================================================
# RESET & ACTIONS
# ==========================================================

st.write("")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    if st.button("Analyze Another Scan", use_container_width=True):

        for key in ["prediction", "predicted_class", "confidence"]:

            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

with col2:

    st.button(
        "Model Ready",
        use_container_width=True,
        disabled=True
    )

st.write("")

# ==========================================================
# AI DISCLAIMER
# ==========================================================

with st.container(border=True):

    st.subheader("AI Disclaimer")

    st.markdown("""

This application is developed for **educational and research purposes**.

The prediction is generated using a Convolutional Neural Network trained on MRI brain images.

It should **not** be considered as a medical diagnosis.

Always consult a qualified radiologist or healthcare professional before making any medical decision.

""")

st.write("")

# ==========================================================
# MODEL INFORMATION
# ==========================================================

with st.expander("About This Model"):

    st.write("### Brain Tumor Classification")

    st.write("""
The deep learning model classifies MRI brain images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The application uses:

- TensorFlow
- Keras
- Streamlit
- NumPy
- Plotly

Input Image Size:

224 × 224 RGB

Output:

Predicted tumor class with confidence score.
""")

st.write("")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

footer_bg = theme["card"]
footer_text = theme["text"]
footer_sub = theme["subtext"]
footer_border = theme["border"]

st.markdown(f"""
<div style="
background:{footer_bg};
padding:28px;
border-radius:18px;
border:1px solid {footer_border};
text-align:center;
">

<div style="
font-size:52px;
margin-bottom:10px;
">
🧠
</div>

<div style="
font-size:30px;
font-weight:700;
color:{footer_text};
">
Brain Tumor Classification
</div>

<div style="
margin-top:8px;
font-size:16px;
color:{footer_sub};
">

Deep Learning Based MRI Brain Scan Classification System

</div>

<hr style="margin-top:20px;margin-bottom:20px;">

<div style="
display:flex;
justify-content:center;
gap:30px;
flex-wrap:wrap;
font-size:15px;
color:{footer_sub};
">

<span>TensorFlow</span>

<span>Keras</span>

<span>Streamlit</span>

<span>Plotly</span>

<span>Python</span>

</div>

<div style="
margin-top:25px;
font-size:15px;
color:{footer_sub};
">

Developed by

<b style="color:{footer_text};">
Udai Pratap Singh
</b>

</div>

<div style="
margin-top:8px;
font-size:14px;
color:{footer_sub};
">

© 2026 All Rights Reserved

</div>

</div>

""", unsafe_allow_html=True)