import streamlit as st
from PIL import Image
import requests
import io
from ultralytics import YOLO
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(layout="wide")

# --------------------------------------------------
# Custom style 
# --------------------------------------------------

st.markdown("""
<style>

.card {
    background-color: #1f2937;
    padding: 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}

.metric {
    font-size: 1.1rem;
    font-weight: 600;
}

.small-text {
    font-size: 0.9rem;
    color: #cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧠 Brain Tumor Detection")
st.caption("Детекция опухолей мозга по МРТ-снимкам")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Inference Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 1.0, 0.25, 0.05
)

MODEL_PATH = "models/best_brain.pt"

# --------------------------------------------------
# Upload Section 
# --------------------------------------------------

st.subheader("👉 Input & Model Overview")

tab1, tab2, tab3 = st.tabs([
    "Upload Files",
    "Load from URL",
    "Model Results"
])

images = []

# --------------------------------------------------
# TAB 1 — Upload Files
# --------------------------------------------------

with tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            images.append((file.name, img))

# --------------------------------------------------
# TAB 2 — Load from URL
# --------------------------------------------------

with tab2:
    url = st.text_input("Direct image URL")
    if st.button("Load from URL"):
        if url:
            try:
                response = requests.get(url)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                images.append(("URL image", img))
            except:
                st.error("⚠ Не удалось загрузить изображение. Проверь URL.")

# --------------------------------------------------
# TAB 3 — Model Results
# --------------------------------------------------

with tab3:

    st.subheader("Model Information")

    st.markdown("""
    <div class="card">
    <div class="small-text">

    • Model: <b>YOLOv11-L</b><br>
    <br>
    • Epochs trained: <b> 148 (Early Stopping)</b>
    <br>
    • Train size: <b>255 images</b>
    <br>
    • Validation size: <b>63 images</b>

    <br><br>

    <div class="metric">Precision: <b>0.74</b></div>
    <div class="metric">Recall: <b>0.769</b></div>
    <div class="metric">mAP50: <b>0.804</b></div>
    <div class="metric">mAP50-95: <b>0.603</b></div>

    </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics images
    st.markdown("### Training Curves")

    loss_path = "models/metrics_brain/loss_curve.jpg"
    pr_path   = "models/metrics_brain/pr_curve.jpg"
    cm_path   = "models/metrics_brain/confusion_matrix.jpg"

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(loss_path):
            st.image(loss_path, caption="Loss Curve")
        else:
            st.info("Loss curve not available.")

    with col2:
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall Curve")
        else:
            st.info("PR curve not available.")

    st.markdown("### Confusion Matrix")
    if os.path.exists(cm_path):
        st.image(cm_path)
    else:
        st.info("Confusion matrix not available.")


# --------------------------------------------------
# --------------------------------------------------

st.divider()
st.subheader("Predictions")

if not images:
    st.info("Загрузите изображения во вкладках Upload или URL.")
else:
    try:
        model = YOLO(MODEL_PATH)
    except:
        st.error("Model weights not found. Проверь путь к best_brain.pt.")
        st.stop()

    for name, img in images:
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Original")
            st.image(img, use_container_width=True)

        with col2:
            st.caption("Detection Result")
            results = model.predict(img, conf=confidence, verbose=False)
            plotted = results[0].plot()[..., ::-1]
            st.image(plotted, use_container_width=True)