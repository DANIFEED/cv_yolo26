import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import io
from ultralytics import YOLO
import os

# --------------------------------------------------
# Face Blur Function
# --------------------------------------------------
def blur_faces(image, boxes, blur_strength=99):
    """Apply Gaussian blur to detected faces"""
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within image bounds
        h, w = img_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract face region
        face_roi = img_array[y1:y2, x1:x2]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 30)
        
        # Replace face with blurred version
        img_array[y1:y2, x1:x2] = blurred
    
    return img_array
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

st.title("🙂 Face Detection & Anonymization")
st.caption("Детекция лиц с последующим маскированием области")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Inference Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 1.0, 0.25, 0.05
)
blur_strength = st.sidebar.slider(
    "Blur Strength",
    min_value=21,
    max_value=151,
    value=99,
    step=10,
    help="Higher = more blur"
)

MODEL_PATH = "models/face_best.pt"
# --------------------------------------------------
# Upload Section
# --------------------------------------------------
st.subheader("👉 Input & Model Overview")
tab1, tab2, tab3 = st.tabs([
    "Upload Files",
    "Load from URL",
    "Model Results"
])

# Отдельные списки для разных типов загрузки
upload_images = []  # Для файлов из Upload
url_images = []     # Для изображений из URL

# --------------------------------------------------
# TAB 1 — Upload Files
# --------------------------------------------------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="upload_files"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            upload_images.append((file.name, img))
    
    # Показываем predictions только для загруженных файлов
    if upload_images:
        st.divider()
        st.subheader("🔍 Predictions (Upload Files)")
        
        try:
            model = YOLO(MODEL_PATH)
        except:
            st.error("Model weights not found. Проверь путь к face_best.pt.")
            st.stop()
        
        for name, img in upload_images:
            with st.expander(f"📄 {name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.caption("📷 Original")
                    with st.container(border=True):
                        st.image(img, use_container_width=True)
                
                with col2:
                    st.caption("🎯 Detection")
                    with st.container(border=True):
                        results = model.predict(img, conf=confidence, verbose=False)
                        plotted = results[0].plot()
                        plotted = plotted[..., ::-1]
                        st.image(plotted, use_container_width=True)
                
                with col3:
                    st.caption("🔒 Anonymized")
                    with st.container(border=True):
                        boxes = []
                        if results[0].boxes is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                        blurred_img = blur_faces(img.copy(), boxes, blur_strength)
                        st.image(blurred_img, use_container_width=True)
            
            st.divider()
# --------------------------------------------------
# TAB 2 — Load from URL
# --------------------------------------------------
with tab2:
    url = st.text_input("Direct image URL", key="url_input")
    
    if st.button("Load from URL", key="load_url_btn"):
        if url:
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                url_images.append(("URL image", img))
                st.success("✅ Image loaded successfully!")
            except Exception as e:
                st.error(f"⚠ Не удалось загрузить изображение: {e}")
    
    # Показываем predictions только для URL изображений
    if url_images:
        st.divider()
        st.subheader("🔍 Predictions (URL)")
        
        try:
            model = YOLO(MODEL_PATH)
        except:
            st.error("Model weights not found. Проверь путь к face_best.pt.")
            st.stop()
        
        for name, img in url_images:
            with st.expander(f"📄 {name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.caption("📷 Original")
                    with st.container(border=True):
                        st.image(img, use_container_width=True)
                
                with col2:
                    st.caption("🎯 Detection")
                    with st.container(border=True):
                        results = model.predict(img, conf=confidence, verbose=False)
                        plotted = results[0].plot()
                        plotted = plotted[..., ::-1]
                        st.image(plotted, use_container_width=True)
                
                with col3:
                    st.caption("🔒 Anonymized")
                    with st.container(border=True):
                        boxes = []
                        if results[0].boxes is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                        blurred_img = blur_faces(img.copy(), boxes, blur_strength)
                        st.image(blurred_img, use_container_width=True)
                
            st.divider()
    
    elif not url:
        st.info("👆 Введите URL изображения и нажмите 'Load from URL'")

# --------------------------------------------------
# TAB 3 — Model Results (Model Information + curves)
# --------------------------------------------------
with tab3:
    st.subheader("📊 Model Information")

    st.markdown("""
    <div class="card">
    <div class="small-text">
    • Model: <b>YOLOv8m</b><br>
    • Epochs trained: <b>30</b><br>
    • Train size: <b>13,386</b><br>
    • Validation size: <b>3,347</b><br>
    <br>
    <div class="metric">Box(P): 0.898</div>
    <div class="metric">Box(R): 0.814</div>
    <div class="metric">mAP50: 0.894</div>
    <div class="metric">mAP50-95: 0.603</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # TRAINING CURVES
    # -------------------------
    st.markdown("### 📈 Training Curves")
    
    col1, col2 = st.columns(2)
    
    loss_path = "images/face_cv_outputs/results.png"
    pr_path = "images/face_cv_outputs/BoxPR_curve.png"
    
    with col1:
        with st.container(border=True):
            if os.path.exists(loss_path):
                st.image(loss_path, caption="Loss Curve", use_container_width=True)
            else:
                st.info("⚠️ Loss curve not found")

    with col2:
        with st.container(border=True):
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)
            else:
                st.info("⚠️ PR curve not found")

    # -------------------------
    # CONFUSION MATRIx
    # -------------------------
    st.markdown("### 🎯 Confusion Matrix")
    cm_path = "images/face_cv_outputs/confusion_matrix_normalized.png"
    
    with st.container(border=True):
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Normalized Confusion Matrix", use_container_width=True)
        else:
            st.info("⚠️ Confusion matrix not found")
