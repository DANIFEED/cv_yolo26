import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import io
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


MEAN = (0.485, 0.456, 0.406) 
STD = (0.229, 0.224, 0.225)
DEVICE = "cpu"



# 1. Загрузка модели (используем ваши параметры)
@st.cache_resource
def load_model(model_type, path):
    # Выбираем класс архитектуры в зависимости от выбора пользователя
    arch_class = smp.Unet if model_type == "Unet" else smp.MAnet
    
    model = arch_class(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        # decoder_attention_type есть только в Unet в SMP, 
        # для MAnet этот параметр можно убрать или оставить только для Unet
        **(dict(decoder_attention_type='scse') if model_type == "Unet" else {})
    )
    
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# препроцессинг изображения
def process_image(image, target_size=(256, 256)):
    img_np = np.array(image)
    transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    tensor = transform(image=img_np)["image"]
    return tensor.unsqueeze(0).to(DEVICE)

def show_predictions(current_images):
    if current_images:
        try:
            model = load_model(selected_arch, MODEL_PATH)
        except:
            st.error("Ошибка загрузки модели.")
            return

        st.subheader(f"Predictions ({selected_arch})")
        for name, img in current_images:
            col1, col2 = st.columns(2)
            # ... далее ваш код отрисовки без изменений ...
            with col1:
                st.markdown("##### Original Image")
                st.image(img, use_container_width=True)
            with col2:
                # (весь ваш код обработки изображения)
                input_tensor = process_image(img)
                model.eval()
                with torch.no_grad():
                    probs = model(input_tensor).sigmoid().cpu().squeeze().numpy()
                    pred = (probs > confidence)
                    mean_conf = probs[pred].mean() if pred.any() else 0.0
                mask_resized = cv2.resize(pred.astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST)
                img_np = np.array(img)
                overlay = img_np.copy()
                overlay[mask_resized > 0] = [34, 34, 180]
                result_img = cv2.addWeighted(img_np, 0.5, overlay, 0.5, 0)
                st.markdown(f"##### Predict (Conf: {mean_conf:.2%})")
                st.image(result_img, use_container_width=True)
    else:
        st.info("Загрузите изображения для инференса.")


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

st.title("🛰 Aerial Image Segmentation")
st.caption("Сегментация объектов на аэроснимках (леса)")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Inference Settings")

selected_arch = st.sidebar.selectbox("Select Architecture", ["Unet", "MAnet"])

confidence = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.5, 0.05)


# ПУТИ К ВЕСАМ
if selected_arch == "Unet":
    MODEL_PATH = "models/model_unet_dan_weights_final.pt"
else:
    MODEL_PATH = "models/model_manet_dan_weights_final.pt"

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
    uploaded_files = st.file_uploader("Upload...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    files_images = [] # Локальный список для этой вкладки
    if uploaded_files:
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            files_images.append((file.name, img))
    show_predictions(files_images)

# --------------------------------------------------
# TAB 2 — Load from URL
# --------------------------------------------------
with tab2:
    url = st.text_input("Direct image URL")
    url_images = [] # Локальный список для этой вкладки
    if st.button("Load from URL"):
        if url:
            try:
                response = requests.get(url)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                url_images.append(("URL image", img))
                st.session_state['url_img'] = url_images # Сохраняем, чтобы не исчезло при нажатии слайдера
            except:
                st.error("⚠ Ошибка URL")
    
    # Выводим сохраненное в сессии изображение (чтобы работало со слайдером)
    show_predictions(st.session_state.get('url_img', []))


# --------------------------------------------------
# TAB 3 — Model Results
# --------------------------------------------------

with tab3:

    st.subheader("Model Information")

    st.markdown("""
    <div class="card">
    <div class="small-text">

    • Model:                    UNET vs MANET (encoder: efficientnet-b3, loss: DiceLoss)
    <br>
    • Epochs trained:           <b>30</b>
    <br>
    • Train size:               <b>4086</b> 
    <br>
    • Validation size:          <b>715</b>
    <br>
    • Validation size:          <b>307</b>
    <br>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Metrics score")
    col01, col02 = st.columns(2)
    with col01:
        st.markdown("##### *UNET*")
        st.markdown("###### IoU:    0.792")        
        st.markdown("###### Loss:   0.117")   
        st.markdown("###### F1:     0.883")   

    with col02:
        st.markdown("##### *MANET*")
        st.markdown("###### IoU (valid / test):    0.790 / 0.7697")
        st.markdown("###### Loss (valid / test):   0.119 / 0.1329")
        st.markdown("###### F1 (valid / test):     0.881 / 0.8679")

    # -------------------------
    # TRAINING CURVES
    # -------------------------

    st.markdown("### Training Curves")

    col1, col2, col3 = st.columns(3)

    loss_path = "models/metrics_aerial/loss_metrics_unet_manet.png" 
    iou_path = "models/metrics_aerial/iou_metrics_unet_manet.png"
    f1_path = "models/metrics_aerial/f1_metrics_unet_manet.png"
    cm_unet_path = "models/metrics_aerial/unet_efi_cm.png"
    cm_manet_path = "models/metrics_aerial/manet_efi_cm.png"


    with col1:
        if os.path.exists(loss_path):
            st.image(loss_path, caption="Loss Curve")
        else:
            st.info("Loss curve not available.")

    with col2:
        if os.path.exists(iou_path):
            st.image(iou_path, caption="IoU Curve")
        else:
            st.info("IoU curve not available.")

    with col3:
        if os.path.exists(f1_path):
            st.image(f1_path, caption="F1 Curve")
        else:
            st.info("F1 curve not available.")

    st.markdown("### Confusion Matrix")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("##### Confusion Matrix Unet (Test)")
        if os.path.exists(cm_unet_path):
            st.image(cm_unet_path)
        else:
            st.info("Confusion matrix not available.")

    with col5:
        st.markdown("##### Confusion Matrix MANet (Test)")
        if os.path.exists(cm_manet_path):
            st.image(cm_manet_path)
        else:
            st.info("Confusion matrix not available.")


# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.divider()

