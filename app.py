import streamlit as st
import base64

def load_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

brain_icon = load_base64("images/brain.jpg")
face_icon = load_base64("images/face.png")
aero_icon = load_base64("images/aero.jpg")

st.set_page_config(
    page_title="Computer Vision Project",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.card {
    background-color: #1f2937;
    padding: 2rem;
    border-radius: 14px;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: start;
    align-items: center;
    text-align: center;
}

.card img {
    width: 70px;
    margin-bottom: 12px;
    border-radius: 6px;
    opacity: 0.95;
}
            
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.card-text {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.35;
}

.subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #9ca3af;
    margin-bottom: 2rem;
}

.section-title {
    font-weight: 600;
    margin-top: 3rem;
}

.footer {
    font-size: 0.8rem;
    color: #6b7280;
    text-align: center;
    margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown("<h1>Computer Vision Project</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>YOLO Detection • UNet Segmentation • Multipage Streamlit Service</div>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Cards with icons
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <img src="data:image/jpg;base64,{brain_icon}">
        <div class="card-title">🧠 Brain Tumor Detection</div>
        <div class="card-text">      
            Детекция опухолей мозга по МРТ-снимкам.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <img src="data:image/png;base64,{face_icon}">
        <div class="card-title">🙂 Face Detection & Anonymization</div>
        <div class="card-text">
            Маскирование детектированной области.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <img src="data:image/jpg;base64,{aero_icon}">
        <div class="card-title">🛰️ Aerial Image Segmentation</div>
        <div class="card-text">
            Сегментация аэрокосмических снимков.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Features
# --------------------------------------------------

st.markdown("<div class='section-title'>Service Features / Функциональность</div>", unsafe_allow_html=True)

st.markdown("""
- Multi-file upload on all pages  
- Direct URL loading  
- Visualization of predictions  
- Блок с информацией о модели и процессе обучения  
- Метрики качества (mAP, PR curve, Confusion Matrix, IoU, Dice)  
""")

st.markdown("<div class='footer'>Phase 2 • Week 2 • CV Team Project</div>", unsafe_allow_html=True)
