from huggingface_hub import hf_hub_download
import streamlit as st
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from streamlit_option_menu import option_menu

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Breast Cancer Detection",
    layout="wide"
)

device = torch.device("cpu")

# =========================
# GLOBAL CSS 
# =========================
st.markdown("""
<style>

/* Remove default top padding */
.block-container {
    padding-top: 0.5rem !important;
}

/* Remove extra gap above everything */
section.main > div {
    padding-top: 0rem !important;
}

/* Optional: tighten overall spacing */
div[data-testid="stAppViewContainer"] {
    margin-top: -10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>


.stButton > button {
    background: linear-gradient(135deg, #ec4899, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Hover */
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(236,72,153,0.3);
}

/* Remove black focus outline */
.stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}




/* Main uploader box */
[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, #f3e8ff, #ede9fe) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    border: 2px dashed #c084fc !important;
}


[data-testid="stFileUploader"] > div {
    background: transparent !important;
}


[data-testid="stFileUploader"] section {
    background: transparent !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}


[data-testid="stFileUploader"] small {
    color: #6b7280 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>


.header-title {
    font-size: 44px;
    font-weight: 800;
    line-height: 1.2;

    margin-top: 40px;  
    margin-bottom: 2px;

    background: linear-gradient(90deg, #6366f1, #9333ea, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow: 0px 4px 20px rgba(99,102,241,0.25);
}


.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 20px;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

header[data-testid="stHeader"] { display: none; }

[data-testid="stSidebar"] { display: none; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #eef2f7,
        #e5eaf3
    );
}

.main .block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: #111827;
}

p, span, label, div {
    color: #1f2937;
}

.header-title {
    font-size: 38px;
    font-weight: 700;
    color: #111827;
}

.subtitle {
    font-size: 18px;
    color: #374151;
}

.card {
    background: linear-gradient(
        145deg,
        #ffffff,
        #f8fafc
    );
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 22px;
    border: 1px solid rgba(0,0,0,0.05);
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.12);
}

.upload-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

.image-card {
    background: white;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.prediction-card {
    background: linear-gradient(
        135deg,
        #10b981,
        #059669
    );
    padding: 24px;
    border-radius: 16px;
    color: white;
    font-size: 22px;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}



.stProgress > div > div {
    background-color: #2563eb;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>


html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
}

[data-testid="stAppViewContainer"] > .main {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}


.block-container {
    flex: 1;
}


.footer {
    text-align: center;
    padding: 20px 10px;
    margin-top: 40px;
    font-size: 14px;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>


pre, code {
    background: transparent !important;
    color: inherit !important;
    padding: 0 !important;
    border-radius: 0 !important;
    font-size: inherit !important;
}


.stMarkdown pre {
    background: transparent !important;
}


.stMarkdown {
    overflow: visible !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LABEL MAPS
# =========================

BREAKHIS_LABELS = {
    0: "Benign",
    1: "Malignant"
}

BUSI_LABELS = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}

# =========================
# PREPROCESSING
# =========================

def apply_clahe(image):

    image = np.array(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2RGB
    )

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# MODEL LOADERS
# =========================

HF_REPO_ID = "dubevarun/BreastCancer"


@st.cache_resource
def load_breakhis_model():

    token = st.secrets["HF_TOKEN"]

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="breakhis_model.pth",
        token=token
    )

    encoder = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=0
    )

    model = nn.Sequential(
        encoder,
        nn.Linear(
            encoder.num_features,
            256
        ),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(
            256,
            2
        )
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    return model


@st.cache_resource
def load_busi_model():

    token = st.secrets["HF_TOKEN"]

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="busi_model.pth",
        token=token
    )

    encoder = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=0
    )

    class BUSIModel(nn.Module):

        def __init__(self, encoder):

            super().__init__()

            self.encoder = encoder

            self.classifier = nn.Sequential(

                nn.Linear(
                    encoder.num_features,
                    512
                ),

                nn.ReLU(),

                nn.Dropout(0.4),

                nn.Linear(
                    512,
                    3
                )
            )

        def forward(self, x):

            features = self.encoder(x)

            out = self.classifier(features)

            return out

    model = BUSIModel(encoder)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    return model

# =========================
# PREDICTION
# =========================

def predict(image, model, labels):

    img = transform(image)

    img = img.unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    return (
        labels[pred.item()],
        confidence.item(),
        probs.numpy()[0]
    )

# =========================
# TOP NAVIGATION 
# =========================

PAGE_OPTIONS = [
    "Home",
    "BreakHis Prediction",
    "BUSI Prediction",
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

selected_page = option_menu(
    menu_title=None,
    options=PAGE_OPTIONS,
    icons=[
        "house-fill",
        "activity",
        "heart-pulse-fill",
        "bar-chart-fill",
        "question-circle-fill"
    ],
    orientation="horizontal",
    default_index=PAGE_OPTIONS.index(st.session_state.page),
    styles={
    "container": {
        "background-color": "#E9ECEF",
        "padding": "2px 0px",              
        "margin-bottom": "5px",           
        "border-radius": "0px",
        "box-shadow": "none",
        "position": "static",
        "width": "100vw",
        "margin-left": "calc(50% - 50vw)",
        "margin-right": "calc(50% - 50vw)",
        "overflow-x": "auto",
        "white-space": "nowrap"
    },

    "icon": {
        "color": "#666666",
        "font-size": "0.95rem"            
    },

    "nav-link": {
        "font-size": "0.95rem",            
        "text-align": "center",
        "margin": "0px 4px",
        "padding": "6px 12px",            
        "color": "#333333",
        "border-radius": "6px",
        "transition": "all 0.25s ease"
    },

    "nav-link-selected": {
        "background-color": "#FFFFFF",
        "color": "#000000",
        "font-weight": "600",
        "border-bottom": "2px solid #7B4BFF"   
    }
}
)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

page = st.session_state.page

# =========================
# HOME PAGE
# =========================

if page == "Home":

    # ---- Local CSS ----
    st.markdown("""
    <style>
    .home-card {
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        padding: 26px;
        border-radius: 18px;
        margin-bottom: 25px;

        border: 1px solid #c4b5fd;
        border-left: 5px solid #7c3aed;

        box-shadow: 0 8px 20px rgba(124,58,237,0.12);
    }

    .tag {
        display:inline-block;
        background:#ede9fe;
        padding:6px 12px;
        border-radius:10px;
        margin:6px 6px 0 0;
        font-size:14px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
    '<div class="header-title">AI-Based Breast Cancer Detection System</div>',
    unsafe_allow_html=True
    )

    st.divider()

    # =========================
    # WELCOME
    # =========================

    st.markdown("""
    <div style="font-size:26px; font-weight:700; color:#db2777;">
    👋 Welcome!
    </div>

    <div style="width:60px;height:4px;background:#ec4899;margin:12px 0 20px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">

    <div style="font-weight:700; color:#7c3aed; margin-bottom:10px;">
    💡 Overview
    </div>

    <div style="color:#4b5563;">
    This platform uses <b>deep learning models</b> to analyze breast cancer images
    and provide <span style="color:#7c3aed;font-weight:600;">fast, reliable predictions</span>.
    </div>

    <br>

    <div style="font-weight:700; color:#7c3aed;">
    🚀 Purpose
    </div>

    <div style="margin-top:8px;">
    ✔ Early detection & diagnosis<br>
    ✔ Awareness through visual analysis<br>
    ✔ AI-assisted research insights
    </div>

    </div>
    """, unsafe_allow_html=True)


    # =========================
    # UNDERSTANDING
    # =========================

    st.markdown("""
    <div style="font-size:24px;font-weight:700;color:#db2777;">
    🧬 Understanding Breast Cancer
    </div>

    <div style="width:50px;height:3px;background:#ec4899;margin:10px 0 15px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">

    <div>
    Breast cancer occurs when abnormal cells grow uncontrollably,
    forming tumors that may spread if not detected early.
    </div>

    <br>

    <div>
    It is one of the most common cancers worldwide.
    <span style="color:#7c3aed;font-weight:600;">
    Early detection significantly improves survival outcomes.
    </span>
    </div>

    <br>

    <div style="font-weight:700;color:#db2777;">
    ⚠️ Common Risk Factors
    </div>

    <div style="margin-top:10px;">
    <span class="tag">✔ Genetic mutations</span>
    <span class="tag">✔ Family history</span>
    <span class="tag">✔ Lifestyle factors</span>
    <span class="tag">✔ Increasing age</span>
    </div>

    <br>

    <div style="
        background:#ede9fe;
        padding:12px;
        border-radius:10px;
        border-left:4px solid #7c3aed;
        font-size:14px;
    ">
    📌 Early screening improves detection accuracy and outcomes.
    </div>

    </div>
    """, unsafe_allow_html=True)


    # =========================
    # HOW SYSTEM HELPS
    # =========================

    st.markdown("""
    <div style="font-size:24px;font-weight:700;color:#db2777;">
    🤖 How This System Helps
    </div>

    <div style="width:50px;height:3px;background:#ec4899;margin:10px 0 15px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">

    <div style="font-weight:700;color:#7c3aed;">
    ⚙️ Capabilities
    </div>

    <div style="margin-top:10px;">
    ✔ Analyze histopathology & ultrasound images<br>
    ✔ Predict benign / malignant / normal<br>
    ✔ Provide confidence scores
    </div>

    <br>

    <div style="background:#ede9fe;padding:10px;border-radius:10px;">
    📊 Designed for research & education
    </div>

    </div>
    """, unsafe_allow_html=True)


    # =========================
    # WHAT YOU CAN DO
    # =========================

    st.markdown("""
    <div style="font-size:24px;font-weight:700;color:#db2777;">
    📌 What You Can Do
    </div>

    <div style="width:50px;height:3px;background:#ec4899;margin:10px 0 15px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">

    📤 Upload images<br>
    ⚡ Instant predictions<br>
    📊 Confidence scores<br>
    📈 Probability analysis

    </div>
    """, unsafe_allow_html=True)


    # =========================
    # DATASETS
    # =========================

    # ---- BreakHis ----
    st.markdown("""
    <div style="font-size:24px;font-weight:700;color:#db2777;">
    🧫 BreakHis Dataset
    </div>

    <div style="width:50px;height:3px;background:#ec4899;margin:10px 0 15px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">

    <div style="margin-bottom:10px;">
    🔬 <b>Type:</b> Histopathology imaging
    </div>

    <div style="margin-bottom:10px;">
    🧪 <b>Use:</b> Microscopic tissue analysis
    </div>

    <div style="font-weight:600;margin-bottom:6px;">
    📊 Classes:
    </div>

    <span class="tag">Benign</span>
    <span class="tag">Malignant</span>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("Dataset/breakhis.png", width=650)


    # ---- BUSI ----
    st.markdown("""
    <div style="font-size:24px;font-weight:700;color:#db2777;">
    🩺 BUSI Dataset
    </div>

    <div style="width:50px;height:3px;background:#ec4899;margin:10px 0 15px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">

    <div style="margin-bottom:10px;">
    📡 <b>Type:</b> Ultrasound imaging
    </div>

    <div style="margin-bottom:10px;">
    🏥 <b>Use:</b> Real clinical scenarios
    </div>

    <div style="font-weight:600;margin-bottom:6px;">
    📊 Classes:
    </div>

    <span class="tag">Benign</span>
    <span class="tag">Malignant</span>
    <span class="tag">Normal</span>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("Dataset/Busi.png", width=650)

# =========================
# BREAKHIS PAGE
# =========================

elif page == "BreakHis Prediction":

    # -------------------------
    # PAGE TITLE
    # -------------------------
    st.markdown("""
    <div style="
        font-size:34px;
        font-weight:700;
        color:#ec4899;
        margin-bottom:10px;
    ">
    🧬 BreakHis Prediction
    </div>
    """, unsafe_allow_html=True)


    # -------------------------
    # UPLOAD SECTION (CLEAN)
    # -------------------------
    st.markdown("""
    <div style="
        background: linear-gradient(145deg, #f3e8ff, #ede9fe);
        padding:18px;
        border-radius:16px;
        border-left:6px solid #ec4899;
        margin-bottom:20px;
    ">
        <div style="font-weight:600; font-size:18px; margin-bottom:6px;">
        📤 Upload Histopathology Image
        </div>
        <div style="color:#6b7280; font-size:13px;">
        Supported: PNG, JPG, JPEG • Max size: 200MB
        </div>
    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )


    # -------------------------
    # IF IMAGE UPLOADED
    # -------------------------
    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("<br>", unsafe_allow_html=True)

        # CENTER IMAGE
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(image, caption="Uploaded Image", width=260)


        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------
        # RUN BUTTON
        # -------------------------
        run = st.button("🔍 Run Prediction")


        # -------------------------
        # PREDICTION
        # -------------------------
        if run:

            with st.spinner("Analyzing image... 🔬 Please wait"):
    
                model = load_breakhis_model()

                processed = apply_clahe(image)

                label, conf, probs = predict(
                    Image.fromarray(processed),
                    model,
                    BREAKHIS_LABELS
    )


            # -------------------------
            # RESULT CARD
            # -------------------------
            color = "#22c55e" if label.lower() == "benign" else "#ef4444"

            st.markdown(f"""<div style="background: linear-gradient(145deg, #f3e8ff, #ede9fe);
padding:20px;
border-radius:16px;
border-left:6px solid {color};
margin-top:20px;">

<div style="font-size:16px;font-weight:600;margin-bottom:8px;color:#374151;">
📊 Diagnosis
</div>

<div style="font-size:22px;font-weight:700;color:{color};margin-bottom:10px;">
{label}
</div>

<div style="font-size:14px;color:#6b7280;">
Confidence Score
</div>

<div style="font-size:20px;font-weight:600;color:#7c3aed;margin-bottom:12px;">
{conf*100:.2f}%
</div>

</div>""", unsafe_allow_html=True)


            # -------------------------
            # PROBABILITY SECTION
            # -------------------------
            st.markdown("""
            <div style="
                font-size:16px;
                font-weight:600;
                margin-top:20px;
                margin-bottom:10px;
                color:#374151;
            ">
            📈 Probability Distribution
            </div>
            """, unsafe_allow_html=True)


            # BENIGN
            st.markdown("**🟢 Benign**")
            st.progress(float(probs[0]))
            st.write(f"{probs[0]*100:.2f}%")


            # MALIGNANT
            st.markdown("**🔴 Malignant**")
            st.progress(float(probs[1]))
            st.write(f"{probs[1]*100:.2f}%")

# =========================
# BUSI PAGE
# =========================

elif page == "BUSI Prediction":

    # -------------------------
    # PAGE TITLE
    # -------------------------
    st.markdown("""
    <div style="
        font-size:34px;
        font-weight:700;
        color:#ec4899;
        margin-bottom:10px;
    ">
    🫀 BUSI Prediction
    </div>
    """, unsafe_allow_html=True)


    # -------------------------
    # UPLOAD SECTION (CLEAN)
    # -------------------------
    st.markdown("""
    <div style="
        background: linear-gradient(145deg, #f3e8ff, #ede9fe);
        padding:18px;
        border-radius:16px;
        border-left:6px solid #ec4899;
        margin-bottom:20px;
    ">
        <div style="font-weight:600; font-size:18px; margin-bottom:6px;">
        📤 Upload Ultrasound Image
        </div>
        <div style="color:#6b7280; font-size:13px;">
        Supported: PNG, JPG, JPEG • Max size: 200MB
        </div>
    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )


    # -------------------------
    # IF IMAGE UPLOADED
    # -------------------------
    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("<br>", unsafe_allow_html=True)

        # CENTER IMAGE
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(image, caption="Uploaded Image", width=260)


        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------
        # RUN BUTTON
        # -------------------------
        run = st.button("🔍 Run Prediction")


        # -------------------------
        # PREDICTION
        # -------------------------
        if run:

            with st.spinner("Analyzing image... 🔬 Please wait"):

                model = load_busi_model()

                label, conf, probs = predict(
                    image,
                    model,
                    BUSI_LABELS
                )


            # -------------------------
            # RESULT CARD
            # -------------------------
            color = "#22c55e" if label.lower() == "benign" else "#ef4444"

            st.markdown(f"""<div style="background: linear-gradient(145deg, #f3e8ff, #ede9fe);
padding:20px;
border-radius:16px;
border-left:6px solid {color};
margin-top:20px;">

<div style="font-size:16px;font-weight:600;margin-bottom:8px;color:#374151;">
📊 Diagnosis
</div>

<div style="font-size:22px;font-weight:700;color:{color};margin-bottom:10px;">
{label}
</div>

<div style="font-size:14px;color:#6b7280;">
Confidence Score
</div>

<div style="font-size:20px;font-weight:600;color:#7c3aed;margin-bottom:12px;">
{conf*100:.2f}%
</div>

</div>""", unsafe_allow_html=True)


            # -------------------------
            # PROBABILITY SECTION
            # -------------------------
            st.markdown("""
            <div style="
                font-size:16px;
                font-weight:600;
                margin-top:20px;
                margin-bottom:10px;
                color:#374151;
            ">
            📈 Probability Distribution
            </div>
            """, unsafe_allow_html=True)


            # BENIGN
            st.markdown("**🟢 Benign**")
            st.progress(float(probs[0]))
            st.write(f"{probs[0]*100:.2f}%")

            # MALIGNANT
            st.markdown("**🔴 Malignant**")
            st.progress(float(probs[1]))
            st.write(f"{probs[1]*100:.2f}%")

            # NORMAL
            st.markdown("**🔵 Normal**")
            st.progress(float(probs[2]))
            st.write(f"{probs[2]*100:.2f}%")


