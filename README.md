# 🩺 MultiFusion Breast Cancer Detection Framework

<div align="center">

**A Deep Learning Framework for Breast Cancer Image Classification Across Multiple Imaging Modalities**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Methodology](#-methodology)
- [Experimental Results](#-experimental-results)
- [Application](#️-application)
- [Model Security & Deployment](#-model-security--deployment)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Live Deployment](#-live-deployment)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [Research](#-research)
- [Author](#️-author)
- [License](#-license)

---

## 📖 Overview

**MultiFusion Breast Cancer Detection Framework** is a deep learning-based framework for automated breast cancer image classification across multiple medical imaging modalities.

The framework combines **Self-Supervised Learning (SSL)** using **SimCLR** with **Swin Transformer-based feature learning** to explore robust visual representation learning in scenarios where large-scale annotated medical imaging data can be difficult and expensive to obtain.

The framework is evaluated across three diverse breast cancer imaging datasets:

- **BUSI** — Breast Ultrasound Images
- **BreakHis** — Breast Histopathology Images
- **INbreast** — Digital Mammography

By evaluating the framework across different imaging modalities, the project investigates the robustness and generalization of deep learning-based breast cancer classification beyond a single dataset or imaging modality.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## ✨ Key Features

- 🧠 **Self-Supervised Learning** using SimCLR
- 🔬 **Swin Transformer-based visual feature learning**
- 🩻 Support for multiple breast imaging modalities
- 📊 Evaluation across BUSI, BreakHis, and INbreast
- 🔄 Focus on cross-dataset generalization
- 🖥️ Interactive Streamlit-based application
- 🔐 Private model-weight hosting using Hugging Face
- ⚡ Deep learning-based image inference
- 🌐 Deployment through Streamlit Community Cloud

---

## 🗂️ Datasets

The framework was evaluated using three publicly available breast cancer imaging datasets representing different medical imaging modalities.

### 🩻 BUSI

**Breast Ultrasound Images Dataset**

- **Modality:** Ultrasound
- **Classes:** Benign, Malignant, Normal
- **Task:** Breast lesion classification

### 🔬 BreakHis

**Breast Cancer Histopathological Image Dataset**

- **Modality:** Histopathology
- **Classes:** Benign, Malignant
- **Task:** Breast cancer classification from histopathological images

### 🩺 INbreast

**INbreast Digital Mammography Dataset**

- **Modality:** Mammography
- **Task:** Breast cancer-related image analysis

The use of multiple datasets enables evaluation across substantially different imaging modalities and helps investigate the generalization of the proposed learning approach.

---

## 🧠 Methodology

The framework combines self-supervised representation learning with transformer-based visual feature extraction.

### 1. Self-Supervised Representation Learning

**SimCLR-based self-supervised learning** is used to learn meaningful visual representations from medical images without relying exclusively on manually annotated data during representation learning.

This approach aims to reduce dependence on large-scale annotated datasets and learn useful visual characteristics for downstream classification.

### 2. Transformer-Based Feature Learning

**Swin Transformer** is used for hierarchical visual feature extraction, enabling the framework to capture both local and broader spatial characteristics present in breast medical images.

### 3. Dataset-Specific Classification

The learned visual representations are used for downstream classification across the supported breast imaging datasets.

### High-Level Pipeline

```text
Medical Images
       │
       ▼
Self-Supervised Representation Learning
       │
       ▼
Learned Visual Representations
       │
       ▼
Swin Transformer-based Feature Learning
       │
       ▼
Dataset-Specific Classification
       │
       ▼
Prediction
```

> **Research Note:** Detailed architectural configurations, training strategies, hyperparameters, ablation studies, and other research-specific implementation details are intentionally not documented in this repository.

---

## 📊 Experimental Results

The framework achieved the following classification accuracy on the evaluated datasets:

| Dataset | Imaging Modality | Accuracy |
|---|---|---:|
| **BreakHis** | Histopathology | **98.04%** |
| **BUSI** | Ultrasound | **96.87%** |


These results demonstrate strong classification performance across three heterogeneous breast imaging datasets and provide an initial evaluation of the framework across different imaging modalities.

> **Note:** Reported results correspond to the experimental setup used during project evaluation. Further validation on independent clinical datasets would be required before considering clinical applications.

---

## 🖥️ Application

The project provides an interactive **Streamlit** interface for demonstrating breast cancer image classification.

### 🏠 Home

The application provides:

- Project overview
- Framework information
- Dataset information
- Navigation to prediction modules
- Research and usage information

### 🔬 BreakHis Prediction

Users can:

- Upload a histopathology image
- Process the image using the trained model
- Obtain the predicted class
- View prediction confidence

**Supported classes:**

```text
Benign
Malignant
```

### 🩻 BUSI Prediction

Users can:

- Upload a breast ultrasound image
- Process the image using the trained model
- Obtain the predicted class
- View prediction confidence

**Supported classes:**

```text
Benign
Malignant
Normal
```

---

## 🔐 Model Security & Deployment

To keep trained model artifacts separate from the public application repository:

- ✅ Model weights are **not stored on GitHub**
- ✅ Trained model weights are hosted in a **private Hugging Face repository**
- ✅ Model access is controlled using a Hugging Face access token
- ✅ Credentials are managed through **Streamlit Secrets**
- ✅ Dataset files are excluded from the public repository
- ✅ Training notebooks and experimental files remain private
- ✅ The public repository contains the application and deployment code

The application downloads the required model weights from the private Hugging Face repository during runtime.

### Deployment Architecture

```text
                 Public GitHub
                      │
                      ▼
              Streamlit Application
                      │
                      │ HF Authentication
                      ▼
              Private Hugging Face
                      │
             ┌────────┴────────┐
             ▼                 ▼
      BreakHis Model      BUSI Model
             │                 │
             └────────┬────────┘
                      ▼
                  Inference
                      │
                      ▼
                  Prediction
```

> 🔒 Model weights are intentionally excluded from this repository.

---

## 📁 Project Structure

The public repository contains only the files required to run and deploy the application:

```text
MultiFusion-Breast-Cancer-Detection-Framework/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Deployment dependencies
├── README.md              # Project documentation
└── .gitignore             # Excluded files and directories
```

Private/local development resources include:

```text
Dataset/
Models/
Notebooks/
.streamlit/secrets.toml
requirements-local.txt
```

These resources are intentionally excluded from the public repository.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Git
- Hugging Face account with access to the private model repository

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Varundube99/MultiFusion-Breast-Cancer-Detection-Framework.git
cd MultiFusion-Breast-Cancer-Detection-Framework
```

#### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scriptsctivate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Hugging Face access

Create:

```text
.streamlit/secrets.toml
```

Add your Hugging Face read token:

```toml
HF_TOKEN = "your_hugging_face_read_token"
```

> ⚠️ Never commit `secrets.toml` or expose your Hugging Face access token publicly.

#### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🌐 Live Deployment

The application is deployed using **Streamlit Community Cloud**.

### 🔗 Live Demo

[MultiFusion Breast Cancer Detection Framework](https://multifusion-breast-cancer-detection-framework.streamlit.app/)

The deployed application retrieves the required private model weights using securely configured Streamlit Secrets.

> The live application is intended for demonstration and research purposes only.

---

## ⚠️ Limitations

- The framework is intended for research and educational purposes.
- Dataset performance does not necessarily represent performance in real-world clinical environments.
- Medical imaging datasets may differ in acquisition protocols, equipment, demographics, and image characteristics.
- Further validation on independent and clinically representative datasets is required.
- Model predictions should not be interpreted as medical diagnoses.
- The current application focuses on image-based classification rather than complete clinical decision support.
- Performance may vary on images that differ substantially from the training and evaluation datasets.

---

## 🚀 Future Work

Potential future directions include:

- [ ] Extensive cross-dataset validation
- [ ] Evaluation on larger and more diverse datasets
- [ ] Improved domain adaptation across imaging modalities
- [ ] Explainable AI for model predictions
- [ ] Uncertainty estimation and model calibration
- [ ] Additional self-supervised learning strategies
- [ ] External validation using independent clinical datasets
- [ ] Multimodal breast cancer analysis
- [ ] Real-time inference optimization
- [ ] Integration of additional medical imaging datasets

---

## 📚 Research

**MultiFusion Breast Cancer Detection Framework** is being developed as part of ongoing research into **self-supervised learning, transformer-based computer vision, and medical image analysis**.

The project investigates self-supervised representation learning and transformer-based feature extraction for breast cancer image classification across multiple imaging modalities.

Detailed research components, including:

- Architectural design choices
- Training methodology
- Ablation studies
- Detailed experimental analysis
- Comparative experiments
- Implementation-specific configurations

are being reserved for the associated research work.

---

## 👨‍💻 Author

**Varun Dubey**

🎓 B.Tech Computer Science Engineering  
🤖 AI / ML & Deep Learning  
🔬 Research Enthusiast

### GitHub
**[@Varundube99](https://github.com/Varundube99)**

### Mail
📧 **Email:** varundube99@gmail.com

---

## 📜 License

This project is intended for **educational and research purposes**.

- Application code is available through this repository.
- Trained model weights are intentionally excluded from the public repository.
- Dataset ownership and licensing remain with the respective dataset providers.
- Users are responsible for complying with the licenses and terms associated with the datasets used by the project.

---

<div align="center">

⭐ **If you find this project interesting, consider starring the repository!** ⭐

</div>
