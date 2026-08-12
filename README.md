# MultiFusion Breast Cancer Detection Framework

An AI-based breast cancer image classification framework built with deep learning, Swin Transformer feature extraction, and an interactive Streamlit application.

> Disclaimer: This project is created for educational, research, and demonstration purposes only. It is not a clinical diagnostic tool and must not be used as a replacement for professional medical advice, screening, or diagnosis.

## Table of Contents

- [Project Overview](#project-overview)
- [Application Pages](#application-pages)
- [Key Features](#key-features)
- [Supported Datasets](#supported-datasets)
- [Methodology](#methodology)
- [Model Workflow](#model-workflow)
- [Results](#results)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Hugging Face Model Access](#hugging-face-model-access)
- [Run the Application](#run-the-application)
- [Deployment Notes](#deployment-notes)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Author](#author)
- [License](#license)

## Project Overview

MultiFusion Breast Cancer Detection Framework is a deep learning project designed to classify breast cancer medical images across different imaging modalities. The project focuses on using AI-assisted image analysis to support research into early breast cancer detection.

The current application provides two prediction modules:

- BreakHis Prediction for histopathology images
- BUSI Prediction for breast ultrasound images

The app allows users to upload an image, run inference through a trained model, and view the predicted class with confidence scores and probability distribution.

## Application Pages

The Streamlit interface is organized into three main pages:

| Page | Purpose |
|---|---|
| Home | Provides an overview of the system, breast cancer awareness information, and dataset details |
| BreakHis Prediction | Accepts histopathology images and predicts Benign or Malignant |
| BUSI Prediction | Accepts ultrasound images and predicts Benign, Malignant, or Normal |

## Key Features

- Interactive Streamlit web interface
- Image upload support for PNG, JPG, and JPEG files
- Breast cancer classification for histopathology and ultrasound images
- Swin Transformer-based deep learning architecture
- Image preprocessing using CLAHE for BreakHis inference
- Confidence score and class-wise probability output
- Private model weight loading from Hugging Face
- Streamlit Secrets support for secure token management
- Lightweight public repository suitable for deployment

## Supported Datasets

### BreakHis

BreakHis is a breast cancer histopathological image dataset used for microscopic tissue image classification.

| Detail | Value |
|---|---|
| Modality | Histopathology |
| Classes | Benign, Malignant |
| Application Page | BreakHis Prediction |

### BUSI

BUSI is a breast ultrasound image dataset used for breast lesion classification.

| Detail | Value |
|---|---|
| Modality | Ultrasound |
| Classes | Benign, Malignant, Normal |
| Application Page | BUSI Prediction |

## Methodology

The framework uses deep learning for medical image classification. The application loads trained model weights and performs inference on uploaded images using a Swin Transformer backbone.

High-level methodology:

1. User uploads a breast cancer image.
2. The image is converted to RGB format.
3. The image is resized to 224 x 224 pixels.
4. Standard ImageNet normalization is applied.
5. The selected trained model performs inference.
6. Softmax probabilities are calculated.
7. The predicted class, confidence score, and probability distribution are displayed.

For BreakHis images, CLAHE preprocessing is applied before prediction to enhance image contrast.

## Model Workflow

```text
Uploaded Medical Image
        |
        v
Image Preprocessing
        |
        v
Swin Transformer Encoder
        |
        v
Classification Head
        |
        v
Softmax Probability Scores
        |
        v
Predicted Class + Confidence
```

## Results

The framework has been evaluated on breast cancer image classification datasets with the following reported performance:

| Dataset | Imaging Modality | Reported Accuracy |
|---|---|---:|
| BreakHis | Histopathology | 98.04% |
| BUSI | Ultrasound | 96.87% |

These results are based on the project evaluation setup and should be interpreted as research results, not clinical validation.

## Technology Stack

- Python
- Streamlit
- PyTorch
- Torchvision
- timm
- Hugging Face Hub
- OpenCV
- Pillow
- NumPy
- streamlit-option-menu

## Project Structure

```text
MultiFusion-Breast-Cancer-Detection-Framework/
|
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Ignored local/private files
```

Private or local-only resources are excluded from the public repository:

```text
Dataset/
Models/
Notebooks/
.streamlit/secrets.toml
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/harshit04isme/MultiFusion-Breast-Cancer-Detection-Framework.git
cd MultiFusion-Breast-Cancer-Detection-Framework
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For macOS or Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Hugging Face Model Access

The trained model weights are not stored directly in this repository. The application downloads them from a Hugging Face repository at runtime.

The app expects a Hugging Face token named `HF_TOKEN` in Streamlit Secrets.

Create this file locally:

```text
.streamlit/secrets.toml
```

Add your token:

```toml
HF_TOKEN = "your_hugging_face_read_token"
```

Never commit `.streamlit/secrets.toml` or expose your Hugging Face token publicly.

## Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deployment Notes

This project is suitable for deployment on Streamlit Community Cloud or similar Python app hosting platforms.

For deployment:

- Add all dependencies to `requirements.txt`.
- Store `HF_TOKEN` securely in the hosting platform's secrets manager.
- Keep model weights, datasets, notebooks, and local experiments outside the public repository.
- Ensure the Hugging Face repository has the required model files:
  - `breakhis_model.pth`
  - `busi_model.pth`

## Limitations

- This project is for research and educational use only.
- It is not approved or validated for clinical diagnosis.
- Model performance may vary on images from different devices, hospitals, populations, or acquisition settings.
- The current app supports only BreakHis and BUSI prediction workflows.
- External clinical validation is required before any real-world medical use.

## Future Scope

- Add explainable AI visualizations such as Grad-CAM.
- Add support for more imaging modalities.
- Improve cross-dataset generalization.
- Add model calibration and uncertainty estimation.
- Include validation reports and confusion matrices.
- Improve deployment monitoring and inference logging.
- Optimize inference speed for low-resource environments.

## Author

**Harshit Yadav**

- Email: [hy.harshityadav01@gmail.com](mailto:hy.harshityadav01@gmail.com)
- GitHub: [harshit04isme](https://github.com/harshit04isme)

## License

This project is intended for educational and research purposes. Dataset ownership and licensing remain with their respective providers. Users are responsible for following all dataset, model, and dependency license terms.
