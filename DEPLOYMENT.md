# Deployment Guide

This guide explains how to deploy the MultiFusion Breast Cancer Detection Framework using Streamlit Community Cloud or another Python-compatible hosting platform.

## Prerequisites

- A GitHub repository containing this project
- A Hugging Face account with access to the model repository
- A Hugging Face read token
- Python dependencies listed in `requirements.txt`

## Required Secret

The app expects this secret:

```toml
HF_TOKEN = "your_hugging_face_read_token"
```

Do not commit real tokens to GitHub.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

5. Add `HF_TOKEN` in the app secrets settings.
6. Deploy the app.

## Model Files

The Hugging Face repository should contain:

```text
breakhis_model.pth
busi_model.pth
```

The app downloads these files during runtime.

## Deployment Checklist

- `requirements.txt` is up to date.
- Real secrets are stored only in the deployment secrets manager.
- Dataset files and trained model files are not committed.
- The Hugging Face token has read access to the model repository.
- The app starts successfully with `streamlit run app.py`.

