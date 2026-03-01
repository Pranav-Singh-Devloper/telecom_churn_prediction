---
title: Telecom Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
---

# 📉 Telecom Customer Churn Predictor (Milestone 1)

This repository contains Milestone 1 for the "Customer Churn Prediction & Agentic Retention Strategy" project. It implements a classical machine learning pipeline to predict customer churn risk based on historical behavioral and transactional data.

## 🚀 Project Features
* **Custom Scikit-Learn Pipeline:** Integrates data cleaning, scaling, and categorical encoding directly into the modeling process.
* **Modular Architecture:** Separated source code (`src/`), application logic (`app.py`), and data layers for maintainability.
* **Interactive UI:** Built with Gradio to allow intuitive risk predictions.

## 🛠️ Setup Instructions

**1. Clone the repository**
```bash
git clone [https://github.com/Pranav-Singh-Devloper/telecom_churn_prediction.git](https://github.com/Pranav-Singh-Devloper/telecom_churn_prediction.git)
cd telecom_churn_prediction
```

**2. Install Dependencies**
It is highly recommended to use a virtual environment (like `conda` or `venv`) to keep your workspace clean. Once your environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

**3. Data Preparation**
This project relies on real-world behavioral data from the [Telecom Churn Dataset on Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) to identify customers at risk.
* Download the primary training dataset (typically named `churn-bigml-80.csv`).
* Rename the downloaded file to exactly `telecom_churn.csv`.
* Move this file into the `data/raw/` directory of this project.

**4. Train the Model**
Execute the training pipeline.This script will automatically load the raw data, apply our custom Scikit-Learn preprocessing steps, train the Logistic Regression model, and evaluate its performance.
```bash
python src/train_model.py
```
* **Success Criteria:** Upon successful execution, a serialized model artifact named `churn_model.pkl` will be generated inside the `models/` directory.

**5. Launch the Web Interface**
Start the local Gradio server to interact with the prediction model via a web UI.

```bash
python app.py
```
* **Access:** Open the provided local network URL (typically http://127.0.0.1:78600) in your web browser to test the predictor.
* **Input:** Provide customer bhavioral metrics such as day minutes and service calls.
* **Output:** The system returns a churn probability and classification (High/Low Risk).
---

## 🌐 Live Demo & Deployment
This application is publicly deployed to fulfill the mandatory project hosting requirements.
**Hosted Link:**[Link](https://huggingface.co/spaces/pranav-singh-developer-1/telecom-churn-predictor)

**📊 Mid-Sem Evaluation Components**
This milestone addresses the following rubric criteria:
* **Technical Implementation:** Logistic Regression and Scikit-Learn pipelines.
* **Code Quality:** Modular structure and frequent Git commits.
* **Documentation:** Comprehensive setup and usage instructions.
* **UI/UX:** Intuitive Streamlit/Gradio interface for real-time inference.




