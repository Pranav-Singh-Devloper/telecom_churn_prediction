---
title: Telecom Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.12
app_file: app.py
pinned: false
---

# 📉 AI Customer Churn Intelligence (Milestone 1)

This repository contains the Mid-Semester submission for the **Customer Churn Prediction & Agentic Retention Strategy** project.

Phase 1 focuses entirely on building a robust, production-ready predictive analytics system using Traditional Machine Learning. It analyzes historical behavioral data to identify customers at risk of canceling their service.

---

## 🚀 State-of-the-Art (SOTA) Features

To move beyond basic templates, this pipeline implements several advanced ML techniques:

### 🔍 Explainable AI (XAI)
Mathematically extracts feature weights to explain *why* a customer is churning (e.g., high customer service calls), rather than just outputting a black-box probability.

### 🎯 Cost-Sensitive Threshold Tuning
Optimizes the decision boundary from the default 50% to a custom 30% threshold, prioritizing detection of high-risk customers over raw accuracy.

### ⚖️ SMOTE Integration
Utilizes the `imbalanced-learn` library to generate synthetic data for the minority class (churners), ensuring the Logistic Regression model learns balanced behavioral patterns.

### 🧩 Custom Scikit-Learn Pipelines
Encapsulates data cleaning, dynamic `StandardScaler` scaling, and `OneHotEncoder` categorical handling directly into the model artifact.

---

## 📁 Repository Architecture

The codebase follows industry-standard modular design for machine learning projects:

```text
churn_project/
├── data/                  # Ignored in version control
│   └── raw/               # Location for telecom_churn.csv
├── models/                
│   └── churn_model.pkl    # Serialized ML pipeline artifact
├── src/                   # Core engine logic
│   ├── __init__.py
│   ├── data_loader.py     # Data validation script
│   ├── preprocessing.py   # Custom Scikit-Learn Transformer
│   └── train_model.py     # Pipeline construction and training
├── app.py                 # Gradio + Plotly interactive UI
├── requirements.txt       # Pinned dependencies for reproducible builds
└── README.md              # Project documentation
```

---

## 🛠️ Setup & Execution Instructions

Follow the steps below to reproduce the full training and deployment workflow.

---

### 1️⃣ Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Pranav-Singh-Devloper/telecom_churn_prediction.git
cd telecom_churn_prediction
pip install -r requirements.txt
```

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

---

### 2️⃣ Data Preparation

1. Download the Telecom Churn [dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets?resource=download&select=churn-bigml-80.csv).
2. Rename the dataset file to:

```
telecom_churn.csv
```

3. Place it inside:

```
data/raw/
```

Expected structure:

```
data/
└── raw/
    └── telecom_churn.csv
```

---

### 3️⃣ Train the Model

Run the training pipeline:

```bash
python src/train_model.py
```

This will:

- Apply preprocessing and feature engineering  
- Perform SMOTE-based class balancing  
- Train the Logistic Regression model  
- Optimize the classification threshold  
- Generate explainability metrics  
- Serialize the trained pipeline  

After successful execution, the following file will be created:

```
models/churn_model.pkl
```

---

### 4️⃣ Launch the Interactive Dashboard

Start the Gradio application:

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:7860
```

The dashboard allows you to:

- Input customer attributes  
- Generate churn probability predictions  
- View explainability insights  
- Interactively analyze risk factors  

---

### 🔁 Complete Workflow Summary

1. Install dependencies  
2. Place dataset in `data/raw/`  
3. Train model (`train_model.py`)  
4. Launch UI (`app.py`)  

Your churn intelligence system is now fully operational.