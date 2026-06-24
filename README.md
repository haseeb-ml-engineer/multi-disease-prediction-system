<div align="center">

# 🏥 Multi-Disease Prediction System

### AI-powered risk screening for Diabetes, Heart Disease, and COVID-19

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**A production-ready machine learning web application that screens patients for three diseases through a single, non-technical interface — no medical jargon, no raw numeric inputs.**

[🚀 Run Locally](#-installation) · [📊 Model Results](#-model-evaluation) · [📸 Screenshots](#-screenshots) · [👥 Team](#-contributors)

</div>

---

## 📋 Table of Contents

- [Overview](#-project-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [ML Workflow](#-machine-learning-workflow)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Model Information](#-model-information)
- [Model Evaluation](#-model-evaluation)
- [Screenshots](#-screenshots)
- [Challenges Faced](#-challenges-faced)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🔬 Project Overview

Chronic and infectious diseases like **diabetes**, **ischemic heart disease**, and **COVID-19** are among the leading causes of preventable deaths worldwide. A significant proportion of adverse outcomes arise not from a lack of treatment options — but from **delayed diagnosis**.

This project addresses that gap by building a unified, AI-powered risk-screening application that:

- Takes patient data as **plain-language input** (sliders, dropdowns, Yes/No questions — no codes, no jargon)
- Runs each input through a **trained ML pipeline** to estimate disease risk
- Returns a **calibrated probability score** (e.g. 78.3% risk) with a visual indicator
- Displays **color-coded results** and **disease-specific medical precautions**

The system was built end-to-end — from raw dataset acquisition and preprocessing, through model training, debugging, and deployment — by a team of BS Information Technology students specialising in ML/AI.

> **This is not a tutorial copy-paste.** Every pipeline was built from scratch, every bug was debugged, and every model was benchmarked against multiple algorithms before selection.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🩸 **Diabetes Screening** | Predicts diabetes risk using 8 clinical features (Glucose, BMI, Insulin, etc.) |
| ❤️ **Cardiac Mortality Screening** | Predicts heart disease mortality risk using longitudinal measurements at age 50 and 62 |
| 🦠 **COVID-19 Screening** | Assesses COVID-19 risk using 20 symptom, condition, and exposure indicators |
| 📊 **Probability Score** | Returns a calibrated risk percentage (e.g. 73.4%) alongside the binary result |
| 🎨 **Color-coded Results** | Red panel for high risk, green for low risk — instantly readable |
| 💊 **Medical Precautions** | Disease-specific actionable guidance returned with every result |
| 🖥️ **Non-technical Interface** | Plain-language sliders, dropdowns, and Yes/No questions — no medical coding knowledge required |
| ⚡ **Real-time Predictions** | Instant inference via serialized Scikit-learn Pipeline objects loaded at runtime |
| 🔧 **Leakage-free Pipelines** | All preprocessing and scaling wrapped inside `sklearn.Pipeline` to prevent data leakage |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      USER                               │
│         (Patient / Non-technical individual)            │
└─────────────────────┬───────────────────────────────────┘
                      │  Interacts via
                      ▼
┌─────────────────────────────────────────────────────────┐
│               STREAMLIT INTERFACE                       │
│  • Disease selector (Diabetes / Heart / COVID-19)       │
│  • Plain-language input forms (sliders, dropdowns,      │
│    Yes/No radio buttons)                                │
│  • Result display (probability, color-coded panel,      │
│    precautions)                                         │
└─────────────────────┬───────────────────────────────────┘
                      │  Preprocessed input DataFrame
                      ▼
┌─────────────────────────────────────────────────────────┐
│            SCIKIT-LEARN PIPELINE (per disease)          │
│  ┌───────────────────┐   ┌───────────────────────────┐  │
│  │  StandardScaler   │ → │  Trained Classifier       │  │
│  │  (feature scaling)│   │  (RF or SVM)              │  │
│  └───────────────────┘   └───────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │  predict() + predict_proba()
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  PREDICTION ENGINE                      │
│  • Binary result (High Risk / Low Risk)                 │
│  • Risk probability (0.0 – 1.0)                         │
│  • Disease-specific precaution lookup                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Machine Learning Workflow

```
1. DATA COLLECTION
   └─ Three independent datasets sourced from UCI / Kaggle
      (Pima Diabetes · WCGS Longitudinal Heart Study · COVID-19 Symptoms)

2. EXPLORATORY DATA ANALYSIS
   └─ Class distribution · Missing value analysis · Statistical summaries

3. DATA CLEANING & PREPROCESSING
   └─ Outlier inspection · Implausible-zero handling (Pima)
   └─ Label encoding (categorical features)
   └─ Binary encoding of Yes/No symptom responses (COVID-19)

4. FEATURE ENGINEERING
   └─ All original features retained (domain-curated datasets)
   └─ Ordinal mappings for doctor-visit frequency, SES, smoking status

5. TRAIN / TEST SPLIT
   └─ 80/20 stratified split · Fixed random seed for reproducibility

6. MODEL TRAINING (inside sklearn.Pipeline)
   └─ StandardScaler → Classifier
   └─ Algorithms evaluated: Random Forest · SVM · Gaussian Naive Bayes
   └─ Heart Disease: RF baseline 71% → SVM rebuilt pipeline 80%

7. MODEL EVALUATION
   └─ Accuracy · Precision · Recall · F1-Score · ROC-AUC · Confusion Matrix
   └─ Recall treated as primary metric (false negatives costliest in healthcare)

8. SERIALIZATION
   └─ joblib.dump() → .pkl files (entire Pipeline, not just classifier)

9. DEPLOYMENT
   └─ Streamlit web app loads .pkl at runtime → live inference
```

---

## 📂 Datasets

| Disease | Dataset | Source | Instances | Features | Target Variable |
|---|---|---|---|---|---|
| 🩸 Diabetes | Pima Indians Diabetes Dataset | [UCI / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | 8 | Diabetic (1) / Not (0) |
| ❤️ Heart Disease | Western Collaborative Group Study (WCGS) | UCI / Course dataset | 200 | 15 | Death (1) / Survived (0) |
| 🦠 COVID-19 | COVID-19 Symptoms & Precautions | [Kaggle](https://www.kaggle.com/datasets/iamhungundji/covid19-symptoms-checker) | 5,434 | 20 | COVID-19 Yes / No |

> **Note:** Raw dataset files are not included in this repository. Download from the linked sources and place in a `/datasets` folder before retraining. Pre-trained `.pkl` models are included in `/models` and the app runs without retraining.

---

## 📁 Project Structure

```
multi-disease-prediction-system/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/
│   ├── diabetes_model.pkl          # Trained diabetes pipeline (RF)
│   ├── heart_model.pkl             # Trained heart disease pipeline (SVM)
│   └── covid_model.pkl             # Trained COVID-19 pipeline (RF)
│
├── notebooks/
│   ├── diabetes_training.ipynb     # Diabetes model training + evaluation
│   ├── heart_training.ipynb        # Heart disease model training + evaluation
│   └── covid_training.ipynb        # COVID-19 model training + evaluation
│
├── screenshots/
│   ├── home.png                    # App home / disease selector
│   ├── diabetes_result.png         # Diabetes high-risk result
│   ├── heart_form.png              # Heart disease input form
│   └── covid_result.png            # COVID-19 result panel
│
└── datasets/                       # (empty — download datasets separately)
    └── .gitkeep
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-disease-prediction-system.git
cd multi-disease-prediction-system
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📖 Usage Guide

1. **Open the application** — run `streamlit run app.py` and the browser opens automatically
2. **Select a disease module** — choose from Diabetes, Heart Disease, or COVID-19 using the top selector
3. **Enter patient information** — fill in the guided form (sliders for measurements, dropdowns for categorical data, Yes/No radio buttons for symptoms)
4. **Click the screening button** — press "Run Diabetes Screening", "Run Cardiac Risk Screening", or "Run COVID-19 Screening"
5. **Read the result** — a color-coded panel shows the risk level, probability score, and personalised medical precautions

> ⚕️ All results include a medical disclaimer. This application is for educational and research purposes only.

---

## 🤖 Model Information

### 🩸 Diabetes Model — Random Forest Classifier

Input features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

Random Forest was selected after evaluating Logistic Regression and other baseline models. Its ability to handle non-linear feature interactions without requiring explicit feature scaling made it the best fit for the Pima dataset's mixed numeric features.

### ❤️ Heart Disease Model — Support Vector Machine (RBF kernel)

Input features: AGE_50, MD_50, SBP_50, DBP_50, HT_50, WT_50, CHOL_50, SES, CL_STATUS, MD_62, SBP_62, DBP_62, CHOL_62, WT_62, IHD_DX

The heart disease pipeline was **rebuilt from scratch** mid-project after a Random Forest baseline achieved only 71% accuracy. Switching to a properly scaled SVM pipeline resolved three production-level engineering issues:

- **Feature scale bias** — cholesterol values (~200) were dominating binary features (0/1); fixed with `StandardScaler`
- **Data leakage** — scaling outside the pipeline was leaking test data into training; fixed by wrapping in `sklearn.Pipeline`
- **No probability output** — SVMs don't compute probabilities by default; fixed by enabling `probability=True`

Result: accuracy improved from **71% → 80%**.

> XGBoost was evaluated during experimentation but Random Forest and SVM achieved better performance on the project datasets and were therefore selected as the final models.

### 🦠 COVID-19 Model — Random Forest Classifier

Input features: 20 binary-encoded symptom, pre-existing condition, and exposure indicators

Random Forest was chosen for its strong performance on binary-encoded categorical feature spaces and its robustness to the class imbalance present in the COVID-19 dataset (81% positive class).

---

## 📊 Model Evaluation

### 🩸 Diabetes — Random Forest

| Metric | Class 0 (No Diabetes) | Class 1 (Diabetes) | Overall |
|---|---|---|---|
| Accuracy | — | — | **75%** |
| Precision | 0.85 | 0.61 | — |
| Recall | 0.74 | 0.76 | — |
| F1-Score | 0.79 | 0.68 | — |

> Recall for the diabetic class (0.76) is treated as the primary success metric — 76% of actual diabetic patients were correctly identified.

### ❤️ Heart Disease — SVM (after pipeline rebuild)

| Metric | Score |
|---|---|
| Accuracy (baseline Random Forest) | 71% |
| Accuracy (final SVM pipeline) | **80%** |
| Precision | `[ add from Colab ]` |
| Recall | `[ add from Colab ]` |
| F1-Score | `[ add from Colab ]` |
| ROC-AUC | `[ add from Colab ]` |

### 🦠 COVID-19 — Random Forest

| Metric | Score |
|---|---|
| Accuracy | `[ add from Colab ]` |
| Precision | `[ add from Colab ]` |
| Recall | `[ add from Colab ]` |
| F1-Score | `[ add from Colab ]` |

> Replace `[ add from Colab ]` placeholders with your exact output from the training notebooks before final submission.

---

## 📸 Screenshots

### Home — Disease Selector
![Home](screenshots/home.png)

### Diabetes Risk Assessment
![Diabetes](screenshots/diabetes_result.png)

### Heart Disease Risk Assessment
![Heart](screenshots/heart_form.png)

### COVID-19 Risk Assessment
![COVID](screenshots/covid_result.png)

> Add your screenshots to the `/screenshots` folder and the images will render automatically here.

---

## ⚠️ Challenges Faced

| Challenge | How It Was Resolved |
|---|---|
| **Feature scale bias in SVM** | Implemented `StandardScaler` inside a unified `sklearn.Pipeline` |
| **Data leakage** | Moved all preprocessing inside the Pipeline so the scaler is fit only on training data |
| **No probability output from SVM** | Enabled `probability=True` and re-tuned hyperparameters to unlock `predict_proba()` |
| **Column name mismatches** | Built explicit `DISPLAY_TO_COL` mapping dictionaries to bridge UI keys and model column names |
| **Dataset type mismatch (COVID-19)** | Discovered model expected integer-encoded input despite CSV containing `"Yes"`/`"No"` strings; handled via consistent encoding pipeline |
| **Multiple independent datasets** | Designed three separate training notebooks and Pipeline objects, each tailored to its dataset's characteristics |

---

## 🔮 Future Improvements

- [ ] **Additional diseases** — Kidney Disease, Parkinson's Disease, Liver Disease
- [ ] **Explainable AI** — SHAP values to show which features drove each individual prediction
- [ ] **Cloud deployment** — Deploy to Streamlit Cloud, AWS, or Hugging Face Spaces for public access
- [ ] **Mobile application** — Flutter-based mobile wrapper around the prediction API
- [ ] **User authentication** — Patient login, history tracking, and longitudinal monitoring
- [ ] **Doctor dashboard** — Clinician-facing view with batch patient screening
- [ ] **Medical report generation** — PDF export of risk assessment results
- [ ] **Larger datasets** — Integrate multi-centre datasets to improve generalizability
- [ ] **Deep learning** — Evaluate feed-forward neural networks once larger datasets are available

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <b>Haseeb Tariq</b><br/>
      <sub>ML Engineering · Frontend (Streamlit) · Pipeline Debugging</sub><br/>
      <a href="https://linkedin.com/in/YOUR_LINKEDIN">LinkedIn</a> ·
      <a href="https://github.com/YOUR_GITHUB">GitHub</a>
    </td>
    <td align="center">
      <b>Sharjeel Ikhlaq</b><br/>
      <sub>Model Training · Data Preprocessing</sub><br/>
      <a href="#">LinkedIn</a> ·
      <a href="#">GitHub</a>
    </td>
    <td align="center">
      <b>Moaz Naeem</b><br/>
      <sub>Model Training · Data Preprocessing</sub><br/>
      <a href="#">LinkedIn</a> ·
      <a href="#">GitHub</a>
    </td>
  </tr>
</table>

---

## ⚕️ Disclaimer

> **This application is developed for educational and research purposes only.**
> It does not constitute medical advice, diagnosis, or treatment.
> The predictions generated by this system are based on statistical patterns in historical datasets and should **never** be used as a substitute for professional medical consultation, diagnosis, or treatment.
> Always consult a qualified and licensed healthcare professional for any medical concerns.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Haseeb Tariq, Sharjeel Ikhlaq, Moaz Naeem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**Built with 🤖 ML · 🐍 Python · ❤️ by Haseeb Tariq, Sharjeel Ikhlaq & Moaz Naeem**

*BS Information Technology · Specialisation: Machine Learning / AI*

</div>
