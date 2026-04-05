# 📡 Telecom Churn AI

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Tests](https://img.shields.io/badge/Tests-Passing-green)
![Status](https://img.shields.io/badge/Status-POC_Completed-brightgreen)

## 📄 Project Context

This project addresses a critical need for a telecommunications company: **predicting customer churn**.
The goal is to develop a supervised Machine Learning pipeline capable of identifying high-risk customers, allowing the marketing team to focus their retention strategies effectively.

## 🎯 Accomplished Objectives
- **Exploration (EDA)**: Conducted thorough data distribution and correlation analysis via Jupyter Notebook.
- **Automated Pipeline**: Developed a modular Python script for data loading, cleaning, encoding, and training.
- **Code Quality**: Implemented unit and integration tests (`pytest`) to ensure pipeline robustness.
- **Modeling**: Compared performance across multiple models: *Logistic Regression*, *Random Forest*, and *SVC*.

---

## 📂 Project Structure

```bash
├── data/
│   └── df_churn.csv          # Raw data file (Source of truth)
├── notebooks/
│   └── eda.ipynb             # Jupyter Notebook: Data Exploration (EDA)
├── src/
│   └── pipeline.py           # Main Pipeline: Load, Preprocess, Train, Evaluate
├── tests/
│   ├── test.py               # Comprehensive Test Suite (Functional)
│   └── test_pipeline.py      # Initial unit tests
├── models/                   # Generated: Saved model files (*.joblib)
├── .venv/                    # Python Virtual Environment
├── requirements.txt          # Project dependencies (pandas, sklearn, pytest, etc.)
└── README.md                 # Project documentation (Current)
```

-----

## 🚀 Getting Started

### 1\. Environment Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2\. Running the Pipeline

To execute the full data processing and training flow:

```bash
python src/pipeline.py
```

### 3\. Running Tests

To verify data quality and pipeline logic:

```bash
pytest tests/test.py
```

*Expected Result: `4 passed`*

-----

## 📊 Results and Performance

The models were trained and evaluated on a test set of **1,409 customers** (20% of the dataset).

| Metric | Logistic Regression (Selected) | Random Forest | SVC |
|----------|--------------------------------|---------------|-----|
| **Accuracy** | **~80%** | ~79% | ~80% |
| **Recall** | **~54%** | ~50% | ~53% |
| **F1-Score** | **0.59** | 0.56 | 0.58 |
| **ROC AUC** | **0.84** | 0.82 | 0.83 |

### 🧠 Technical Analysis

The **Logistic Regression** model was selected for production because:

1.  **Superior Detection (Recall)**: It identifies more actual churners (54.3%) compared to Random Forest.
2.  **Robustness (ROC AUC)**: With a score of 0.84, it shows high discrimination capability between loyal and churning customers.
3.  **Efficiency**: It is lightweight, fast to train, and highly interpretable for business stakeholders.

-----

## ⚙️ Pipeline Details (Feature Engineering)

The `pipeline.py` script automatically performs the following transformations:

1.  **Cleaning**: Coerces `TotalCharges` to numeric and handles missing values via mean imputation.
2.  **Encoding**: Transforms categorical features (e.g., 'Yes'/'No' -> 1/0) using `LabelEncoder`.
3.  **Scaling**: Normalizes numerical features using `MinMaxScaler` to optimize algorithm convergence.
4.  **Stratified Split**: Ensures the training and testing sets maintain the original churn distribution.
