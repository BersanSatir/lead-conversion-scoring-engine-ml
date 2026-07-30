# 🎯 Lead Conversion Prediction & Scoring Engine

An end-to-end Machine Learning pipeline built to predict lead conversion probabilities and score incoming leads to optimize sales team efficiency. This project compares **Logistic Regression**, **Random Forest**, and **XGBoost** models, optimized via **GridSearchCV** with cross-validation.

---

## 📌 Business Problem

In B2B and high-ticket B2C sales funnels, sales representatives waste significant time contacting unqualified or cold leads. The goal of this project is to:
1. Predict whether a lead will convert (`1`) or not (`0`).
2. Identify high-priority **"Hot Leads"** using model-generated probability scores (0–100 scale).
3. Maximize **Precision** and **ROC-AUC** to minimize false alarms and focus team efforts on viable leads.

---

## 📊 Dataset Overview

This project utilizes the **Lead Scoring Dataset** (often referred to as `Leads.csv`). 

* **File Name:** `Leads.csv`
* **Description:** Contains interaction metrics, website behavior logs, and lead demographic attributes used to predict lead conversion probabilities.
* **Key Features:** `Lead Origin`, `Tags`, `Total Time Spent on Website`, `Page Views Per Visit`, `Last Activity`.
* **Note:** The raw data file is excluded from this repository via `.gitignore` to maintain a lightweight environment. You can place your own `Leads.csv` in the root directory to run the pipeline.

---

## 🛠️ Data Pipeline & Feature Engineering

To handle noise, high cardinality, and missing values, a structured preprocessing pipeline was applied:

* **High Cardinality Reduction:**
  * Categorical features such as `tags`, `lead_source`, `specialization`, and `last_activity` were grouped into domain-specific clusters (e.g., `lost_or_ineligible`, `high_engagement`, `unreachable`).
* **Data Cleaning & Imputation:**
  * Cleaned inconsistent whitespace and redundant text entries.
  * Dropped low-variance and non-informative features (e.g., `magazine`, `newspaper`).
  * Imputed sparse categorical and numerical missing values appropriately.
* **Encoding & Scaling:**
  * Binary features were mapped to `0`/`1`.
  * One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) was applied to nominal categorical variables.
  * Numerical variables (`totalvisits`, `total_time_spent_on_website`, `page_views_per_visit`) were standardized using `StandardScaler` fitted **only on the training set** to prevent data leakage.

---

## 📊 Model Comparison & Hyperparameter Tuning

All models were evaluated on an out-of-sample test set (20% split) using **ROC-AUC**, **Precision**, **Recall**, and **F1-Score**. Hyperparameter optimization was conducted via 3-Fold `GridSearchCV`.

### 🏆 Final Benchmark Results

| Model | Test ROC-AUC | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9382 | 88% | 88% | 80% | 0.84 |
| **Random Forest** | 0.9539 | 90% | 90% | 83% | 0.86 |
| **XGBoost (Champion 🏆)** | **0.9579** | **91%** | **90%** | **86%** | **0.88** |

### 🔍 Key Metrics of the Final XGBoost Model

```text
Confusion Matrix:
[[1057   68]   <- 68 False Positives
 [  96  594]]  <- 594 True Positives (86% Recall)