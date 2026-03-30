# 🧠 Late Refill Risk Prediction — Pharmacy2U Hackathon

Predicting late prescription refills using CMS PDE + Beneficiary data.

This project builds an end-to-end machine learning pipeline to identify high-risk patient–drug pairs **before** they miss their next refill — enabling proactive pharmacist intervention and improved medication adherence.

---

## 🚀 Project Overview

Medication non-adherence is a major driver of poor health outcomes and avoidable healthcare costs.
A strong early signal of non-adherence is **late prescription refills**.

This project:

* Cleans and merges **5.5M+ prescription events** with patient demographics
* Engineers **behavioural, cost, demographic, and clinical features**
* Builds baseline and advanced ML models (**Logistic Regression, XGBoost**)
* Evaluates performance using **PR-AUC (primary metric)**
* Provides **global & local explainability (SHAP)**
* Visualises **patient refill timelines**
* Produces **actionable risk scores** for intervention

---

## 📂 Data Sources
### 1. Prescription Drug Events (PDE)

* ~5.5 million rows
* Contains:

  * Fill dates
  * Drug identifiers
  * Quantity dispensed
  * Days supply
  * Costs

### 2. Beneficiary Summary File

* ~112k patients
* Contains:

  * Demographics
  * Chronic condition indicators

---

## 🧹 Data Processing

Key steps:

* Convert IDs and date columns to correct formats
* Merge PDE + Beneficiary data on `DESYNPUF_ID`
* Sort data by **patient → drug → time**

### Derived Fields:

* Next fill date
* Expected run-out date
* Late refill label (7-day grace window)

### Class Imbalance:

* **91% Late refills**
* **9% On-time refills**

---

## 🧠 Feature Engineering

### 🔹 Day 1 Features

**Demographics**

* Age
* Sex
* Race

**Chronic Conditions**

* All `SP_*` flags

**Refill Behaviour**

* `n_prior_fills`
* `gap_prev`
* `avg_gap_hist`

**Cost & Quantity**

* `QTY_DSPNSD_NUM`
* `DAYS_SUPLY_NUM`
* `PTNT_PAY_AMT`
* `TOT_RX_CST_AMT`

**Polypharmacy Proxy**

* Number of distinct drugs per patient

---

### 🔹 Day 2 Enhancements

* Rolling gap variability (`gap_var_3`)
* Adherence score (`adherence_score`)
* Early refill indicator (`early_refill`)
* Drug class (`drug_class`)
* Drug class frequency (`drug_class_freq`)
* Polypharmacy (last 90 days) (`poly_90d`)
* Patient-level consistency:

  * `patient_gap_mean`
  * `patient_gap_std`

---

## 🤖 Models

### Baseline Models

| Model               | PR-AUC | Notes                        |
| ------------------- | ------ | ---------------------------- |
| Logistic Regression | 0.749  | Balanced class weights       |
| XGBoost             | 0.753  | Captures non-linear patterns |

---

### 🚀 Tuned XGBoost

* 400 trees
* Max depth: 5
* Learning rate: 0.05
* Subsample: 0.9
* Regularization:

  * `gamma`
  * `min_child_weight`

**Calibration:** Isotonic Regression

---

## 📈 Evaluation Metrics

### Primary Metric

* **PR-AUC** (robust under class imbalance)

### Secondary Metrics

* ROC-AUC
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🔍 Key Findings

* XGBoost outperforms Logistic Regression
* High recall is critical for healthcare risk detection
* Accuracy is misleading due to class imbalance

---

## 🔍 Explainability (SHAP)

### 🌍 Global Insights

Top drivers of late refills:

* Days supply
* Chronic conditions
* Quantity dispensed
* Age
* Patient cost
* Drug class
* Refill stability

---

### 🧪 Local Explainability (Waterfall)

Explains individual predictions using:

* Cost burden
* Short supply durations
* High medication complexity
* Chronic conditions

---

## 🧬 Patient Timeline Visualization

Each patient view shows:

* Fill dates
* Expected run-out windows
* Late refill events
* Model-predicted risk over time

👉 Provides a **clinically interpretable narrative**

---

## 📊 Risk Ranking

Generates a ranked list of:

> 🔝 Top 10 highest-risk patient–drug pairs

Used for:

* Pharmacist outreach
* Intervention prioritisation

---

## 🏥 Business Impact

* Identify high-risk patients early
* Enable proactive intervention
* Improve medication adherence
* Reduce avoidable complications
* Support NHS cost savings

---

## ⚠️ Limitations

* Last refill is unlabeled (censoring issue)
* Beneficiary data only available for 2010
* No ATC drug-class mapping
* Fixed 7-day grace window
* No personalised adherence baselines

---

## 🏁 Conclusion

This project demonstrates how **data-driven risk prediction** can transform medication adherence from a reactive process into a **proactive healthcare strategy**.

---

⭐ If you found this useful, feel free to star the repo!
