# Late Refill Risk Prediction (Pharmacy2U Hackathon)

Predicting which patient–drug fills are at high risk of **late refill**, using CMS PDE + Beneficiary data and a time‑based validation pipeline.

---

## 1. Project overview

- **Goal:** Flag high‑risk patient–drug pairs *before* a late refill occurs, enabling proactive outreach.
- **Task:** Binary classification — `late_refill` (1) vs `on_time` (0).
- **Label definition:**  
  For each patient–drug sequence, a fill is labeled **late** if the **next service date** is later than:

  

\[
  \text{SRVC\_DT} + \text{DAYS\_SUPLY\_NUM} + 7\ \text{(grace days)}
  \]



- **Primary metric:** PR‑AUC (average precision) on a **time‑based hold‑out set**.
- **Models:**
  - Logistic Regression (baseline)
  - XGBoost (baseline)
  - Tuned XGBoost + calibration + SHAP explainability

The main logic is implemented in a single notebook:

- `notebooks/pharmacy2u_late_refill.ipynb`

---

## 2. Repository structure

```text
.
├── notebooks/
│   └── pharmacy2u_late_refill.ipynb
├── src/                # (optional) helpers if you refactor later
│   ├── features.py
│   ├── model.py
│   ├── evaluation.py
│   └── utils.py
├── data/               # (local only, not committed)
│   ├── raw/
│   └── processed/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
└── .gitignore
