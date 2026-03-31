# 🧠 Late Refill Risk Prediction

### Pharmacy2U Hackathon — Healthcare AI

Predicting **late prescription refills** using CMS PDE + Beneficiary data to enable **proactive pharmacist intervention** and improve medication adherence.

---

## 🚀 Overview

Medication non-adherence is a major driver of poor health outcomes and avoidable healthcare costs.

A strong early signal of non-adherence is **late prescription refills**.

We define a refill as **late** if:

```
next_SRVC_DT > SRVC_DT + DAYS_SUPLY_NUM + 7 days
```

This project builds an **end-to-end machine learning pipeline** to predict late refill risk at the **patient–drug level**.

---

## 📂 Data Sources

### 1. Prescription Drug Events (PDE)

* ~5.5 million records
* Includes:

  * Fill dates
  * Drug identifiers
  * Quantity dispensed
  * Days supply
  * Cost

### 2. Beneficiary Summary File

* ~112k patients
* Includes:

  * Demographics
  * Chronic condition indicators

---

## ⚙️ Pipeline Overview

The notebook implements:

1. Data loading & cleaning
2. Patient–drug sequence construction
3. Late refill label generation
4. Feature engineering:

   * Refill behaviour
   * Cost patterns
   * Demographics
   * Clinical indicators
5. Time-based train/test split (no leakage)
6. Model training:

   * Logistic Regression (baseline)
   * XGBoost (final model)
7. Evaluation using PR-AUC
8. Explainability (SHAP)
9. Patient-level timeline visualization
10. Risk ranking for intervention

---

## 📈 Results

| Model               | PR-AUC |
| ------------------- | ------ |
| Logistic Regression | 0.749  |
| XGBoost             | 0.753  |

### Key Observations

* XGBoost captures non-linear refill behaviour
* High recall supports healthcare risk detection
* Accuracy is not reliable due to class imbalance

---

## 🔍 Key Drivers (Model Insights)

Top predictors of late refills:

* Days supply
* Patient cost burden
* Chronic conditions
* Age
* Refill consistency
* Drug usage patterns

---

## 🧬 Patient-Level Explainability

* **Global SHAP** → overall feature importance
* **Local SHAP** → explains individual predictions
* **Timeline visualization** → risk evolution over time

---

## 📊 Business Impact

* Identify high-risk patients early
* Enable proactive pharmacist outreach
* Improve medication adherence
* Reduce avoidable complications
* Support healthcare cost reduction

---

## 🧪 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Download CMS data

Place files inside:

```
data/raw/
```

Required files:

* PDE dataset
* Beneficiary dataset

### 3. Run notebook

```
notebooks/pharmacy2u_late_refill.ipynb
```

Run all cells top-to-bottom.

---

## ⚠️ Limitations

* Last refill has no label (censoring)
* Beneficiary data limited to 2010
* Fixed 7-day grace window
* No drug-class mapping (ATC)
* Synthetic dataset

---

## 🏁 Conclusion

This project demonstrates how **data-driven risk prediction** can transform medication adherence from reactive to **proactive healthcare intervention**.

---

## 📄 License

MIT License
