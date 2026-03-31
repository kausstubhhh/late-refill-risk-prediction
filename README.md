# Late Refill Risk Prediction  
Pharmacy2U x Data & AI Hackathon — Track 1 (Healthcare & Digital Pharmacy)  
University of Leeds — 30–31 March 2026

Predicting **late prescription refills** from synthetic claims-style data to enable proactive pharmacist intervention and improve medication adherence.

---

##  Hackathon Context

**Event:** DATA-AI-Hackathon-Track-1  
**Track:** Track 1 — Healthcare & Digital Pharmacy  
**Challenge Host:** Pharmacy2U  
**Theme:** Prescription refill risk & responsible “next-best” recommendations, built only from prescription order events.  

This repo implements **Challenge A — Late refill risk**:

> Goal: predict which patient–drug pairs are likely to refill late next time, and produce a usable risk score.

This is a **modelling + product-thinking exercise** on **synthetic data** — not clinical advice.

---

##  Dataset

We use the official **CMS DE-SynPUF — Prescription Drug Events (PDE), Sample 1 (2008–2010)** and the **2010 Beneficiary Summary File**.

Fully synthetic, Part D-style prescription event data designed for training and software development.

### Main resources

- CMS Sample 1 page (file list):  
  https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1  

- Direct PDE ZIP (primary file):  
  https://downloads.cms.gov/files/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.zip  

- DE 1.0 Codebook (columns):  
  https://www.cms.gov/files/document/de-10-codebook.pdf-0  

- Optional: 2010 Beneficiary Summary:  
  https://www.cms.gov/sites/default/files/2020-09/DE1_0_2010_Beneficiary_Summary_File_Sample_1.zip  

- Backup (if blocked): AWS OMOP mirror:  
  https://registry.opendata.aws/cmsdesynpuf-omop/

> **Download tip:** if clicking does nothing, paste the URL into your browser address bar.

### Key PDE columns used

- **DESYNPUF_ID** — pseudonymised patient/beneficiary ID  
- **SRVC_DT** — prescription fill/service date  
- **PROD_SRVC_ID** — NDC-11 drug product code (dispensed item)  
- **DAYS_SUPLY_NUM** — expected duration of the fill (days of supply)  
- **QTY_DSPNSD_NUM** — quantity dispensed  
- **PTNT_PAY_AMT** — patient pay amount (cost signal)  
- **TOT_RX_CST_AMT** — total drug cost (cost signal)

### Quick glossary

- **NDC-11:** 11-digit National Drug Code (product identifier)  
- **Days of supply:** how long the dispensed quantity is expected to last  
- **Expected run-out date:** `SRVC_DT + DAYS_SUPLY_NUM`  
- **Late refill:** next fill happens after run-out (with a grace window, e.g. +7 days)

---

##  Problem Framing — Challenge A: Late Refill Risk

We model **late refill risk at the patient–drug level**.

### Label definition

For each patient–drug pair (`DESYNPUF_ID`, `PROD_SRVC_ID`):

- Compute **expected run-out**:  
  `expected_runout = SRVC_DT + DAYS_SUPLY_NUM`
- Look at the **next fill** for the same patient–drug:
  - If  
    `next_SRVC_DT > expected_runout + 7 days` → `late_refill = 1`  
  - Else → `late_refill = 0`
- Last fill in a sequence has no next fill → excluded from training (censoring).

### Features (high level)

- **Refill behaviour:** gaps between fills, average gap history, variability, early refills, adherence score  
- **Cost & quantity:** days supply, quantity dispensed, patient pay, total cost  
- **Demographics & clinical:** age, sex, race, chronic condition flags (SP_* columns)  
- **Polypharmacy & patterns:** number of distinct drugs, simple drug class, short-term polypharmacy proxies  

### Validation

- **Time-based split** to avoid leakage:  
  - Train: fills with `SRVC_DT ≤ 2009-12-31`  
  - Test: fills with `SRVC_DT > 2009-12-31`  

---

##  Pipeline Overview

Implemented in:  
`notebooks/pharmacy2u_late_refill.ipynb`

Steps:

1. **Data loading & cleaning**
   - Load PDE + Beneficiary CSVs
   - Type casting, date parsing, merge on `DESYNPUF_ID`

2. **Patient–drug sequence construction**
   - Sort by `DESYNPUF_ID`, `PROD_SRVC_ID`, `SRVC_DT`
   - Compute next fill date per patient–drug

3. **Label generation**
   - Compute `expected_runout`
   - Define `late_refill` using a **7-day grace window**

4. **Feature engineering**
   - **Demographics:** age at fill, sex, race  
   - **Chronic conditions:** SP_* flags  
   - **Refill behaviour:**  
     - `n_prior_fills`, `gap_prev`, `avg_gap_hist`, `gap_var_3`  
     - `adherence_score`, `early_refill`  
   - **Cost/quantity:** `QTY_DSPNSD_NUM`, `DAYS_SUPLY_NUM`, `PTNT_PAY_AMT`, `TOT_RX_CST_AMT`  
   - **Polypharmacy & patterns:** `n_distinct_drugs_patient`, `drug_class`, `drug_class_freq`, `poly_90d`  
   - **Patient-level stats:** `patient_gap_mean`, `patient_gap_std`

5. **Time-based train/test split**
   - Cutoff: `2009-12-31`

6. **Models**
   - **Logistic Regression (baseline)**  
   - **XGBoost (baseline)**  
   - **Tuned XGBoost + calibration (isotonic)**

7. **Evaluation**
   - Primary metric: **PR-AUC** (precision–recall AUC)  
   - Secondary: ROC-AUC, accuracy, precision, recall, F1  
   - Confusion matrix at a default threshold

8. **Explainability**
   - SHAP summary plot (global feature importance)  
   - SHAP waterfall / force plot (local explanations)

9. **Patient timeline visualization**
   - For a selected patient–drug, plot fills, run-out windows, and late events

10. **Risk ranking**
    - Rank test-set patient–drug pairs by predicted risk score

---

## 📈 Results (Day 2)

### Main metric — PR-AUC

| Model               | PR-AUC  |
|---------------------|---------|
| Logistic Regression | 0.749   |
| XGBoost             | 0.753   |
| Tuned XGBoost       | higher (with Day 2 features + calibration) |

### Other metrics (XGBoost baseline)

- ROC-AUC: ~0.60  
- Accuracy: ~0.66–0.69 (not very informative due to imbalance)  
- Precision: ~0.74–0.76  
- Recall: ~0.81–0.84  
- F1: ~0.78–0.80  

> **Note:** For risk detection, **PR-AUC + recall** are more meaningful than accuracy.

### Visuals produced

- Precision–Recall curve (LogReg vs XGBoost)  
- Calibration curve (XGBoost / tuned XGBoost)  
- SHAP summary plot (global drivers)  
- SHAP waterfall / force plot (local explanation)  
- Patient refill timeline (fills vs run-out vs late events)  
- Top 10 highest-risk patient–drug pairs

---

##  Key Drivers (Model Insights)

From feature importance + SHAP:

- **DAYS_SUPLY_NUM** — days of supply (longer fills change risk dynamics)  
- **Chronic conditions** — e.g. SP_OSTEOPRS, SP_COPD, SP_RA_OA, SP_ALZHDMTA, SP_STRKETIA  
- **Cost signals** — `PTNT_PAY_AMT`, `TOT_RX_CST_AMT`  
- **Age** — older patients show different adherence patterns  
- **Refill behaviour** — `gap_prev`, `avg_gap_hist`, `gap_var_3`, `adherence_score`  
- **Polypharmacy proxies** — `n_distinct_drugs_patient`, `drug_class_freq`

---

##  Patient-Level Explainability

- **Global SHAP:** which features generally push risk up or down  
- **Local SHAP:** why a specific patient–drug pair is flagged as high risk  
- **Timeline visualization:** how fills, run-out windows, and late events evolve over time

This supports **transparent, explainable risk scores** suitable for a pharmacist-facing tool.

---

##  Business Impact (Pharmacy2U framing)

If deployed (on real data, with proper validation):

- Identify **high-risk patients** before they miss refills  
- Enable **proactive pharmacist outreach** (reminders, support)  
- Improve **medication adherence** and patient experience  
- Reduce **avoidable inbound contact** and downstream complications  
- Provide a foundation for **responsible “next-best” support** (information, prompts, non-clinical recommendations)

---

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/late-refill-risk-prediction.git
cd late-refill-risk-prediction
```
(Alternatively, download the repository as a ZIP file and extract it.)

---

### 2. Install Dependencies
Ensure you have **Python 3.8+** installed, then run:
```bash
pip install -r requirements.txt
```

---

### 3. Download CMS Data

The model requires the following **synthetic datasets from CMS**:

- **Prescription Drug Events (PDE)** → Download PDE ZIP (2010)  
- **Beneficiary Summary** → Download Beneficiary ZIP  

#### Data Setup
- Extract the CSV files from the ZIPs  
- Place them inside:
```
data/raw/
```

> Note: Update file paths in the notebook if filenames or directories differ from defaults.

---

### 4. Run the Pipeline

Open the main notebook:
```bash
jupyter notebook notebooks/pharmacy2u_late_refill.ipynb
```

Run all cells from **top to bottom** to generate:

-  **Target Labeling**  
  - Defines "late" refills using a 7-day grace window  

-  **Model Training**  
  - Logistic Regression  
  - XGBoost  

-  **Performance Metrics**  
  - Precision-Recall (PR) curves  
  - AUC scores  

-  **Explainability**  
  - Feature importance  
  - Key risk drivers  

-  **Patient Timeline**  
  - Individual refill history  
  - Predicted risk visualization  

---

## Important Limitations

- **Synthetic Data**  
  This project uses fully synthetic data. Results should **not** be applied to real Medicare populations.

- **Not Clinical Advice**  
  This is a data science exercise. Outputs are **not medical recommendations**.
