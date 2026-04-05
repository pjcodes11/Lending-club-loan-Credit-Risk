# Loan Approval Decision Engine
### End-to-end credit risk system built on 176,868 Lending Club loans

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Credit%20Risk-darkgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

A two-layer loan approval system that mirrors how real banks and NBFCs make credit decisions — a **machine learning PD model** combined with a **rule-based policy override layer**. Built on real Lending Club data with full business-objective threshold tuning, reason coding, and portfolio simulation.

> Most credit risk projects stop at a model score. This one goes further — threshold tuning for business objectives, hard policy rules with regulatory reason codes, and a what-if simulator that quantifies the approval volume vs. expected loss trade-off.

---

## Results

| Metric | Value |
|---|---|
| Dataset | 176,868 resolved Lending Club loans |
| AUC (test) | **0.7412** |
| Gini coefficient | **0.4823** |
| KS statistic | **0.3535** |
| CV AUC (5-fold) | **0.7405 ± 0.0012** |
| Working threshold | 0.635 |
| Approval rate | 67.7% |
| Bad rate (approved) | **13.92%** vs 15.00% cap ✓ |
| Rule overrides | 20.9% of applications |

---

## Architecture

```
Raw Lending Club CSV (150+ columns)
        │
        ▼
┌─────────────────────────┐
│   Data Cleaning         │  Parse text fields, impute medians,
│                         │  derive PTI / LTI / FICO midpoint
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Feature Engineering   │  Log transforms, binary risk flags,
│                         │  categorical dummies (home, purpose)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   PD Model              │  Logistic Regression + StandardScaler
│   (Logistic Regression) │  class_weight='balanced', C=0.5
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Threshold Tuning      │  200 cutoffs evaluated across:
│                         │  bad rate cap / max profit / max F1
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│           Rule-Based Override Layer         │
│                                             │
│  Hard Declines (HD001–HD007)                │
│  ├── DTI > 50%                              │
│  ├── FICO < 580                             │
│  ├── Bankruptcy on record                   │
│  ├── 3+ delinquencies in 24 months          │
│  ├── Interest rate >= 28%                   │
│  ├── PTI > 30%                              │
│  └── 3+ adverse public records              │
│                                             │
│  Auto-Approvals (AA001–AA002)               │
│  ├── Prime profile (FICO>=750, DTI<20%)     │
│  └── Stable prime (FICO>=720, 7+ yr emp)    │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│   Final Decision        │  APPROVE / DECLINE /
│   + Audit Trail         │  AUTO_APPROVE / HARD_DECLINE
└─────────────────────────┘  + reason code + reason description
```

---

## Project Structure

```
loan-approval-engine/
│
├── data/
│   └── accepted_2007_to_2018Q4.csv     ← download from Kaggle (not tracked)
│
├── notebooks/
│   └── loan_approval_engine_lending_club.ipynb
│
├── outputs/
│   ├── lc_loan_decisions_audit.csv      ← full audit trail (row-level)
│   ├── lc_decision_band_summary.csv     ← Tableau portfolio summary
│   ├── lc_rule_trigger_frequency.csv    ← which rules fire most
│   └── lc_policy_simulation.csv         ← what-if scenario results
│
├── README.md
└── requirements.txt
```

---

## Notebook Sections

| Section | What it does |
|---|---|
| 0 — Setup | Library imports and plot config |
| 1 — Data Loading | Column-pruned CSV load (23 of 150 columns) |
| 2 — Target Variable | Binary default flag from resolved loan statuses |
| 3 — Data Cleaning | Text parsing, FICO midpoint, median imputation |
| 4 — EDA | Default rate by grade, FICO, DTI, purpose, int rate |
| 5 — Feature Engineering | Log transforms, risk flags, categorical dummies |
| 6 — Model Training | Logistic Regression pipeline, 5-fold CV |
| 7 — Evaluation | AUC, Gini, KS, ROC curve, calibration |
| 8 — Threshold Tuning | 200-point sweep across 3 business objectives |
| 9 — Rule Layer | 7 hard decline + 2 auto-approve rules with reason codes |
| 10 — Override Analysis | Default rate validation by decision category |
| 11 — Grade Heatmap | Decision mix and bad rate by Lending Club grade |
| 12 — Audit Trail | Row-level decision log with risk tier and timestamp |
| 13 — Policy Simulation | 7 risk appetite scenarios, approval vs. expected loss |
| 14 — Exports | 4 CSVs for Tableau dashboarding |
| 15 — Final Summary | Full portfolio metrics printout |

---

## Key Concepts

**Why logistic regression over XGBoost?**
Basel III and RBI guidelines require interpretable models. Logistic regression coefficients are directly auditable — every decision can be traced to individual feature contributions, which is required for adverse action notices.

**Why a rule layer on top of the model?**
No bank uses a model alone. Hard decline rules handle cases so extreme that no model score can compensate (e.g. bankruptcy on record). Auto-approve rules fast-track premium borrowers without model review. The 20.9% override rate in this project reflects a realistic policy layer.

**Why tune thresholds?**
The default scikit-learn threshold of 0.5 is almost never optimal for credit. Approving a bad loan costs ~3.75× more than declining a good one. This asymmetry means the profit-maximising threshold is always well above 0.5. This project evaluates 200 thresholds against three business objectives and selects based on a 15% bad rate cap.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/picodes11/loan-approval-engine.git
cd loan-approval-engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `accepted_2007_to_2018Q4.csv` from [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place it in the `data/` folder.

### 4. Run the notebook
```bash
jupyter notebook notebooks/loan_approval_engine_lending_club.ipynb
```

> **Quick start:** The notebook has `nrows=200_000` set by default — runs in under 2 minutes. Remove this line to use the full 2.2M row dataset.

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
jupyter>=1.0
```

---

## Outputs for Tableau

| File | Use |
|---|---|
| `lc_loan_decisions_audit.csv` | Row-level audit trail — every loan with PD, decision, reason code, and actual outcome |
| `lc_decision_band_summary.csv` | Portfolio composition by decision category — drives the approval mix dashboard |
| `lc_rule_trigger_frequency.csv` | How often each reason code fires — drives the policy monitoring dashboard |
| `lc_policy_simulation.csv` | Approval rate and expected loss across 7 risk appetite scenarios |

---

## Skills Demonstrated

- **Credit risk modelling** — PD model with AUC, Gini, KS evaluation; logistic regression scorecard approach aligned with Basel III
- **Business-objective optimisation** — threshold tuning across bad rate cap, profit maximisation, and F1 simultaneously
- **Policy rule design** — RBI-aligned hard decline and auto-approve rules with regulatory reason coding
- **Portfolio analytics** — what-if simulation quantifying the approval volume vs. expected loss trade-off
- **Production thinking** — full audit trail with reason codes, risk tiers, and decision timestamps

---

## Author

**Pranjay Kamalvanshi**
Data Analyst · Mirza International · Delhi NCR
MSc Statistics, University of Delhi

[LinkedIn](https://linkedin.com/in/pranjay-kamalvanshi) · [GitHub](https://github.com/picodes11)
