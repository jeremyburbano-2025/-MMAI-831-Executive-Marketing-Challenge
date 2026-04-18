# Credit Canada — DCP Completion Prediction & Intervention Framework

## About

This repository contains the technical pipeline, analysis outputs, and visualisations for the **MMAI 831 Executive Marketing Challenge** at Queen's University Smith School of Business (MMAI 2026). The project was developed by **Team Broadview Analytics** in partnership with **Credit Canada**, a national non-profit credit counselling agency.

The goal: build a predictive framework that identifies which Debt Consolidation Program (DCP) clients are most likely to drop out — and when — so Credit Canada can intervene earlier and help more families complete their plans.

The pipeline combines **binary classification** (Logistic Regression, Random Forest, XGBoost, LightGBM, Decision Tree), **Cox Proportional Hazards survival analysis**, **K-Means persona segmentation**, and **SHAP explainability** into a single reproducible Python script. Everything runs from one command and produces all charts, CSV artifacts, and a processed Excel workbook.

> *From 39% to 48%: a predictive framework to help Credit Canada finish more plans — projected to deliver 147–190 additional completed families per year.*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context](#2-business-context)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Exploratory Analysis & Statistical Tests](#5-exploratory-analysis--statistical-tests)
6. [Client Personas](#6-client-personas)
7. [Modelling Strategy](#7-modelling-strategy)
8. [Model Results](#8-model-results)
9. [SHAP Driver Analysis](#9-shap-driver-analysis)
10. [Survival Analysis (Cox Proportional Hazards)](#10-survival-analysis-cox-proportional-hazards)
11. [Intervention Simulation & ROI](#11-intervention-simulation--roi)
12. [Fairness Audit](#12-fairness-audit)
13. [Recommendations](#13-recommendations)
14. [Limitations & Future Work](#14-limitations--future-work)
15. [How to Run](#15-how-to-run)
16. [Repository Structure](#16-repository-structure)
17. [Team](#17-team)

---

## 1. Executive Summary

Credit Canada's Debt Consolidation Program (DCP) consolidates unsecured debts into one monthly payment, with creditors typically waiving interest under a fair-share agreement. Across the 3,212 closed plans in our sample, only **39.23%** reached successful completion. The most common exit — non-payment at **46.6%** — means more clients walk away than finish.

Team Broadview Analytics built an end-to-end machine-learning pipeline that:

- Predicts completion from **intake data alone** (AUC **0.915**, recall **0.836**)
- Improves to AUC **0.932** and recall **0.873** when the first six months of payment behaviour are added
- Models **when** dropout occurs using Cox Proportional Hazards survival analysis
- Segments clients into **five actionable personas** via K-Means clustering
- Projects **+147 to +190 additional completions per year** through three stacked interventions

**Five findings that drive the strategy:**

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Early payment behaviour is destiny | Completers average ≥ 0.87 payment score in months 1–6; dropouts average 0.54–0.58. Gap appears in month one. |
| 2 | Payday debt is the single largest intake red flag | Completion drops from 45.2% → 29.6% (Chi² = 76.95, p < 10⁻¹⁸) |
| 3 | Plan duration matters almost as much as money | 2nd-largest SHAP driver — shorter plans are easier to finish |
| 4 | Geography is not uniform | Completion ranges from 77.8% (QC, n=54) to 19.0% (NL). Ontario (82% of book) sits at 39.4% |
| 5 | One persona carries disproportionate risk | P3 (younger, lower-income, payday-reliant, Ontario; n=708) completes at only 29.9% |

---

## 2. Business Context

### What a DCP Is

A Debt Consolidation Program rolls multiple unsecured debts (credit cards, payday loans, lines of credit) into one monthly payment. Creditors sign a fair-share agreement — they accept reduced or zero interest and take a slice of each completed payment as revenue. Plans typically run **four to five years**.

### Why Completion Is the Metric That Matters

Enrolment is easy. Completion is hard. If a client drops out mid-plan:

- They go back into collections at full interest rates
- Credit Canada loses the fair-share revenue it was counting on
- The creditor trust that makes the whole model work gets eroded

### The Numbers We Start From

| Metric | Value |
|--------|-------|
| Closed plans in dataset | 3,212 |
| New enrolments per year | ~2,200 |
| Completion rate | 39.23% (1,260 successful) |
| Non-payment exits | 46.6% (1,497 dropouts) |
| Ontarians | 82.5% of the book |

Credit Canada helped **41,000+ Canadians** in 2023. Even small percentage improvements scale into hundreds of families per year.

### SMART Objectives

- Two validated models with test AUC ≥ 0.78 and recall ≥ 0.70 on the minority completion class
- Low / Medium / High risk tier deployable at intake
- Month-3 re-score trigger using early payment behaviour
- 3–5 high-impact, low-cost intervention recommendations with estimated uplift

---

## 3. Dataset

**3,212 rows × 38 raw columns**, expanded to **70 columns** after feature engineering. Each row is one person who enrolled in a DCP and has since closed their plan — either by finishing or by exiting through one of five other paths.

### Outcome Distribution

| Exit Status | Count | % |
|-------------|------:|----:|
| Non Payment | 1,497 | 46.6% |
| **Successful Completion** | **1,260** | **39.2%** |
| Consumer Proposal | 293 | 9.1% |
| Self Administration | 103 | 3.2% |
| Bankruptcy | 54 | 1.7% |
| Deceased | 5 | 0.2% |

Everything except Successful Completion is treated as **target = 0**. This is a binary classification problem.

### Three Data-Quality Realities

1. **`reason_for_DCP_at_activation` is 63% missing.** Rather than impute (which would mean making up data for 2 out of every 3 records), we created a `reason_missing_flag`. This preserves the signal ("was this field populated?") without inventing values.

2. **Payment-status data decays over time.** Missingness climbs from 1.3% at month 1 to 38.5% at month 12 — plans that ended early never generated months 7–12. That's why modelling focuses on **months 1–6** (enough signal, not too much missingness).

3. **Class imbalance.** Only 39% of clients completed. Without handling, models will lazily predict "won't complete" for everyone and still look ~61% accurate. The fix: **class-weight balancing** and evaluation on **recall**, not accuracy.

---

## 4. Feature Engineering

All features are built in `engineer_features()` within the pipeline. Five families of engineered features:

### 4.1 Financial Ratios

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `debt_to_income` | DCP debt balance ÷ monthly income | $30K debt means very different things for $2K vs $8K earners |
| `expense_to_income` | Monthly expenses ÷ monthly income | Measures cash-flow pressure |
| `disposable_income` | Monthly income − monthly expenses | Raw headroom |
| `payment_burden_ratio` | (Debt ÷ duration) ÷ monthly income | What share of income each payment consumes |
| `net_worth_proxy` | Asset value − asset amount owed | Balance-sheet position |
| `payday_ratio` | Payday debt ÷ total DCP debt balance | How much of the debt is high-cost |

### 4.2 Geographic

| Feature | Logic |
|---------|-------|
| `province_grouped` | Small provinces bucketed as "Other" (several have < 30 obs) |
| `ontario_flag` | Binary — 82.5% of clients are from Ontario |

### 4.3 Demographic

| Feature | Logic |
|---------|-------|
| `age_group` | Binned: <25, 25–34, 35–44, 45–54, 55–64, 65+ |
| `household_density` | Household size ÷ (dependents + 1) |
| `dependents_flag` | 1 = any dependents at all |

### 4.4 Missingness-as-Information Flags

| Feature | What It Captures |
|---------|-----------------|
| `reason_missing_flag` | `reason_for_DCP` is 63% missing — likely correlates with intake workflow |
| `occupation_missing` | Occupation field was blank |
| `employment_missing` | Employment status field was blank |

### 4.5 Early Payment Behaviour (Months 1–6)

Raw `pmtstatus01`–`pmtstatus12` are text (`full` / `partial` / `zero`). Converted to numeric payment scores: **full = 1.0, partial = 0.5, zero = 0.0**.

| Feature | Definition |
|---------|-----------|
| `early_payment_score` | Mean of months 1–6 payment scores |
| `early_full_count` | Count of full payments in months 1–6 |
| `early_zero_count` | Count of zero payments in months 1–6 |
| `early_missed_any` | 1 = at least one zero payment in months 1–6 |
| `pmt_score_01`…`pmt_score_06` | Per-month individual scores |

This is the **single most important group of features** in the project.

---

## 5. Exploratory Analysis & Statistical Tests

### 5.1 Payday Debt Is the Biggest Demographic Red Flag

| Group | Completion Rate |
|-------|:--------------:|
| No payday debt | 45.2% |
| Has payday debt | 29.6% |

**34% relative drop.** Chi² = 76.95, p = 1.75 × 10⁻¹⁸.

### 5.2 Early Payment Behaviour Is Destiny

| Group | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Completers | 0.87+ | 0.87+ | 0.87+ | 0.87+ | 0.87+ | 0.87+ |
| Dropouts | 0.54–0.58 | 0.54–0.58 | 0.54–0.58 | 0.54–0.58 | 0.54–0.58 | 0.54–0.58 |

The gap appears in **month one** and never closes.

### 5.3 Income and Debt-to-Income Both Matter

| Test | Statistic | p-value |
|------|----------:|--------:|
| Chi² payday-debt vs completion | 76.95 | 1.75 × 10⁻¹⁸ |
| Welch t-test: income (complete vs drop) | −3.15 | 0.0017 |
| Welch t-test: debt-to-income | 3.70 | 0.00022 |
| Chi² province vs completion | 47.35 | 3.37 × 10⁻⁷ |

All four pre-registered tests significant at p < 0.002.

---

## 6. Client Personas

K-Means clustering (k=5) on standardised intake signals — age, income, DTI, payday-debt flag, Ontario flag, dependents flag, disposable income.

| Persona | n | Median Age | Median Income | Median DTI | % Payday | Completion |
|---------|--:|:----------:|:-------------:|:----------:|:--------:|:----------:|
| P2: Younger · lower-income · Ontario | 962 | 33 | $2,200 | 4.63 | 0.0% | **46.9%** |
| P1: Mid-career | 247 | 48 | $4,931 | 11.10 | 24.7% | 41.3% |
| P5: Younger · lower-income | 523 | 36 | $2,800 | 4.12 | 23.7% | 38.2% |
| P4: Mid-career · with dependents · Ontario | 772 | 38 | $3,243 | 3.44 | 44.7% | 38.2% |
| **P3: Younger · payday-reliant** | **708** | **37** | **$2,400** | **4.07** | **100.0%** | **29.9%** |

**P3 is the highest-leverage intervention target:** 22% of enrolments, worst completion rate by a wide margin, and 100% have payday debt.

---

## 7. Modelling Strategy

### 7.1 Two Parallel Feature Sets

| Feature Set | Features Available | When Deployable | Purpose |
|-------------|-------------------|:---------------:|---------|
| **Intake-only** | 23 intake variables (demographic, financial, geographic, missingness flags) | Day 0 | "Can we flag at-risk clients on day one?" |
| **Intake + Early** | Same 23 + 10 months-1–6 payment features | Month 6 | "Does early payment behaviour buy us meaningful lift?" |

### 7.2 Five Algorithms

| Algorithm | Rationale |
|-----------|-----------|
| Logistic Regression | Interpretable baseline; coefficients map to a scorecard |
| Decision Tree | Fully transparent rule set; useful for counsellor explanation |
| Random Forest | Ensemble bagging; strong on tabular data |
| XGBoost | Gradient-boosted trees; common benchmark |
| LightGBM | Histogram-based boosting; fast, handles categoricals natively |

**5 algorithms × 2 feature sets = 10 models trained and compared.**

### 7.3 Training Protocol

```text
Raw data ──→ Feature engineering ──→ Stratified 70/15/15 split
                                         │
                    ┌────────────────────┘
                    ▼
             sklearn Pipeline
             (imputation → scaling → one-hot encoding)
                    │
                    ▼
         5-fold stratified CV on training set
                    │
                    ▼
         Fit on train → evaluate on validation → final metrics on test
                    │
                    ▼
         SHAP on held-out test set (TreeExplainer / LinearExplainer)
```

- **Stratified split** preserves the 39/61 class balance across all three splits
- **Pipeline-wrapped preprocessing** — imputer fits on training data only, preventing data leakage
- **Class-weight balancing** — minority class weighted more heavily so models don't default to "won't complete"
- **Recall as primary KPI** — false negatives are families the model wrongly labels safe; they drop out without intervention

### 7.4 Survival Analysis (Cox PH)

In parallel, a **Cox Proportional Hazards** model answers a different question: not *if* a client will drop out, but *when*. The Cox model:

- Uses `duration_of_dcp` as the time variable and `target` as the event indicator
- Includes `debt_to_income`, `has_payday_debt`, `early_payment_score`, `payment_burden_ratio`, `housing_status`, `age_at_activation`, and `disposable_income`
- Outputs a **monthly hazard profile** and **median survival time** per client
- Feeds the **Month-3 counsellor alert window**

```text
Raw data ──→ Feature engineering ──→ [LR classifier] ──→ Risk tier (Low/Med/High)
                                 ──→ [Cox model]     ──→ Dropout timing window
                                 ──→ [K-Means]       ──→ Client persona
```

---

## 8. Model Results

### 8.1 Full 10-Model Comparison

#### Intake-Only

| Model | CV AUC | Val AUC | Test AUC | Recall | Precision | F1 |
|-------|:------:|:-------:|:--------:|:------:|:---------:|:--:|
| **Logistic Regression** | **0.882** | **0.884** | **0.915** | **0.836** | **0.794** | **0.814** |
| Random Forest | 0.880 | 0.884 | 0.907 | 0.815 | 0.798 | 0.806 |
| XGBoost | 0.875 | 0.876 | 0.899 | 0.778 | 0.774 | 0.776 |
| LightGBM | 0.875 | 0.874 | 0.897 | 0.799 | 0.763 | 0.780 |
| Decision Tree | 0.812 | 0.823 | 0.862 | 0.857 | 0.730 | 0.788 |

#### Intake + Early Payment

| Model | CV AUC | Val AUC | Test AUC | Recall | Precision | F1 |
|-------|:------:|:-------:|:--------:|:------:|:---------:|:--:|
| **Logistic Regression** | **0.896** | **0.890** | **0.932** | **0.873** | **0.805** | **0.838** |
| Random Forest | 0.900 | 0.894 | 0.926 | 0.847 | 0.800 | 0.823 |
| XGBoost | 0.895 | 0.899 | 0.921 | 0.783 | 0.783 | 0.783 |
| LightGBM | 0.895 | 0.889 | 0.911 | 0.799 | 0.770 | 0.784 |
| Decision Tree | 0.850 | 0.820 | 0.863 | 0.815 | 0.778 | 0.796 |

### 8.2 Why Logistic Regression Won

- **Only 3,212 rows.** Boosting models are data-hungry and can overfit on smaller datasets.
- **Strong L2 regularisation** kept coefficients from running wild.
- **The signal is largely linear.** When `early_payment_score` is 1.0, you almost always complete; when it's 0.0, you almost always drop out.

The best model is also the most interpretable — there is no accuracy-interpretability trade-off to apologise for.

### 8.3 The Delta Between Feature Sets

| Metric | Intake-Only | Intake + Early | Delta |
|--------|:-----------:|:--------------:|:-----:|
| Test AUC | 0.915 | 0.932 | +0.017 |
| Recall | 0.836 | 0.873 | +0.037 |
| Precision | 0.794 | 0.805 | +0.011 |
| F1 | 0.814 | 0.838 | +0.024 |

---

## 9. SHAP Driver Analysis

SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature contributions in probability points.

### 9.1 Intake + Early Payment Model — Top 12 Drivers

| Rank | Feature | Mean |SHAP| |
|:----:|---------|:----:|
| 1 | `early_payment_score` | 1.629 |
| 2 | `duration_of_dcp` | 1.247 |
| 3 | `early_full_count` | 0.476 |
| 4 | `payment_burden_ratio` | 0.332 |
| 5 | `monthly_expenses_at_activation` | 0.290 |
| 6 | `has_payday_debt_on_DCP_at_activation` | 0.282 |
| 7 | `disposable_income` | 0.229 |
| 8 | `expense_to_income` | 0.228 |
| 9 | `DCP_debt_balance_at_activation` | 0.202 |
| 10 | `monthly_income_at_activation` | 0.186 |
| 11 | `early_missed_any` | 0.182 |
| 12 | `early_zero_count` | 0.168 |

### 9.2 Intake-Only Model — Top 10 Drivers

| Rank | Feature | Mean |SHAP| |
|:----:|---------|:----:|
| 1 | `duration_of_dcp` | 1.454 |
| 2 | `payment_burden_ratio` | 0.514 |
| 3 | `has_payday_debt_on_DCP_at_activation` | 0.353 |
| 4 | `monthly_expenses_at_activation` | 0.311 |
| 5 | `expense_to_income` | 0.201 |
| 6 | `monthly_income_at_activation` | 0.196 |
| 7 | `disposable_income` | 0.163 |
| 8 | `number_in_household_at_activation` | 0.157 |
| 9 | `dependents_at_activation` | 0.130 |
| 10 | `DCP_debt_balance_at_activation` | 0.117 |

### 9.3 Three Insights

1. **Behaviour > demographics.** No demographic variable (age, gender, location) cracks the top 5. What someone *does* predicts the future better than who they *are*.
2. **Payday debt persists.** Even after the model has seen income, expenses, debt, and payment burden, the payday-debt flag still adds unique information (mean |SHAP| = 0.28).
3. **Duration is a design lever.** Plan duration is set by Credit Canada, not by the client. Its prominence as the #2 driver suggests renegotiating plan length for high-risk clients could move outcomes as much as behavioural coaching.

### 9.4 "Meet Alex" — Local SHAP Walkthrough

"Alex" is a composite client built from the median stats of Persona P3: age 37, Ontario, $2,400/mo income, has payday debt, 60-month plan, month-1 payment score 0.5.

```
Population baseline:                  39% probability of completion
  Has payday debt:          −18pp  →  21%
  Low monthly income:        −6pp  →  15%
  60-month duration:         −5pp  →  10%
  Low early-payment score:   −5pp  →   5%
                              ───────────
  Final prediction:                    5%   →  FLAG: HIGH RISK
```

---

## 10. Survival Analysis (Cox Proportional Hazards)

The LR classifier answers **IF** a client will complete. The Cox model answers **WHEN** they are most likely to drop out. These are complementary — LR tiers the client at Day 0; Cox sets the counsellor alert window.

### 10.1 Cox Model Coefficients

| Covariate | Hazard Ratio | 95% CI | p-value | Interpretation |
|-----------|:------------:|:------:|:-------:|----------------|
| `early_payment_score` | **2.223** | [1.563, 3.161] | 8.8 × 10⁻⁶ | Higher payment score → 2.2× higher hazard of *completing* (protective factor) |
| `payment_burden_ratio` | **1.371** | [1.211, 1.552] | 6.3 × 10⁻⁷ | Higher burden → faster dropout |
| `housing_status: Rent` | **0.796** | [0.683, 0.928] | 0.0036 | Renters have 20% lower completion hazard vs owners |
| `debt_to_income` | 0.941 | [0.928, 0.955] | 1.4 × 10⁻¹⁵ | Higher DTI → slower time to completion |
| `age_at_activation` | 0.990 | [0.985, 0.994] | 8.9 × 10⁻⁷ | Younger clients drop out sooner |
| `has_payday_debt` | 0.918 | [0.810, 1.041] | 0.184 | Directional but not independently significant in Cox (captured by other covariates) |
| `disposable_income` | 1.000 | [1.000, 1.000] | 0.994 | No independent timing signal once other financials controlled |

### 10.2 Kaplan-Meier Survival by LR Risk Tier

Clients are split into **Low / Medium / High** risk terciles using the LR intake model's predicted probabilities. The Kaplan-Meier curves show:

- **High-risk clients** see the steepest active-survival decline in months 1–12
- **Medium-risk** decline steadily through months 12–36
- **Low-risk** maintain high survival probability throughout the plan

This validates the **Month-3 re-engagement trigger** — high-risk clients show the most attrition early, when intervention can still change the trajectory.

---

## 11. Intervention Simulation & ROI

### 11.1 Four Scenarios

| Scenario | Baseline Completions | Projected Completions | Extra Families/yr | Uplift |
|----------|:--------------------:|:---------------------:|:-----------------:|:------:|
| Do nothing | 863 | 863 | 0 | 0% |
| Intake-only flag (check-ins) | 863 | 949 | **+86** | +10% |
| Intake + month-3 re-engagement | 863 | 1,010 | **+147** | +17% |
| Both combined (ceiling) | 863 | 1,053 | **+190** | +22% |

### 11.2 ROI Calculation (Mid Scenario)

| Component | Value |
|-----------|------:|
| Clients flagged high-risk (30% of 2,200) | 660 |
| Cost per counsellor contact (fully-loaded) | $60 |
| Incremental intervention cost | ~$40,000 |
| Additional completions | +147 |
| Fair-share revenue per completed plan* | ~$5,000 |
| Additional revenue recovered | ~$735,000 |
| **Net benefit** | **~$695,000** |
| **Return on investment** | **~18×** |
| Cost per additional family helped | ~$270 |

*\*$5,000 is an industry benchmark. Replace with Credit Canada actuals before external reporting.*

---

## 12. Fairness Audit

Test-set AUC computed separately for demographic subgroups to check for systematic mis-prediction:

| Attribute | Group | n | AUC | Completion Rate |
|-----------|-------|--:|:---:|:--------------:|
| Gender | Female | 274 | 0.934 | 42.3% |
| Gender | Male | 190 | 0.921 | 35.3% |
| Age | <25 | 55 | 0.899 | 34.5% |
| Age | 25–34 | 154 | 0.920 | 40.9% |
| Age | 35–44 | 106 | 0.961 | 34.9% |
| Age | 45–54 | 92 | 0.953 | 41.3% |
| Age | 55–64 | 38 | 0.949 | 31.6% |
| Age | 65+ | 35 | 0.893 | 57.1% |
| Province | Ontario | 396 | 0.931 | 38.4% |

No subgroup shows systematic mis-prediction. The recall gap across genders is < 4 points.

---

## 13. Recommendations

### Action 1: Score Every New Enrolment at Day 0

Assign **Low / Medium / High** risk tier using the intake-only LR model. High-tier clients enter an **intensive onboarding track** from Week 1 — early check-in calls, payday-debt-specific financial literacy materials, and review of plan duration.

**Expected uplift:** +86 completions/yr.

### Action 2: Month-3 Re-Engagement Trigger

Re-score all clients using the intake + early-payment model at Month 3. Clients in the **bottom 30%** of predicted probability trigger a counsellor check-in and a plan-duration review.

**Expected uplift:** +147 completions/yr (stacked with Action 1).

### Action 3: Target Persona P3

Build a **payday-debt-specific financial literacy track** with direct-mail and SMS reminders timed to first payment. P3 is 22% of the book but only 29.9% completion — the single highest-leverage segment.

### Deployment Roadmap

| Phase | Timeline | Action |
|-------|----------|--------|
| **Pilot** | Months 1–6 | Run intake-only model in parallel with current practice; validate risk tiers against observed outcomes |
| **Deploy** | Month 7+ | Integrate Low/Med/High tier into CRM at enrolment; trigger intensive onboarding for High tier |
| **Expand** | Month 12+ | Add month-3 re-score as second trigger once intake tier is stable; retrain annually on new closed-plan data |

---

## 14. Limitations & Future Work

1. **Intervention uplift is literature-based, not pilot-validated.** The 10%/17%/22% estimates draw on published DCP intervention literature, not a Credit Canada RCT. A pilot study is needed to confirm effect sizes before scaling.
2. **Sample is 82% Ontario.** Regional generalisation should be tested on a broader sample before national rollout.
3. **Only closed plans.** The dataset contains only closed plans; ongoing plans may have different characteristics. Model should be validated on a prospective cohort before deployment.
4. **`reason_for_DCP` is 63% missing.** The missing-flag is informative but likely correlates with data-entry workflow rather than client characteristics.
5. **Fair-share dollar figure is an industry estimate.** The $5,000 per-plan figure should be replaced with Credit Canada's actual recovery data before external communication.
6. **No causal inference.** The observational dataset does not support defensible causal claims. A pilot RCT is the proper path to causality.

---

## 15. How to Run

```bash
pip install -r requirements.txt
python credit_canada_pipeline.py --input "Credit Canada Data.xlsx"
```

### Pipeline Outputs

| Output | Description |
|--------|-------------|
| `outputs/Credit_Canada_Data_PROCESSED.xlsx` | Processed workbook with 7 sheets |
| `outputs/model_comparison.csv` | 10-model performance table across both feature sets |
| `outputs/persona_profiles.csv` | Five-persona profile table with completion rates |
| `outputs/intervention_sim.csv` | Four intervention scenarios with projected completions |
| `outputs/hypothesis_tests.csv` | Statistical significance tests (Chi², Welch t) |
| `outputs/fairness_by_group.csv` | AUC parity diagnostics by gender, age, province |
| `outputs/shap_top_intake-only.csv` | Top 20 SHAP drivers for intake-only champion |
| `outputs/shap_top_intake_early.csv` | Top 20 SHAP drivers for intake+early champion |
| `outputs/cox_summary.csv` | Cox proportional hazards coefficient table |

### Charts (300 dpi, editorial palette)

| Chart | Description |
|-------|-------------|
| `charts/01_outcome_breakdown.png` | DCP outcome distribution |
| `charts/02_completion_by_province.png` | Completion rates by province |
| `charts/03_payday_effect.png` | Completion impact of payday debt |
| `charts/04_early_payment_heatmap.png` | Months 1–6 payment pattern heatmap |
| `charts/05_dti_box.png` | Debt-to-income distribution by outcome |
| `charts/06_model_comparison.png` | AUC comparison across all 10 models |
| `charts/07_roc_curves.png` | ROC curves for champion models |
| `charts/08_shap_intake-only.png` | SHAP importance for intake-only champion |
| `charts/08_shap_intake_early.png` | SHAP importance for intake+early champion |
| `charts/09_personas.png` | Completion rates by persona |
| `charts/10_intervention_sim.png` | Projected intervention lift |
| `charts/11_survival_by_risk_tier.png` | Kaplan-Meier survival curves by LR risk tier |
| `charts/12_cox_hazard_forest.png` | Hazard ratio forest plot from Cox model |

---

## 16. Repository Structure

```text
-MMAI-831-Executive-Marketing-Challenge/
├── README.md                                  # This document — full project report
├── credit_canada_pipeline.py                  # End-to-end pipeline (feature eng, modelling, Cox, SHAP, charts)
├── requirements.txt                           # Python dependencies
├── Credit_Canada_Final_Report.docx            # Final written report
├── Credit Canada Data.xlsx                    # Raw source data (3,212 records)
├── Credit_Canada_Data_PROCESSED.xlsx          # Processed workbook snapshot
├── CreditCanada_Proposal_TeamBroadview_Final.pdf
├── Credit_Canada_Presentation_Final_V4.pptx
├── EMC Rubric Updated.xlsx
├── charts/                                    # All generated charts (300 dpi PNG)
│   ├── 01_outcome_breakdown.png
│   ├── 02_completion_by_province.png
│   ├── 03_payday_effect.png
│   ├── 04_early_payment_heatmap.png
│   ├── 05_dti_box.png
│   ├── 06_model_comparison.png
│   ├── 07_roc_curves.png
│   ├── 08_shap_intake-only.png
│   ├── 08_shap_intake_early.png
│   ├── 09_personas.png
│   ├── 10_intervention_sim.png
│   ├── 11_survival_by_risk_tier.png
│   └── 12_cox_hazard_forest.png
├── csv/                                       # CSV artifacts from pipeline runs
│   ├── fairness_by_group.csv
│   ├── hypothesis_tests.csv
│   ├── intervention_sim.csv
│   ├── model_comparison.csv
│   ├── persona_profiles.csv
│   ├── shap_top_intake-only.csv
│   └── shap_top_intake_early.csv
└── outputs/
    └── cox_summary.csv
```

---

## 17. Team

**Team Broadview Analytics**
Queen's University · Smith School of Business · MMAI 2026 · AI in Marketing

---

*Pipeline seed: 42 · Dependencies: scikit-learn, xgboost, lightgbm, shap, lifelines, pandas, numpy, matplotlib, seaborn, openpyxl*
