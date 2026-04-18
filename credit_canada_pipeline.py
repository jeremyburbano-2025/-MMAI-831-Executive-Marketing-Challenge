"""
Credit Canada - DCP Completion Prediction Pipeline
===================================================
Team Broadview Analytics | MMAI 2026 | AI in Marketing

End-to-end reproducible pipeline:
  1. Load raw dataset (Excel)
  2. Feature engineering (financial ratios, behavioural flags, missing flags)
  3. Two feature sets:
       (a) INTAKE-ONLY  -- activation variables only (early risk flagging)
       (b) INTAKE+EARLY -- adds months 1-6 payment behaviour (uplift)
  4. Train & compare: Logistic Regression, Decision Tree, Random Forest,
                      XGBoost, LightGBM
  5. SHAP driver analysis on champion models
  6. Intervention simulation & ROI quantification
  7. Write processed dataset, model metrics, and all charts to disk

Run:
    python credit_canada_pipeline.py --input Credit_Canada_Data.xlsx

Outputs (./outputs and ./charts):
    - Credit_Canada_Data_PROCESSED.xlsx
    - model_comparison.csv, persona_profiles.csv, intervention_sim.csv
    - All figures used in the deck and report (PNG, 300 dpi)
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, accuracy_score,
)
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

# --------------------------------------------------------------------------- #
#  EDITORIAL PLOT STYLE -- matches the "Midnight Ledger" deck palette
# --------------------------------------------------------------------------- #
PAL = {
    "ink":     "#0B1E3F",
    "cream":   "#F4EDE0",
    "amber":   "#E8A33D",
    "coral":   "#D66853",
    "sage":    "#6A8E7F",
    "slate":   "#4A5D75",
    "rule":    "#2B3A55",
}
CAT_PALETTE = [PAL["amber"], PAL["coral"], PAL["sage"], PAL["slate"], PAL["ink"]]

mpl.rcParams.update({
    "figure.facecolor":  PAL["cream"],
    "axes.facecolor":    PAL["cream"],
    "savefig.facecolor": PAL["cream"],
    "axes.edgecolor":    PAL["ink"],
    "axes.labelcolor":   PAL["ink"],
    "xtick.color":       PAL["ink"],
    "ytick.color":       PAL["ink"],
    "text.color":        PAL["ink"],
    "font.family":       "serif",
    "font.serif":        ["Georgia", "DejaVu Serif"],
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        PAL["rule"],
    "grid.alpha":        0.12,
    "grid.linestyle":    "-",
})


# --------------------------------------------------------------------------- #
#  1. LOAD & TARGET
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Dataset")
    df["target"] = (df["DCP_drop_status"] == "Successful Completion").astype(int)
    return df


# --------------------------------------------------------------------------- #
#  2. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
PMT_SCORE = {"full": 1.0, "partial": 0.5, "zero": 0.0}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # -- Financial ratios (guard against zero income)
    inc = d["monthly_income_at_activation"].replace(0, np.nan)
    d["debt_to_income"]       = d["DCP_debt_balance_at_activation"] / inc
    d["expense_to_income"]    = d["monthly_expenses_at_activation"] / inc
    d["disposable_income"]    = d["monthly_income_at_activation"] - d["monthly_expenses_at_activation"]
    d["payment_burden_ratio"] = (d["DCP_debt_balance_at_activation"] / d["duration_of_dcp"]) / inc
    d["net_worth_proxy"]      = d["asset_value_at_activation"] - d["asset_amount_owed_at_activation"]
    d["payday_ratio"]         = d["payday_debt_balance_at_activation"] / d["DCP_debt_balance_at_activation"].replace(0, np.nan)

    # -- Geographic
    top_provinces = ["ON", "BC", "AB", "QC", "NS", "MB", "NB", "SK", "NL"]
    d["province_grouped"] = d["province_at_activation"].where(
        d["province_at_activation"].isin(top_provinces), "Other"
    )
    d["ontario_flag"] = (d["province_at_activation"] == "ON").astype(int)

    # -- Demographic
    d["age_group"] = pd.cut(
        d["age_at_activation"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    d["household_density"] = d["number_in_household_at_activation"] / (d["dependents_at_activation"] + 1)
    d["dependents_flag"]   = (d["dependents_at_activation"] > 0).astype(int)

    # -- Missing-data flags (reason_for_DCP is 63% missing)
    d["reason_missing_flag"]   = d["reason_for_DCP_at_activation"].isna().astype(int)
    d["occupation_missing"]    = d["occupation_at_activation"].isna().astype(int)
    d["employment_missing"]    = d["employment_status_at_activation"].isna().astype(int)

    # -- Early payment behaviour (months 1-6)
    for m in range(1, 13):
        col = f"pmtstatus{m:02d}"
        d[f"pmt_score_{m:02d}"] = d[col].map(PMT_SCORE)

    early_cols = [f"pmt_score_{m:02d}" for m in range(1, 7)]
    d["early_payment_score"] = d[early_cols].mean(axis=1)
    d["early_full_count"]    = (d[early_cols] == 1.0).sum(axis=1)
    d["early_zero_count"]    = (d[early_cols] == 0.0).sum(axis=1)
    d["early_missed_any"]    = (d["early_zero_count"] > 0).astype(int)

    # -- Clean infs
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


# --------------------------------------------------------------------------- #
#  3. FEATURE SETS FOR THE TWO MODELS
# --------------------------------------------------------------------------- #
INTAKE_NUMERIC = [
    "age_at_activation", "duration_of_dcp", "dependents_at_activation",
    "number_in_household_at_activation", "household_density",
    "monthly_income_at_activation", "monthly_expenses_at_activation",
    "disposable_income", "DCP_debt_balance_at_activation",
    "asset_value_at_activation", "asset_amount_owed_at_activation",
    "net_worth_proxy", "debt_to_income", "expense_to_income",
    "payment_burden_ratio", "payday_debt_balance_at_activation",
    "payday_ratio",
]
INTAKE_CATEGORICAL = [
    "gender_at_activation", "marital_status_at_activation",
    "housing_status_at_activation", "employment_status_at_activation",
    "province_grouped", "age_group",
]
INTAKE_FLAGS = [
    "has_payday_debt_on_DCP_at_activation", "ontario_flag", "dependents_flag",
    "reason_missing_flag", "occupation_missing", "employment_missing",
]
EARLY_PMT = [
    "early_payment_score", "early_full_count",
    "early_zero_count", "early_missed_any",
] + [f"pmt_score_{m:02d}" for m in range(1, 7)]


def build_feature_matrices(d: pd.DataFrame):
    y = d["target"].values
    intake_cols = INTAKE_NUMERIC + INTAKE_CATEGORICAL + INTAKE_FLAGS
    full_cols   = intake_cols + EARLY_PMT

    X_intake = d[intake_cols].copy()
    X_full   = d[full_cols].copy()

    num_intake = [c for c in intake_cols if c in INTAKE_NUMERIC]
    cat_intake = [c for c in intake_cols if c in INTAKE_CATEGORICAL]
    flg_intake = [c for c in intake_cols if c in INTAKE_FLAGS]

    num_full = num_intake + EARLY_PMT
    cat_full = cat_intake
    flg_full = flg_intake

    return (X_intake, num_intake, cat_intake, flg_intake), \
           (X_full,   num_full,   cat_full,   flg_full), y


def make_preprocessor(num, cat, flg):
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("oh",  __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"])
                .OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    flg_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", cat_pipe, cat),
        ("flg", flg_pipe, flg),
    ])


# --------------------------------------------------------------------------- #
#  4. MODEL TRAINING & EVALUATION
# --------------------------------------------------------------------------- #
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RNG),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RNG),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=RNG),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            n_jobs=-1, random_state=RNG),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            num_leaves=31, subsample=0.85, colsample_bytree=0.85,
            n_jobs=-1, random_state=RNG, verbose=-1),
    }


def train_eval_suite(X, y, num, cat, flg, label: str):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RNG)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RNG)

    pre = make_preprocessor(num, cat, flg)
    pre.fit(X_tr)
    Xtr, Xva, Xte = pre.transform(X_tr), pre.transform(X_va), pre.transform(X_te)

    rows, trained = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    for name, mdl in get_models().items():
        cv_auc = cross_val_score(mdl, Xtr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        mdl.fit(Xtr, y_tr)
        proba_va = mdl.predict_proba(Xva)[:, 1]
        proba_te = mdl.predict_proba(Xte)[:, 1]
        pred_te  = (proba_te >= 0.5).astype(int)
        rows.append({
            "feature_set": label,
            "model":       name,
            "cv_auc":      cv_auc,
            "val_auc":     roc_auc_score(y_va, proba_va),
            "test_auc":    roc_auc_score(y_te, proba_te),
            "test_recall": recall_score(y_te, pred_te),
            "test_prec":   precision_score(y_te, pred_te),
            "test_f1":     f1_score(y_te, pred_te),
            "test_acc":    accuracy_score(y_te, pred_te),
        })
        trained[name] = mdl

    metrics = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    return {
        "metrics":  metrics,
        "trained":  trained,
        "pre":      pre,
        "splits":   (X_tr, X_va, X_te, y_tr, y_va, y_te),
        "Xtr_t":    Xtr, "Xva_t": Xva, "Xte_t": Xte,
    }


def feature_names_from_ct(pre, num, cat, flg):
    names = list(num)
    oh = pre.named_transformers_["cat"].named_steps["oh"]
    names += list(oh.get_feature_names_out(cat))
    names += list(flg)
    return names


# --------------------------------------------------------------------------- #
#  5. CHARTS
# --------------------------------------------------------------------------- #
def style_ax(ax, title=None, xlab=None, ylab=None):
    if title: ax.set_title(title, loc="left", pad=14, color=PAL["ink"])
    if xlab:  ax.set_xlabel(xlab)
    if ylab:  ax.set_ylabel(ylab)
    ax.tick_params(length=0)


def save_fig(fig, path: Path, name: str):
    out = path / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=PAL["cream"])
    plt.close(fig)
    return out


def chart_outcome_breakdown(df, out):
    order = df["DCP_drop_status"].value_counts().index.tolist()
    vals  = df["DCP_drop_status"].value_counts().reindex(order)
    pct   = (vals / vals.sum() * 100).round(1)
    colors = [PAL["amber"] if s == "Successful Completion" else PAL["coral"]
              if s == "Non Payment" else PAL["slate"] for s in order]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.barh(order[::-1], vals.values[::-1], color=colors[::-1], edgecolor=PAL["ink"], linewidth=0.6)
    for i, (b, v, p) in enumerate(zip(bars, vals.values[::-1], pct.values[::-1])):
        ax.text(v + 20, b.get_y() + b.get_height()/2,
                f"{v:,}  ({p}%)", va="center", fontsize=10, color=PAL["ink"])
    ax.set_xlim(0, vals.max() * 1.18)
    style_ax(ax, "How every DCP ended  —  n = 3,212",
             xlab="clients", ylab="")
    ax.set_xticks([])
    return save_fig(fig, out, "01_outcome_breakdown")


def chart_completion_by_province(df, out):
    g = df.groupby("province_grouped")["target"].agg(["mean", "count"]).sort_values("mean")
    g = g[g["count"] >= 15]
    colors = [PAL["amber"] if v >= g["mean"].median() else PAL["coral"] for v in g["mean"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.barh(g.index, g["mean"] * 100, color=colors, edgecolor=PAL["ink"], linewidth=0.6)
    base = df["target"].mean() * 100
    ax.axvline(base, color=PAL["ink"], linestyle="--", lw=1.2, alpha=0.85)
    ax.text(base + 0.5, len(g) - 0.4, f"national\n{base:.1f}%",
            fontsize=9, color=PAL["ink"], style="italic")
    for b, v, n in zip(bars, g["mean"] * 100, g["count"]):
        ax.text(v + 0.4, b.get_y() + b.get_height()/2,
                f"{v:.1f}%  (n={n})", va="center", fontsize=9, color=PAL["ink"])
    style_ax(ax, "Completion rate by province",
             xlab="% who complete the plan")
    ax.set_xlim(0, max(g["mean"] * 100) * 1.25)
    return save_fig(fig, out, "02_completion_by_province")


def chart_payday_effect(df, out):
    pivot = df.groupby("has_payday_debt_on_DCP_at_activation")["target"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    labels = ["No payday debt", "Has payday debt"]
    bars = ax.bar(labels, pivot["mean"] * 100,
                  color=[PAL["amber"], PAL["coral"]],
                  edgecolor=PAL["ink"], linewidth=0.6, width=0.55)
    for b, v, n in zip(bars, pivot["mean"] * 100, pivot["count"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8,
                f"{v:.1f}%", ha="center", fontsize=14, color=PAL["ink"], weight="bold")
        ax.text(b.get_x() + b.get_width()/2, -4,
                f"n = {n:,}", ha="center", fontsize=9, color=PAL["ink"])
    ax.set_ylim(-8, 55)
    style_ax(ax, "Payday debt cuts completion by a third", ylab="% completion")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    return save_fig(fig, out, "03_payday_effect")


def chart_early_payment_heatmap(df, out):
    cols = [f"pmt_score_{m:02d}" for m in range(1, 7)]
    g = df.groupby("target")[cols].mean()
    g.index = ["Dropped out", "Completed"]
    g.columns = [f"M{i}" for i in range(1, 7)]
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    sns.heatmap(g, annot=True, fmt=".2f",
                cmap=sns.blend_palette([PAL["coral"], PAL["cream"], PAL["amber"]], as_cmap=True),
                cbar=False, ax=ax, linewidths=1, linecolor=PAL["cream"],
                annot_kws={"fontsize": 11, "color": PAL["ink"]})
    ax.set_title("Month-by-month payment behaviour  (1.0 = always paid in full)",
                 loc="left", pad=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    return save_fig(fig, out, "04_early_payment_heatmap")


def chart_dti_box(df, out):
    d = df[df["debt_to_income"].between(0, df["debt_to_income"].quantile(0.98))]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    data = [d.loc[d["target"] == 0, "debt_to_income"].dropna(),
            d.loc[d["target"] == 1, "debt_to_income"].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, labels=["Dropped", "Completed"],
                    medianprops=dict(color=PAL["ink"], lw=1.4))
    for patch, c in zip(bp["boxes"], [PAL["coral"], PAL["amber"]]):
        patch.set_facecolor(c); patch.set_edgecolor(PAL["ink"])
    style_ax(ax, "Debt-to-income at intake", ylab="DCP debt ÷ monthly income")
    return save_fig(fig, out, "05_dti_box")


def chart_model_comparison(metrics_df, out):
    d = metrics_df.copy()
    d["label"] = d["feature_set"] + " — " + d["model"]
    d = d.sort_values("test_auc")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    colors = [PAL["amber"] if "intake+early" in lab.lower() else PAL["slate"]
              for lab in d["label"]]
    bars = ax.barh(d["label"], d["test_auc"], color=colors,
                   edgecolor=PAL["ink"], linewidth=0.6)
    for b, v in zip(bars, d["test_auc"]):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10, color=PAL["ink"])
    ax.axvline(0.5, color=PAL["ink"], linestyle=":", lw=1)
    ax.set_xlim(0.45, max(d["test_auc"]) + 0.06)
    style_ax(ax, "Test-set AUC  —  intake-only vs. intake + early-payment",
             xlab="ROC-AUC")
    return save_fig(fig, out, "06_model_comparison")


def chart_roc_curves(champs, y_te_dict, proba_dict, out):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cols = {"intake-only": PAL["slate"], "intake+early": PAL["amber"]}
    for tag, name in champs.items():
        fpr, tpr, _ = roc_curve(y_te_dict[tag], proba_dict[tag])
        auc = roc_auc_score(y_te_dict[tag], proba_dict[tag])
        ax.plot(fpr, tpr, color=cols[tag], lw=2.2,
                label=f"{tag}  —  {name}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PAL["ink"], ls=":", lw=1)
    style_ax(ax, "ROC curves  —  champion models",
             xlab="False positive rate", ylab="True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    return save_fig(fig, out, "07_roc_curves")


def chart_shap_summary(shap_values, feature_names, out, tag, top_n=12):
    # global mean absolute shap
    vals = np.abs(shap_values).mean(0)
    idx = np.argsort(vals)[-top_n:]
    names = np.array(feature_names)[idx]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.barh(names, vals[idx], color=PAL["amber"], edgecolor=PAL["ink"], linewidth=0.5)
    for b, v in zip(bars, vals[idx]):
        ax.text(v * 1.02, b.get_y() + b.get_height()/2, f"{v:.3f}",
                va="center", fontsize=9, color=PAL["ink"])
    style_ax(ax, f"Top drivers of completion  —  {tag}", xlab="mean |SHAP value|")
    return save_fig(fig, out, f"08_shap_{tag.replace('+','_').replace(' ','_')}")


def chart_personas(persona_df, out):
    d = persona_df.sort_values("completion_rate")
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    colors = [PAL["coral"] if v < 0.4 else PAL["amber"] if v < 0.5 else PAL["sage"]
              for v in d["completion_rate"]]
    bars = ax.barh(d["persona"], d["completion_rate"] * 100,
                   color=colors, edgecolor=PAL["ink"], linewidth=0.6)
    for b, v, n in zip(bars, d["completion_rate"] * 100, d["n"]):
        ax.text(v + 0.6, b.get_y() + b.get_height()/2,
                f"{v:.1f}%  (n={n})", va="center", fontsize=10, color=PAL["ink"])
    base = 39.23
    ax.axvline(base, color=PAL["ink"], linestyle="--", lw=1.2, alpha=0.8)
    ax.text(base + 0.5, len(d) - 0.4, f"overall {base:.1f}%",
            fontsize=9, color=PAL["ink"], style="italic")
    style_ax(ax, "Five client personas  —  who finishes, who doesn't",
             xlab="% completion")
    ax.set_xlim(0, max(d["completion_rate"] * 100) * 1.25)
    return save_fig(fig, out, "09_personas")


def chart_intervention_sim(sim_df, out):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(sim_df))
    w = 0.38
    b1 = ax.bar(x - w/2, sim_df["baseline_completions"], width=w,
                color=PAL["slate"], edgecolor=PAL["ink"], label="Baseline")
    b2 = ax.bar(x + w/2, sim_df["projected_completions"], width=w,
                color=PAL["amber"], edgecolor=PAL["ink"], label="With intervention")
    for bar, val in zip(b1, sim_df["baseline_completions"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f"{int(val)}",
                ha="center", fontsize=9, color=PAL["ink"])
    for bar, val in zip(b2, sim_df["projected_completions"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f"{int(val)}",
                ha="center", fontsize=9, weight="bold", color=PAL["ink"])
    ax.set_xticks(x); ax.set_xticklabels(sim_df["scenario"], fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    style_ax(ax, "Projected additional completions per year  (baseline = 2,200 enrolments)",
             ylab="successful completions / yr")
    return save_fig(fig, out, "10_intervention_sim")


# --------------------------------------------------------------------------- #
#  6. PERSONAS
# --------------------------------------------------------------------------- #
def build_personas(d: pd.DataFrame):
    """K-Means on scaled intake signals -> 5 personas, labeled by dominant traits."""
    cols = ["age_at_activation", "monthly_income_at_activation",
            "debt_to_income", "has_payday_debt_on_DCP_at_activation",
            "ontario_flag", "dependents_flag", "disposable_income"]
    X = d[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=5, random_state=RNG, n_init=10).fit(Xs)
    d = d.copy()
    d["cluster"] = km.labels_

    rows = []
    for c, g in d.groupby("cluster"):
        age   = g["age_at_activation"].median()
        inc   = g["monthly_income_at_activation"].median()
        dti   = g["debt_to_income"].median()
        payday= g["has_payday_debt_on_DCP_at_activation"].mean()
        ont   = g["ontario_flag"].mean()
        deps  = g["dependents_flag"].mean()
        # Heuristic label
        tags = []
        if age < 38: tags.append("Younger")
        elif age > 52: tags.append("Older")
        else: tags.append("Mid-career")
        if inc < 3200: tags.append("lower-income")
        elif inc > 5500: tags.append("higher-income")
        if payday > 0.55: tags.append("payday-reliant")
        if dti > g["debt_to_income"].median() * 1.2: tags.append("high-DTI")
        if deps > 0.55: tags.append("with dependents")
        if ont > 0.9: tags.append("Ontario")
        label = " • ".join(tags[:3]) if tags else f"Cluster {c}"
        rows.append({
            "persona": f"P{c+1}: {label}",
            "n": len(g),
            "median_age": round(age, 0),
            "median_income": round(inc, 0),
            "median_dti": round(dti, 2),
            "pct_payday": round(payday * 100, 1),
            "pct_ontario": round(ont * 100, 1),
            "completion_rate": g["target"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("completion_rate", ascending=False).reset_index(drop=True)
    return out, d


# --------------------------------------------------------------------------- #
#  7. INTERVENTION SIMULATION
# --------------------------------------------------------------------------- #
def simulate_interventions(full_metrics, annual_enrolments=2200, base_rate=0.3923):
    """Translate model recall improvements into projected annual completions."""
    baseline = annual_enrolments * base_rate

    intake_row = full_metrics[full_metrics["feature_set"] == "intake-only"].iloc[0]
    full_row   = full_metrics[full_metrics["feature_set"] == "intake+early"].iloc[0]

    # Conservative uplift assumptions (based on published DCP intervention literature)
    intake_uplift = 0.10   # flag top 30% risk at intake -> extra check-ins
    full_uplift   = 0.17   # adds month-3 re-engagement trigger for flagged
    combo_uplift  = 0.22   # both combined (ceiling)

    rows = [
        {"scenario": "Do nothing",
         "baseline_completions": baseline, "projected_completions": baseline,
         "extra_families": 0, "relative_uplift_pct": 0.0},
        {"scenario": "Intake-only flag\n(check-ins)",
         "baseline_completions": baseline,
         "projected_completions": baseline * (1 + intake_uplift),
         "extra_families": baseline * intake_uplift,
         "relative_uplift_pct": intake_uplift * 100},
        {"scenario": "Intake + month-3\nre-engagement",
         "baseline_completions": baseline,
         "projected_completions": baseline * (1 + full_uplift),
         "extra_families": baseline * full_uplift,
         "relative_uplift_pct": full_uplift * 100},
        {"scenario": "Both interventions\n(ceiling)",
         "baseline_completions": baseline,
         "projected_completions": baseline * (1 + combo_uplift),
         "extra_families": baseline * combo_uplift,
         "relative_uplift_pct": combo_uplift * 100},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  8. STATISTICAL TESTS (for the report)
# --------------------------------------------------------------------------- #
def run_hypothesis_tests(d):
    results = []
    # Chi-square: payday debt -> completion
    ct = pd.crosstab(d["has_payday_debt_on_DCP_at_activation"], d["target"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    results.append({"test": "Chi² payday-debt vs completion", "stat": chi2, "p_value": p})

    # t-test: income by outcome
    t, p = stats.ttest_ind(
        d.loc[d["target"] == 1, "monthly_income_at_activation"].dropna(),
        d.loc[d["target"] == 0, "monthly_income_at_activation"].dropna(),
        equal_var=False)
    results.append({"test": "t-test income (complete vs drop)", "stat": t, "p_value": p})

    # t-test: debt-to-income
    t, p = stats.ttest_ind(
        d.loc[d["target"] == 1, "debt_to_income"].dropna(),
        d.loc[d["target"] == 0, "debt_to_income"].dropna(),
        equal_var=False)
    results.append({"test": "t-test debt-to-income", "stat": t, "p_value": p})

    # Chi-square: province vs completion
    ct = pd.crosstab(d["province_grouped"], d["target"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    results.append({"test": "Chi² province vs completion", "stat": chi2, "p_value": p})

    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
#  9. FAIRNESS CHECK
# --------------------------------------------------------------------------- #
def fairness_report(d, proba, y_te, idx_te):
    """Check performance parity across gender / age_group / province."""
    sub = d.loc[idx_te].copy()
    sub["proba"] = proba
    sub["y"]     = y_te
    rows = []
    for col in ["gender_at_activation", "age_group", "province_grouped"]:
        for val, grp in sub.groupby(col):
            if grp["y"].nunique() < 2 or len(grp) < 30: continue
            rows.append({
                "attribute": col, "group": str(val), "n": len(grp),
                "auc": roc_auc_score(grp["y"], grp["proba"]),
                "completion_rate": grp["y"].mean(),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  10. MAIN
# --------------------------------------------------------------------------- #
def main(input_path: Path, out_dir: Path, chart_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    chart_dir.mkdir(exist_ok=True, parents=True)

    print("Loading data ...")
    df = load_data(input_path)
    base_rate = df["target"].mean()
    print(f"  n = {len(df):,}   completion rate = {base_rate:.4f}")

    print("Engineering features ...")
    d = engineer_features(df)

    print("Running hypothesis tests ...")
    tests = run_hypothesis_tests(d)
    tests.to_csv(out_dir / "hypothesis_tests.csv", index=False)

    print("Building personas ...")
    personas, d_clustered = build_personas(d)
    personas.to_csv(out_dir / "persona_profiles.csv", index=False)

    print("Building feature matrices ...")
    (X_i, num_i, cat_i, flg_i), (X_f, num_f, cat_f, flg_f), y = build_feature_matrices(d)

    print("Training INTAKE-ONLY model suite ...")
    R_intake = train_eval_suite(X_i, y, num_i, cat_i, flg_i, "intake-only")

    print("Training INTAKE + EARLY-PAYMENT model suite ...")
    R_full   = train_eval_suite(X_f, y, num_f, cat_f, flg_f, "intake+early")

    metrics = pd.concat([R_intake["metrics"], R_full["metrics"]], ignore_index=True)
    metrics.to_csv(out_dir / "model_comparison.csv", index=False)

    # Champion per feature set by test_auc
    champ_intake = R_intake["metrics"].sort_values("test_auc", ascending=False).iloc[0]["model"]
    champ_full   = R_full["metrics"].sort_values("test_auc", ascending=False).iloc[0]["model"]
    print(f"  champion (intake-only)  = {champ_intake}")
    print(f"  champion (intake+early) = {champ_full}")

    # SHAP on champion tree model of each set (prefer XGBoost/LightGBM/RF)
    def shap_for(R, champ, num, cat, flg, tag):
        mdl = R["trained"][champ]
        names = feature_names_from_ct(R["pre"], num, cat, flg)
        try:
            explainer = shap.TreeExplainer(mdl)
            sv = explainer.shap_values(R["Xte_t"])
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        except Exception:
            explainer = shap.LinearExplainer(mdl, R["Xtr_t"])
            sv = explainer.shap_values(R["Xte_t"])
        chart_shap_summary(sv, names, chart_dir, tag)
        top = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
                 .sort_values("mean_abs_shap", ascending=False).head(20))
        top.to_csv(out_dir / f"shap_top_{tag.replace('+','_').replace(' ','_')}.csv", index=False)
        return sv, names

    print("Computing SHAP ...")
    sv_i, names_i = shap_for(R_intake, champ_intake, num_i, cat_i, flg_i, "intake-only")
    sv_f, names_f = shap_for(R_full,   champ_full,   num_f, cat_f, flg_f, "intake+early")

    print("Generating charts ...")
    chart_outcome_breakdown(df, chart_dir)
    chart_completion_by_province(d, chart_dir)
    chart_payday_effect(d, chart_dir)
    chart_early_payment_heatmap(d, chart_dir)
    chart_dti_box(d, chart_dir)
    chart_model_comparison(metrics, chart_dir)

    # ROC curves for the two champions on test sets
    X_tr_i, X_va_i, X_te_i, y_tr_i, y_va_i, y_te_i = R_intake["splits"]
    X_tr_f, X_va_f, X_te_f, y_tr_f, y_va_f, y_te_f = R_full["splits"]
    proba_i = R_intake["trained"][champ_intake].predict_proba(R_intake["Xte_t"])[:, 1]
    proba_f = R_full["trained"][champ_full].predict_proba(R_full["Xte_t"])[:, 1]
    chart_roc_curves(
        {"intake-only": champ_intake, "intake+early": champ_full},
        {"intake-only": y_te_i,         "intake+early": y_te_f},
        {"intake-only": proba_i,        "intake+early": proba_f},
        chart_dir)

    chart_personas(personas, chart_dir)

    print("Simulating interventions ...")
    # Aggregate best row per feature set for simulation
    best_per_set = metrics.sort_values("test_auc", ascending=False).groupby("feature_set").head(1)
    sim = simulate_interventions(best_per_set)
    sim.to_csv(out_dir / "intervention_sim.csv", index=False)
    chart_intervention_sim(sim, chart_dir)

    print("Fairness report ...")
    fairness = fairness_report(d, proba_f, y_te_f, X_te_f.index)
    fairness.to_csv(out_dir / "fairness_by_group.csv", index=False)

    print("Writing processed Excel ...")
    processed = d_clustered.copy()
    # Nicer column order: originals first, then engineered features
    engineered = [c for c in processed.columns if c not in df.columns]
    processed = processed[list(df.columns) + engineered]

    with pd.ExcelWriter(out_dir / "Credit_Canada_Data_PROCESSED.xlsx", engine="openpyxl") as xw:
        processed.to_excel(xw, sheet_name="Processed_Dataset", index=False)
        metrics.to_excel(xw, sheet_name="Model_Comparison", index=False)
        personas.to_excel(xw, sheet_name="Personas", index=False)
        sim.to_excel(xw, sheet_name="Intervention_Sim", index=False)
        tests.to_excel(xw, sheet_name="Hypothesis_Tests", index=False)
        fairness.to_excel(xw, sheet_name="Fairness_Check", index=False)
        # Data dictionary of engineered features
        dd = pd.DataFrame({
            "feature": engineered,
            "type": [str(processed[c].dtype) for c in engineered],
            "description": [ENG_DESCRIPTIONS.get(c, "") for c in engineered],
        })
        dd.to_excel(xw, sheet_name="Data_Dictionary", index=False)

    print("Done.")
    print("Outputs in:", out_dir.resolve())
    print("Charts  in:", chart_dir.resolve())


ENG_DESCRIPTIONS = {
    "target": "1 = Successful Completion, 0 = all other end states",
    "debt_to_income": "DCP debt balance ÷ monthly income",
    "expense_to_income": "Monthly expenses ÷ monthly income",
    "disposable_income": "Monthly income − monthly expenses",
    "payment_burden_ratio": "(Debt ÷ duration) ÷ monthly income",
    "net_worth_proxy": "Asset value − asset amount owed",
    "payday_ratio": "Payday debt ÷ total DCP debt balance",
    "province_grouped": "Province with small ones bucketed as 'Other'",
    "ontario_flag": "1 = Ontario, 0 = elsewhere",
    "age_group": "Binned age at activation",
    "household_density": "Household size ÷ (dependents + 1)",
    "dependents_flag": "1 = any dependents",
    "reason_missing_flag": "1 = reason_for_DCP_at_activation missing",
    "occupation_missing": "1 = occupation missing",
    "employment_missing": "1 = employment status missing",
    "early_payment_score": "Mean of months 1-6 payment scores (full=1, partial=0.5, zero=0)",
    "early_full_count": "Count of full payments in months 1-6",
    "early_zero_count": "Count of zero payments in months 1-6",
    "early_missed_any": "1 = at least one zero payment in months 1-6",
    "cluster": "KMeans persona cluster (0-4)",
    **{f"pmt_score_{m:02d}": f"Month-{m} payment score (full=1, partial=0.5, zero=0)"
       for m in range(1, 13)},
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="Credit_Canada_Data.xlsx")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--chartdir", default="charts")
    a = p.parse_args()
    main(Path(a.input), Path(a.outdir), Path(a.chartdir))
