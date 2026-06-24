"""
Employee Attrition Prediction — Classification Model
=====================================================
Dataset: IBM HR Analytics Employee Attrition (public Kaggle dataset)
This script generates realistic synthetic data matching the real dataset's
distributions so the code is fully runnable. Swap with the real CSV from:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Skills demonstrated: Data cleaning, EDA, feature engineering, 
Logistic Regression + Random Forest, evaluation metrics, feature importance.

Author: Mamadu Jalloh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── 1. GENERATE REALISTIC IBM-STYLE HR DATA ───────────────────────────────────
def generate_hr_data(n=1470):
    dept = np.random.choice(["Sales","R&D","HR"], n, p=[0.30,0.56,0.14])
    job_level = np.random.choice([1,2,3,4,5], n, p=[0.25,0.28,0.22,0.15,0.10])
    monthly_income = (job_level * 2200 + 
                      np.random.normal(0, 800, n)).clip(1000, 20000).astype(int)
    years_at_co = np.random.exponential(7, n).clip(0, 40).astype(int)
    overtime = np.random.choice([0, 1], n, p=[0.72, 0.28])
    satisfaction = np.random.choice([1,2,3,4], n, p=[0.12,0.18,0.33,0.37])
    work_life_bal = np.random.choice([1,2,3,4], n, p=[0.05,0.22,0.61,0.12])
    distance = np.random.exponential(9, n).clip(1, 29).astype(int)
    age = np.random.normal(36, 9, n).clip(18, 60).astype(int)
    num_co_worked = np.random.choice(range(1, 10), n)
    env_satisfaction = np.random.choice([1,2,3,4], n, p=[0.10,0.22,0.36,0.32])

    # Attrition probability: logistic with realistic factors
    log_odds = (
        -2.5
        + 0.9  * overtime
        - 0.6  * (satisfaction - 2.5)
        - 0.5  * (work_life_bal - 2.5)
        + 0.04 * distance
        - 0.15 * (monthly_income / 1000)
        + 0.4  * (job_level == 1).astype(float)
        - 0.03 * years_at_co
        + 0.1  * num_co_worked
        - 0.5  * (env_satisfaction - 2.5)
        + np.random.normal(0, 0.3, n)
    )
    prob_attr = 1 / (1 + np.exp(-log_odds))
    attrition = (np.random.rand(n) < prob_attr).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Department": dept,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_co,
        "OverTime": overtime,
        "JobSatisfaction": satisfaction,
        "WorkLifeBalance": work_life_bal,
        "DistanceFromHome": distance,
        "NumCompaniesWorked": num_co_worked,
        "EnvironmentSatisfaction": env_satisfaction,
        "Attrition": attrition
    })

df = generate_hr_data()

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print(f"Dataset shape   : {df.shape}")
print(f"Attrition rate  : {df['Attrition'].mean()*100:.1f}%")
print(f"Missing values  : {df.isnull().sum().sum()}")
print("\nClass distribution:")
print(df["Attrition"].value_counts())

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
le = LabelEncoder()
df["Department_enc"] = le.fit_transform(df["Department"])
df["IncomePerYear"] = df["MonthlyIncome"] * 12
df["TenureLevel"] = pd.cut(df["YearsAtCompany"],
                            bins=[-1, 2, 5, 10, 40],
                            labels=["<2yrs","2-5yrs","5-10yrs","10+yrs"])
df["TenureLevel_enc"] = LabelEncoder().fit_transform(df["TenureLevel"])

features = ["Age","JobLevel","MonthlyIncome","YearsAtCompany","OverTime",
            "JobSatisfaction","WorkLifeBalance","DistanceFromHome",
            "NumCompaniesWorked","EnvironmentSatisfaction","Department_enc",
            "TenureLevel_enc"]

X = df[features]
y = df["Attrition"]

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. MODELS ─────────────────────────────────────────────────────────────────
lr  = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
rf  = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                              max_depth=8, random_state=42, n_jobs=-1)

lr.fit(X_train_sc, y_train)
rf.fit(X_train, y_train)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv  = cross_val_score(lr, X_train_sc, y_train, cv=cv, scoring="roc_auc").mean()
rf_cv  = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

# Test metrics
y_pred_lr = lr.predict(X_test_sc)
y_pred_rf = rf.predict(X_test)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, y_pred_lr, target_names=["Stay","Leave"]))
print(f"AUC: {auc_lr:.3f} | CV-AUC: {lr_cv:.3f}")

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=["Stay","Leave"]))
print(f"AUC: {auc_rf:.3f} | CV-AUC: {rf_cv:.3f}")

# Feature importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

# ── 6. PLOT ───────────────────────────────────────────────────────────────────
DARK  = "#1A1D1B"
PAPER = "#EFEAE0"
GREEN = "#6FFFB0"
AMBER = "#E8A23D"
BLUE  = "#5BA4CF"
RED   = "#E85D3D"
MUTED = "#8A8678"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "text.color": PAPER, "axes.labelcolor": PAPER,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#2E332F", "grid.color": "#2E332F",
    "grid.linestyle": "--", "grid.alpha": 0.4,
    "font.family": "monospace", "figure.dpi": 130,
})

fig = plt.figure(figsize=(16, 14))
fig.suptitle("EMPLOYEE ATTRITION PREDICTION · ML CLASSIFICATION REPORT",
             fontsize=13, fontweight="bold", color=PAPER, y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.36)

# — Panel 1: Attrition by dept & overtime ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
dept_attr = df.groupby("Department")["Attrition"].mean() * 100
bars = ax1.bar(dept_attr.index, dept_attr.values,
               color=[AMBER, GREEN, BLUE], alpha=0.85, width=0.5)
for bar, val in zip(bars, dept_attr.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=9, color=PAPER)
ax1.set_title("ATTRITION RATE BY DEPARTMENT", fontsize=9, color=MUTED, loc="left", pad=8)
ax1.set_ylabel("Attrition rate (%)")
ax1.grid(True, axis="y")

# — Panel 2: Income distribution by attrition ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for label, color, name in [(0, GREEN, "Stayed"), (1, RED, "Left")]:
    ax2.hist(df.loc[df.Attrition==label, "MonthlyIncome"],
             bins=25, alpha=0.55, color=color, label=name, density=True)
ax2.set_title("MONTHLY INCOME DISTRIBUTION BY ATTRITION", fontsize=9, color=MUTED, loc="left", pad=8)
ax2.set_xlabel("Monthly Income (USD)")
ax2.set_ylabel("Density")
ax2.legend(fontsize=8, framealpha=0.1)
ax2.grid(True)

# — Panel 3: ROC curves ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for y_prob, name, color, auc in [
    (y_prob_lr, "Logistic Regression", AMBER, auc_lr),
    (y_prob_rf, "Random Forest", GREEN, auc_rf)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
ax3.plot([0,1],[0,1], color=MUTED, linestyle="--", lw=1, label="Random baseline")
ax3.set_title("ROC CURVES — MODEL COMPARISON", fontsize=9, color=MUTED, loc="left", pad=8)
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(fontsize=8, framealpha=0.1)
ax3.grid(True)

# — Panel 4: Feature importance ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
colors_fi = [GREEN if v == fi.values.max() else AMBER if v > fi.values.mean() else MUTED
             for v in fi.values]
ax4.barh(fi.index, fi.values, color=colors_fi, alpha=0.85)
ax4.set_title("FEATURE IMPORTANCE (RANDOM FOREST)", fontsize=9, color=MUTED, loc="left", pad=8)
ax4.set_xlabel("Importance")
ax4.grid(True, axis="x")

# — Panel 5: Confusion matrix RF ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm, display_labels=["Stay","Leave"])
disp.plot(ax=ax5, colorbar=False, cmap="Greens")
ax5.set_title("CONFUSION MATRIX · RANDOM FOREST", fontsize=9, color=MUTED, loc="left", pad=8)
ax5.tick_params(colors=PAPER)

# — Panel 6: Overtime vs Satisfaction attrition heatmap ───────────────────────
ax6 = fig.add_subplot(gs[2, 1])
pivot = df.groupby(["JobSatisfaction","OverTime"])["Attrition"].mean().unstack() * 100
im = ax6.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=60)
ax6.set_xticks([0,1])
ax6.set_xticklabels(["No Overtime","Overtime"])
ax6.set_yticks(range(4))
ax6.set_yticklabels(["Low (1)","Fair (2)","Good (3)","High (4)"])
ax6.set_xlabel("Overtime")
ax6.set_ylabel("Job Satisfaction")
for i in range(4):
    for j in range(2):
        ax6.text(j, i, f"{pivot.values[i,j]:.1f}%",
                 ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, ax=ax6, label="Attrition rate (%)")
ax6.set_title("ATTRITION RATE: SATISFACTION vs OVERTIME", fontsize=9, color=MUTED, loc="left", pad=8)

plt.savefig("/home/claude/projects/ml-model/attrition_analysis.png",
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("\n✓ Chart saved")

print(f"""
╔══════════════════════════════════════════════════════════╗
║        KEY FINDINGS — ATTRITION ML MODEL               ║
╠══════════════════════════════════════════════════════════╣
║ Dataset attrition rate   : {df['Attrition'].mean()*100:.1f}%                       ║
║ Best model               : Random Forest                ║
║ Random Forest AUC        : {auc_rf:.3f}                          ║
║ Logistic Regression AUC  : {auc_lr:.3f}                          ║
║ CV-AUC (RF, 5-fold)      : {rf_cv:.3f}                          ║
║ Top predictor            : {fi.idxmax()}            ║
║ Overtime attrition rate  : {df[df.OverTime==1]['Attrition'].mean()*100:.1f}% vs {df[df.OverTime==0]['Attrition'].mean()*100:.1f}% (no OT)         ║
╚══════════════════════════════════════════════════════════╝
""")
