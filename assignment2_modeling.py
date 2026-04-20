"""
Assignment 2: Model Exploration & Results
AI-Based Prediction of Student Dropout and Academic Success in Higher Education
Samsung Innovation Campus — AI Course
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# OUTPUT DIRECTORY — all new files go here
# ─────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {name}")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading Dataset")
print("="*60)

df = pd.read_csv(os.path.join(OUT_DIR, "data111.csv"), sep=";")
# Strip whitespace from column names (tab/space artifacts)
df.columns = [c.strip() for c in df.columns]
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Target distribution:\n{df['Target'].value_counts()}")

# ─────────────────────────────────────────────
# 2. DATA PREPARATION FOR MODELING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Data Preparation for Modeling")
print("="*60)

# 2a. Encode target
le = LabelEncoder()
df["Target_enc"] = le.fit_transform(df["Target"])
class_names = list(le.classes_)
print(f"  Classes: {class_names}  →  {list(range(len(class_names)))}")

# 2b. Separate features and target
X = df.drop(columns=["Target", "Target_enc"])
y = df["Target_enc"]

# 2c. Feature selection — keep top features identified in EDA
#     (all 36 numerical features are kept; correlated weak features remain
#      as tree models handle them naturally; LR/MLP benefit from StandardScaler)
print(f"  Features used: {X.shape[1]}")

# 2d. Train / test split  (80% train, 20% test — stratified)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train set: {X_train_raw.shape[0]} samples")
print(f"  Test  set: {X_test_raw.shape[0]} samples")

# 2e. Normalise / Standardise
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# 2f. SMOTE — address class imbalance on TRAINING set only
print("  Applying SMOTE to training set ...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"  After SMOTE — train size: {X_train_sm.shape[0]}")
print(f"  Class dist after SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ─────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Model Training & Evaluation")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu",
        max_iter=300, random_state=42
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training: {name}")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Cross-val F1
    cv_f1 = cross_val_score(model, X_train_sm, y_train_sm,
                             cv=cv, scoring="f1_macro", n_jobs=-1)

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "f1"       : f1,
        "cv_f1_mean": cv_f1.mean(),
        "cv_f1_std" : cv_f1.std(),
    }
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"    CV F1    : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ─────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Generating Visualisations")
print("="*60)

palette = {
    "Logistic Regression" : "#4C72B0",
    "Random Forest"        : "#55A868",
    "Gradient Boosting"    : "#C44E52",
    "Neural Network (MLP)" : "#8172B2",
}

# ── Figure A2-1: Confusion Matrices (2x2 grid) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold", y=1.01)

for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="grey"
    )
    ax.set_title(f"{name}\n(Acc={res['accuracy']:.3f}  F1={res['f1']:.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)

plt.tight_layout()
save_fig("a2_01_confusion_matrices.png")

# ── Figure A2-2: ROC Curves (one figure per model, 2x2 layout) ──────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("ROC Curves (One-vs-Rest) — All Models", fontsize=16, fontweight="bold", y=1.01)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_colors = ["#E74C3C", "#2ECC71", "#3498DB"]

for ax, (name, res) in zip(axes.flatten(), results.items()):
    model = res["model"]
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)
    else:
        from sklearn.calibration import CalibratedClassifierCV
        y_score = model.predict_proba(X_test_scaled)

    auc_scores = []
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=roc_colors[i], lw=2,
                label=f"{cls} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    mean_auc = np.mean(auc_scores)
    ax.set_title(f"{name}\n(Mean AUC = {mean_auc:.3f})", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig("a2_02_roc_curves.png")

# ── Figure A2-3: Model Comparison Bar Chart ──────────────────────────────────
metrics = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)"]
model_names = list(results.keys())
short_names = ["Log. Reg.", "Rnd. Forest", "Grad. Boost", "MLP"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

for ax, metric, label in zip(axes, metrics, metric_labels):
    values = [results[m][metric] for m in model_names]
    bars = ax.bar(short_names, values,
                  color=[palette[m] for m in model_names],
                  edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticklabels(short_names, fontsize=9)

plt.tight_layout()
save_fig("a2_03_model_comparison.png")

# ── Figure A2-4: Cross-Validation F1 (with error bars) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
cv_means = [results[m]["cv_f1_mean"] for m in model_names]
cv_stds  = [results[m]["cv_f1_std"]  for m in model_names]
colors   = [palette[m] for m in model_names]

bars = ax.bar(short_names, cv_means, yerr=cv_stds,
              color=colors, edgecolor="white", linewidth=0.8,
              capsize=7, width=0.5, error_kw={"elinewidth": 2, "ecolor": "black"})
ax.set_ylim(0, 1.05)
ax.set_title("5-Fold Cross-Validation F1-Score (Macro)\nwith Standard Deviation", fontsize=13, fontweight="bold")
ax.set_ylabel("CV F1-Score (Macro)")
ax.grid(axis="y", alpha=0.3)
for bar, val, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
save_fig("a2_04_cross_validation.png")

# ── Figure A2-5: Feature Importance (Random Forest + XGBoost) ───────────────
feature_names = list(X.columns)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Top 15 Feature Importances", fontsize=15, fontweight="bold")

for ax, model_key, color in [
    (axes[0], "Random Forest",     "#55A868"),
    (axes[1], "Gradient Boosting", "#C44E52")
]:
    model = results[model_key]["model"]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    bars = ax.barh(top_features[::-1], top_values[::-1],
                   color=color, edgecolor="white", linewidth=0.5, alpha=0.88)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"{model_key}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
save_fig("a2_05_feature_importance.png")

# ── Figure A2-6: Per-Class F1 Heatmap ────────────────────────────────────────
per_class_f1 = {}
for name, res in results.items():
    report = classification_report(y_test, res["y_pred"],
                                   target_names=class_names, output_dict=True)
    per_class_f1[name] = {cls: report[cls]["f1-score"] for cls in class_names}

df_heatmap = pd.DataFrame(per_class_f1).T
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="YlGn",
            ax=ax, vmin=0, vmax=1, linewidths=0.5,
            annot_kws={"size": 12, "weight": "bold"})
ax.set_title("Per-Class F1-Score Across Models", fontsize=13, fontweight="bold")
ax.set_xlabel("Student Outcome Class", fontsize=11)
ax.set_ylabel("Model", fontsize=11)
plt.tight_layout()
save_fig("a2_06_per_class_f1_heatmap.png")

# ─────────────────────────────────────────────
# 5. PRINT FULL CLASSIFICATION REPORTS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Detailed Classification Reports")
print("="*60)
for name, res in results.items():
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(classification_report(y_test, res["y_pred"], target_names=class_names))

# ─────────────────────────────────────────────
# 6. RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Summary Table")
print("="*60)
summary = []
for name, res in results.items():
    summary.append({
        "Model"     : name,
        "Accuracy"  : f"{res['accuracy']:.4f}",
        "Precision" : f"{res['precision']:.4f}",
        "Recall"    : f"{res['recall']:.4f}",
        "F1-Score"  : f"{res['f1']:.4f}",
        "CV F1 (±)" : f"{res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}",
    })
df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

# Save summary CSV
csv_path = os.path.join(OUT_DIR, "a2_model_results_summary.csv")
df_summary.to_csv(csv_path, index=False)
print(f"\n  [saved] a2_model_results_summary.csv")

print("\n" + "="*60)
print("ALL DONE — All outputs saved with prefix 'a2_'")
print("="*60)
