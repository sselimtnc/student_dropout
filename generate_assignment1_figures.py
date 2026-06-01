"""
Generate Assignment 1 EDA figures using the UPV Longitudinal Student Dataset (dataset_2022_hash.csv).
Outputs: a1_01_target_distribution.png, a1_02_academic_performance.png,
         a1_03_demographic_financial.png, a1_04_dropout_correlation.png.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, "..", "student_dropout-main", "dataset_2022_hash.csv")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=";", decimal=",", low_memory=False)

colors = {"A": "#e74c3c", "B": "#2ecc71"} # A = Abandoned (Red), B = Continuing (Green)
labels = {"A": "A (Abandoned)", "B": "B (Continuing)"}

# 1. Target Distribution
print("Generating Target Distribution plot...")
fig, ax = plt.subplots(figsize=(8, 5))
vc = df["abandono_hash"].value_counts()
ax.bar(vc.index, vc.values, color=[colors[x] for x in vc.index], edgecolor="white")
ax.set_title("Assignment 1 — Student Outcome Distribution (UPV)", fontweight="bold")
ax.set_ylabel("Count")
ax.set_xlabel("Outcome (A = Abandoned, B = Continuing)")
for i, (lab, v) in enumerate(vc.items()):
    ax.text(i, v + 2000, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a1_01_target_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. Academic Performance by Outcome
print("Generating Academic Performance histograms...")
academic = [
    "nota_asig_hash",
    "nota14_hash",
    "nota10_hash",
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, academic):
    for lab in colors:
        data = df[df["abandono_hash"] == lab][col].dropna()
        ax.hist(data, bins=25, alpha=0.55, label=labels[lab], color=colors[lab])
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
fig.suptitle("Assignment 1 — Academic Performance by Outcome", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a1_02_academic_performance.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3. Demographics and Financial Factors
print("Generating Financial cross-tabulations...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Tuition fee payment issue (impagado_curso_mat = 1 means unpaid fee)
ct_fees = pd.crosstab(df["impagado_curso_mat"], df["abandono_hash"])
ct_fees.plot(kind="bar", ax=axes[0], color=[colors[c] for c in ["A", "B"]])
axes[0].set_title("Unpaid Course Fee (impagado_curso_mat) vs Outcome", fontweight="bold")
axes[0].set_xlabel("Unpaid Course Fee (0 = Paid, 1 = Unpaid)")
axes[0].legend(title="Outcome", labels=["A (Abandoned)", "B (Continuing)"])

# Displaced student status (desplazado_hash = A / B)
ct_displaced = pd.crosstab(df["desplazado_hash"], df["abandono_hash"])
ct_displaced.plot(kind="bar", ax=axes[1], color=[colors[c] for c in ["A", "B"]])
axes[1].set_title("Displaced Status vs Outcome", fontweight="bold")
axes[1].set_xlabel("Displaced Status (A or B)")
axes[1].legend(title="Outcome", labels=["A (Abandoned)", "B (Continuing)"])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a1_03_demographic_financial.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4. Correlation with abandonment
print("Generating Correlation horizontal bar chart...")
df_tmp = df.copy()
df_tmp["is_dropout"] = (df_tmp["abandono_hash"] == "A").astype(int)

# Identify numerical features, dropping leakage columns and ID columns
ID_COLS = ["dni_hash", "tit_hash", "asi_hash", "campus_hash", "grupos_por_tipocredito_hash", "baja_fecha", "fecha_datos"]
leakage_cols = ["matricula_activa", "rendimiento_total", "rend_total_ultimo", "rend_total_penultimo", "rend_total_antepenultimo"]
LMS_PREFIXES = ("pft_", "resource_", "n_wifi_")
drop_cols = ID_COLS + leakage_cols + [c for c in df_tmp.columns if c.startswith(LMS_PREFIXES)]

df_tmp = df_tmp.drop(columns=[c for c in drop_cols if c in df_tmp.columns])

# Encode objects to get numeric values for correlation
for col in df_tmp.select_dtypes(include=["object", "string"]).columns:
    if col != "abandono_hash":
        df_tmp[col] = LabelEncoder = pd.factorize(df_tmp[col])[0]

num_cols = df_tmp.select_dtypes(include=[np.number]).columns
corr = df_tmp[num_cols].corr()["is_dropout"].drop("is_dropout").sort_values(key=abs, ascending=False).head(12)

fig, ax = plt.subplots(figsize=(10, 6))
bar_c = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr.values]
ax.barh(corr.index[::-1], corr.values[::-1], color=bar_c[::-1])
ax.set_title("Assignment 1 — Top Features Correlated with Trajectory Abandonment", fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a1_04_dropout_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()

print("All Assignment 1 figures generated successfully!")
