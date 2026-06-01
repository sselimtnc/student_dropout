"""
AI-Based Prediction of Student Trajectory Abandonment
Exploratory Data Analysis (EDA) on UPV longitudinal dataset
=========================================================================
Dataset: UPV Longitudinal Student Dropout (dataset_2022_hash.csv)
SDG Alignment: SDG 4 - Quality Education
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUTPUT_DIR, "..", "student_dropout-main", "dataset_2022_hash.csv")

# ============================================================
# 1. DATA LOADING & INITIAL INSPECTION
# ============================================================
print("=" * 70)
print("1. DATA LOADING & INITIAL INSPECTION")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=";", decimal=",", low_memory=False)

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Samples: {df.shape[0]:,}")
print(f"Number of Features: {df.shape[1] - 1} + 1 Target")

TARGET_COL = "abandono_hash"
colors = {'A': '#e74c3c', 'B': '#2ecc71'} # A = Abandoned, B = Continuing
labels = {'A': 'A (Abandoned)', 'B': 'B (Continuing)'}

# ============================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("2. TARGET VARIABLE ANALYSIS")
print("=" * 70)

target_counts = df[TARGET_COL].value_counts()
target_pcts = df[TARGET_COL].value_counts(normalize=True) * 100
print("\nTarget Distribution:")
for label, count in target_counts.items():
    print(f"  {label}: {count:,} ({target_pcts[label]:.1f}%)")

# Plot Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar Chart
color_list = [colors[label] for label in target_counts.index]
bars = axes[0].bar(target_counts.index.astype(str), target_counts.values, color=color_list, edgecolor='white', linewidth=2)
axes[0].set_title('Distribution of Student Outcomes', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Student Trajectory Status')
axes[0].set_ylabel('Count')
for bar, count, pct in zip(bars, target_counts.values, target_pcts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Pie Chart
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
           colors=color_list, startangle=90, explode=(0.05, 0.05),
           shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Proportion of Student Outcomes', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_target_distribution.png")

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 70)

# Save descriptive stats to CSV
# Drop ID/hash columns first
hash_cols = ["dni_hash", "tit_hash", "asi_hash", "campus_hash", "grupos_por_tipocredito_hash", "baja_fecha", "fecha_datos"]
desc_df = df.drop(columns=[c for c in hash_cols if c in df.columns])
# Try to convert to numeric where possible
for col in desc_df.columns:
    if col != TARGET_COL:
        desc_df[col] = pd.to_numeric(desc_df[col], errors='coerce')

desc_stats = desc_df.describe()
desc_stats.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_statistics.csv'))
print("\n✅ Saved: descriptive_statistics.csv")

# ============================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("4. MISSING VALUES ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

print(f"\nTotal columns with missing values: {len(missing_df)}")
print(f"Total missing cells: {df.isnull().sum().sum():,}")
print("\nTop 15 Columns with most missing values:")
print(missing_df.head(15))

# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("5. CORRELATION ANALYSIS")
print("=" * 70)

# Select key numerical columns to keep matrix readable
key_numerical = [
    'nota_asig_hash', 'nota14_hash', 'nota10_hash',
    'cred_mat_total', 'cred_sup_total', 'cred_pend_sup_tit',
    'rendimiento_cuat_b', 'asig1', 'anyo_ingreso', 'impagado_curso_mat'
]
key_numerical = [c for c in key_numerical if c in df.columns]

# Correlation matrix heatmap using pure matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[key_numerical].corr()

# Draw heatmap
cax = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
fig.colorbar(cax)

# Set ticks
ax.set_xticks(np.arange(len(key_numerical)))
ax.set_yticks(np.arange(len(key_numerical)))
ax.set_xticklabels(key_numerical, rotation=45, ha='right')
ax.set_yticklabels(key_numerical)

# Write correlation values on heatmap cells
for i in range(len(key_numerical)):
    for j in range(len(key_numerical)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white",
                fontsize=8, fontweight="bold")

ax.set_title('Correlation Matrix of Key Numerical Features', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_matrix.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 02_correlation_matrix.png")

# ============================================================
# 6. ACADEMIC PERFORMANCE ANALYSIS (Key Feature)
# ============================================================
print("\n" + "=" * 70)
print("6. ACADEMIC PERFORMANCE ANALYSIS")
print("=" * 70)

academic_cols = [
    'nota_asig_hash',
    'nota14_hash',
    'cred_mat_total',
    'cred_sup_total',
    'rendimiento_cuat_b'
]
academic_cols = [c for c in academic_cols if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(academic_cols):
    ax = axes[idx]
    for label in colors:
        data = df[df[TARGET_COL] == label][col].dropna()
        ax.hist(data, bins=30, alpha=0.6, color=colors[label], label=labels[label], edgecolor='white')
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)

# Remove extra subplots
for i in range(len(academic_cols), 6):
    axes[i].axis('off')

fig.suptitle('Academic Performance Distribution by Student Outcome', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_academic_performance.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 03_academic_performance.png")

# ============================================================
# 7. DEMOGRAPHIC ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("7. DEMOGRAPHIC ANALYSIS (Admissions and Displaced)")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# University admission year
ax = axes[0, 0]
for label in colors:
    data = df[df[TARGET_COL] == label]['anyo_ingreso'].dropna()
    ax.hist(data, bins=15, alpha=0.6, color=colors[label], label=labels[label], edgecolor='white')
ax.set_title('Admission Year by Outcome', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Frequency')
ax.legend()

# Admission type
ax = axes[0, 1]
ct_type = pd.crosstab(df['tipo_ingreso'], df[TARGET_COL])
# Plot top 6 types
top_types = ct_type.sum(axis=1).nlargest(6).index
ct_type_subset = ct_type.loc[top_types]
ct_type_subset.plot(kind='bar', ax=ax, color=[colors[c] for c in ct_type_subset.columns], edgecolor='white')
ax.set_title('Admission Type (Top 6) by Outcome', fontweight='bold')
ax.set_xlabel('Admission Type Code')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])

# Displaced student status
ax = axes[1, 0]
ct_disp = pd.crosstab(df['desplazado_hash'], df[TARGET_COL])
ct_disp.plot(kind='bar', ax=ax, color=[colors[c] for c in ct_disp.columns], edgecolor='white')
ax.set_title('Displaced Status by Outcome', fontweight='bold')
ax.set_xlabel('Displaced Status (A or B)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])

# Student dedication (Full-time TC vs Part-time TP)
ax = axes[1, 1]
ct_ded = pd.crosstab(df['dedicacion'], df[TARGET_COL])
ct_ded.plot(kind='bar', ax=ax, color=[colors[c] for c in ct_ded.columns], edgecolor='white')
ax.set_title('Study Dedication by Outcome', fontweight='bold')
ax.set_xlabel('Dedication (TC = Full Time, TP = Part Time)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])

fig.suptitle('Demographic & Enrollment Factors Analysis', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_demographic_analysis.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 04_demographic_analysis.png")

# ============================================================
# 8. FINANCIAL & SOCIOECONOMIC FACTORS
# ============================================================
print("\n" + "=" * 70)
print("8. FINANCIAL & SOCIOECONOMIC FACTORS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Unpaid fees (impagado_curso_mat = 1 means unpaid)
ax = axes[0]
fees_target = pd.crosstab(df['impagado_curso_mat'], df[TARGET_COL])
fees_target.plot(kind='bar', ax=ax, color=[colors[c] for c in fees_target.columns], edgecolor='white')
ax.set_title('Unpaid Fee Status by Outcome', fontweight='bold')
ax.set_xlabel('Unpaid Fee (0: Paid, 1: Unpaid)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])

# Mother's education level
ax = axes[1]
ct_m_edu = pd.crosstab(df['estudios_m_hash'], df[TARGET_COL])
# Plot top 5 education categories
top_m_edu = ct_m_edu.sum(axis=1).nlargest(5).index
ct_m_edu_subset = ct_m_edu.loc[top_m_edu]
ct_m_edu_subset.plot(kind='bar', ax=ax, color=[colors[c] for c in ct_m_edu_subset.columns], edgecolor='white')
ax.set_title("Mother's Education (Top 5) by Outcome", fontweight="bold")
ax.set_xlabel("Education Category Hash")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])

# Father's education level
ax = axes[2]
ct_p_edu = pd.crosstab(df['estudios_p_hash'], df[TARGET_COL])
# Plot top 5 education categories
top_p_edu = ct_p_edu.sum(axis=1).nlargest(5).index
ct_p_edu_subset = ct_p_edu.loc[top_p_edu]
ct_p_edu_subset.plot(kind='bar', ax=ax, color=[colors[c] for c in ct_p_edu_subset.columns], edgecolor='white')
ax.set_title("Father's Education (Top 5) by Outcome", fontweight="bold")
ax.set_xlabel("Education Category Hash")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])

fig.suptitle('Socioeconomic & Financial Factors', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_financial_factors.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 05_financial_factors.png")

# ============================================================
# 9. BOX PLOTS FOR KEY FEATURES
# ============================================================
print("\n" + "=" * 70)
print("9. BOX PLOTS - KEY FEATURES BY OUTCOME")
print("=" * 70)

key_features = [
    'nota_asig_hash',
    'nota14_hash',
    'cred_mat_total',
    'cred_sup_total',
    'cred_pend_sup_tit',
    'rendimiento_cuat_b'
]
key_features = [c for c in key_features if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(key_features):
    ax = axes[idx]
    
    # Custom matplotlib boxplot grouped by targets
    target_groups = [df[df[TARGET_COL] == 'A'][col].dropna().values,
                     df[df[TARGET_COL] == 'B'][col].dropna().values]
    
    box = ax.boxplot(target_groups, labels=['A (Abandoned)', 'B (Continuing)'],
                     patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    
    # Fill colors
    for patch, label in zip(box['boxes'], ['A', 'B']):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.6)
        
    ax.set_title(col, fontweight='bold', fontsize=11)

fig.suptitle('Key Features Distribution by Student Outcome (Box Plots)', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_boxplots.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 06_boxplots.png")

# ============================================================
# 10. COURSE-WISE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("10. COURSE-WISE DROPOUT ANALYSIS")
print("=" * 70)

# Note: In the hash representation, asi_hash is the course/subject identifier
course_analysis = pd.crosstab(df['asi_hash'], df[TARGET_COL])
course_analysis['Total'] = course_analysis.sum(axis=1)
course_analysis['Dropout_Rate'] = (course_analysis.get('A', 0) / course_analysis['Total'] * 100).round(1)
course_analysis = course_analysis.sort_values('Dropout_Rate', ascending=False)

# Plot top 10 courses by dropout rate
fig, ax = plt.subplots(figsize=(14, 6))
top_courses = course_analysis[course_analysis['Total'] >= 50].head(10) # Minimum 50 samples
bars = ax.barh(np.arange(len(top_courses)), top_courses['Dropout_Rate'],
              color=plt.cm.Reds(np.linspace(0.8, 0.3, len(top_courses))), edgecolor='white')
ax.set_yticks(np.arange(len(top_courses)))
ax.set_yticklabels([f"Subject {idx[:8]}..." for idx in top_courses.index])
ax.set_xlabel('Abandonment Rate (%)')
ax.set_title('Top 10 Subjects by Abandonment Rate (n >= 50)', fontweight='bold', fontsize=14)
for i, (rate, total) in enumerate(zip(top_courses['Dropout_Rate'], top_courses['Total'])):
    ax.text(rate + 0.5, i, f'{rate}% (n={total})', va='center', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_course_dropout.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 07_course_dropout.png")

# ============================================================
# 11. SEMESTER PERFORMANCE COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("11. SEMESTER PERFORMANCE COMPARISON")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sem A vs Sem B Enrolled
ax = axes[0]
for label in colors:
    subset = df[df[TARGET_COL] == label]
    ax.scatter(subset['cred_mat_sem_a'],
              subset['cred_mat_sem_b'],
              alpha=0.1, color=colors[label], label=labels[label], s=15)
ax.set_xlabel('Semester A Enrolled Credits')
ax.set_ylabel('Semester B Enrolled Credits')
ax.set_title('Enrolled Credits: Sem A vs Sem B', fontweight='bold')
ax.legend()

# Sem A vs Sem B Completed
ax = axes[1]
for label in colors:
    subset = df[df[TARGET_COL] == label]
    ax.scatter(subset['cred_sup_sem_a'],
              subset['cred_sup_sem_b'],
              alpha=0.1, color=colors[label], label=labels[label], s=15)
ax.set_xlabel('Semester A Completed Credits')
ax.set_ylabel('Semester B Completed Credits')
ax.set_title('Completed Credits: Sem A vs Sem B', fontweight='bold')
ax.legend()

fig.suptitle('Semester Enrollment & Completion Comparison', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_semester_comparison.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 08_semester_comparison.png")

# ============================================================
# 12. PREFERENCE SELECTION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("12. PREFERENCE SELECTION ANALYSIS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 5))
pref_target = pd.crosstab(df['preferencia_seleccion'], df[TARGET_COL])
# Plot top 6 preferences
ct_pref_subset = pref_target.head(6)
ct_pref_subset.plot(kind='bar', ax=ax, color=[colors[c] for c in ct_pref_subset.columns], edgecolor='white')
ax.set_title('Outcome Distribution by Admission Preference Rank', fontweight='bold', fontsize=14)
ax.set_xlabel('Selection Preference Rank (1 = First Choice)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome', labels=["A (Abandoned)", "B (Continuing)"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_marital_status.png'), bbox_inches='tight') # Override old marital_status.png name
plt.close()
print("\n✅ Saved: 09_marital_status.png")

# ============================================================
# 13. FEATURE IMPORTANCE PREVIEW - POINT-BISERIAL CORRELATION WITH TARGET
# ============================================================
print("\n" + "=" * 70)
print("13. FEATURE IMPORTANCE (Correlation with Abandonment)")
print("=" * 70)

# Create binary dropout indicator
df['is_dropout'] = (df[TARGET_COL] == 'A').astype(int)

# Select key numeric features (excluding ID/temporal leakage)
leakage_cols = ["matricula_activa", "rendimiento_total", "rend_total_ultimo", "rend_total_penultimo", "rend_total_antepenultimo"]
LMS_PREFIXES = ("pft_", "resource_", "n_wifi_")
all_leakage = hash_cols + leakage_cols + [c for c in df.columns if c.startswith(LMS_PREFIXES)]
num_corr_cols = [c for c in df.columns if c not in all_leakage and c != TARGET_COL and c != 'is_dropout']


# Encode objects to get numeric correlations
df_corr_tmp = df[num_corr_cols + ['is_dropout']].copy()
for col in df_corr_tmp.select_dtypes(include=["object", "string"]).columns:
    df_corr_tmp[col] = pd.factorize(df_corr_tmp[col])[0]

dropout_corr = df_corr_tmp.corr()['is_dropout'].drop('is_dropout').sort_values(key=abs, ascending=False)

print("\nTop Features Correlated with Trajectory Abandonment:")
for feat, corr_val in dropout_corr.head(15).items():
    direction = "↑ more abandonment" if corr_val > 0 else "↓ less abandonment"
    print(f"  {feat}: {corr_val:+.4f} ({direction})")

# Plot feature importance
fig, ax = plt.subplots(figsize=(14, 8))
top_features = dropout_corr.head(15)
colors_bar = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_features.values]
ax.barh(np.arange(len(top_features)), top_features.values, color=colors_bar, edgecolor='white')
ax.set_yticks(np.arange(len(top_features)))
ax.set_yticklabels(top_features.index)
ax.set_xlabel('Correlation with Abandonment')
ax.set_title('Top 15 Features Correlated with Student Trajectory Abandonment', fontweight='bold', fontsize=14)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.invert_yaxis()

# Add legend manually
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='↑ Higher value → More Abandonment'),
                   Patch(facecolor='#2ecc71', label='↑ Higher value → Less Abandonment')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_feature_importance.png'), bbox_inches='tight')
plt.close()
print("\n✅ Saved: 10_feature_importance.png")

# Clean up temp column
df.drop('is_dropout', axis=1, inplace=True)

# ============================================================
# 14. PREPROCESSING SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("14. DATA PREPROCESSING SUMMARY")
print("=" * 70)

print(f"""
📊 Dataset Overview:
   - Total Samples: {len(df):,}
   - Total Features: {df.shape[1] - 1}
   - Target Classes: 2 (A, B)
   
🎯 Class Distribution:
   - B (Continuing): {target_counts.get('B', 0):,} ({target_pcts.get('B', 0):.1f}%)
   - A (Abandoned): {target_counts.get('A', 0):,} ({target_pcts.get('A', 0):.1f}%)

✅ Missing Values Analyzed.
📈 Numerical Features: {len(num_corr_cols)}
""")

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nGenerated Files:")
print(f"  📊 01_target_distribution.png")
print(f"  📊 02_correlation_matrix.png")
print(f"  📊 03_academic_performance.png")
print(f"  📊 04_demographic_analysis.png")
print(f"  📊 05_financial_factors.png")
print(f"  📊 06_boxplots.png")
print(f"  📊 07_course_dropout.png")
print(f"  📊 08_semester_comparison.png")
print(f"  📊 09_marital_status.png")
print(f"  📊 10_feature_importance.png")
print(f"  📋 descriptive_statistics.csv")
