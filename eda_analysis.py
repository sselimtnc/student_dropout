"""
AI-Based Prediction of Student Dropout and Academic Success in Higher Education
Exploratory Data Analysis (EDA)
=========================================================================
Dataset: Predict Students' Dropout and Academic Success (UCI ML Repository)
SDG Alignment: SDG 4 - Quality Education
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
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
sns.set_style("whitegrid")
sns.set_palette("Set2")

OUTPUT_DIR = "/Users/selim/Desktop/ödev/"

# ============================================================
# 1. DATA LOADING & INITIAL INSPECTION
# ============================================================
print("=" * 70)
print("1. DATA LOADING & INITIAL INSPECTION")
print("=" * 70)

df = pd.read_csv("/Users/selim/Desktop/ödev/data111.csv", sep=";")

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Samples: {df.shape[0]}")
print(f"Number of Features: {df.shape[1] - 1} + 1 Target")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print(f"\n--- Data Types ---")
print(df.dtypes)

print(f"\n--- First 5 Rows ---")
print(df.head())

# ============================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("2. TARGET VARIABLE ANALYSIS")
print("=" * 70)

target_counts = df['Target'].value_counts()
target_pcts = df['Target'].value_counts(normalize=True) * 100
print("\nTarget Distribution:")
for label, count in target_counts.items():
    print(f"  {label}: {count} ({target_pcts[label]:.1f}%)")

# Plot Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar Chart
colors = {'Dropout': '#e74c3c', 'Graduate': '#2ecc71', 'Enrolled': '#3498db'}
color_list = [colors[label] for label in target_counts.index]
bars = axes[0].bar(target_counts.index, target_counts.values, color=color_list, edgecolor='white', linewidth=2)
axes[0].set_title('Distribution of Student Outcomes', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Student Status')
axes[0].set_ylabel('Count')
for bar, count, pct in zip(bars, target_counts.values, target_pcts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Pie Chart
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
           colors=color_list, startangle=90, explode=(0.05, 0.05, 0.05),
           shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Proportion of Student Outcomes', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}01_target_distribution.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_target_distribution.png")

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 70)

desc_stats = df.describe()
print("\nDescriptive Statistics (Numerical Features):")
print(desc_stats.to_string())

# Save descriptive stats to CSV
desc_stats.to_csv(f'{OUTPUT_DIR}descriptive_statistics.csv')
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
missing_df = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df) == 0:
    print("\n✅ No missing values found in the dataset!")
else:
    print("\nMissing Values:")
    print(missing_df)

print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("5. CORRELATION ANALYSIS")
print("=" * 70)

# Select numerical columns only
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Correlation matrix heatmap
fig, ax = plt.subplots(figsize=(22, 18))
corr_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
           vmin=-1, vmax=1, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix of All Numerical Features', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}02_correlation_matrix.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 02_correlation_matrix.png")

# Top correlations
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Feature 1': corr_matrix.columns[i],
            'Feature 2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })
corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
print("\nTop 15 Highest Correlations:")
print(corr_df.head(15).to_string(index=False))

# ============================================================
# 6. ACADEMIC PERFORMANCE ANALYSIS (Key Feature)
# ============================================================
print("\n" + "=" * 70)
print("6. ACADEMIC PERFORMANCE ANALYSIS")
print("=" * 70)

academic_cols = [
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Admission grade'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(academic_cols):
    ax = axes[idx]
    for label, color in colors.items():
        data = df[df['Target'] == label][col]
        ax.hist(data, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)

# Remove extra subplot
axes[5].axis('off')

fig.suptitle('Academic Performance Distribution by Student Outcome', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_academic_performance.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 03_academic_performance.png")

# Mean academic metrics by target
print("\nMean Academic Metrics by Student Outcome:")
print(df.groupby('Target')[academic_cols].mean().round(2).to_string())

# ============================================================
# 7. DEMOGRAPHIC ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("7. DEMOGRAPHIC ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Age Distribution
ax = axes[0, 0]
for label, color in colors.items():
    data = df[df['Target'] == label]['Age at enrollment']
    ax.hist(data, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_title('Age at Enrollment by Outcome', fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.legend()

# Gender Distribution
ax = axes[0, 1]
gender_target = pd.crosstab(df['Gender'], df['Target'])
gender_target.plot(kind='bar', ax=ax, color=[colors[c] for c in gender_target.columns], edgecolor='white')
ax.set_title('Gender Distribution by Outcome', fontweight='bold')
ax.set_xlabel('Gender (0: Female, 1: Male)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome')

# Scholarship
ax = axes[1, 0]
scholar_target = pd.crosstab(df['Scholarship holder'], df['Target'])
scholar_target.plot(kind='bar', ax=ax, color=[colors[c] for c in scholar_target.columns], edgecolor='white')
ax.set_title('Scholarship Status by Outcome', fontweight='bold')
ax.set_xlabel('Scholarship Holder (0: No, 1: Yes)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome')

# Tuition Fees
ax = axes[1, 1]
tuition_target = pd.crosstab(df['Tuition fees up to date'], df['Target'])
tuition_target.plot(kind='bar', ax=ax, color=[colors[c] for c in tuition_target.columns], edgecolor='white')
ax.set_title('Tuition Fees Status by Outcome', fontweight='bold')
ax.set_xlabel('Tuition Fees Up To Date (0: No, 1: Yes)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome')

fig.suptitle('Demographic & Financial Factors Analysis', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_demographic_analysis.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 04_demographic_analysis.png")

# Demographic statistics
print("\nAge Statistics by Outcome:")
print(df.groupby('Target')['Age at enrollment'].describe().round(2).to_string())

print("\nDropout Rate by Gender:")
gender_dropout = pd.crosstab(df['Gender'], df['Target'], normalize='index') * 100
print(gender_dropout.round(1).to_string())

print("\nDropout Rate by Scholarship Status:")
scholar_dropout = pd.crosstab(df['Scholarship holder'], df['Target'], normalize='index') * 100
print(scholar_dropout.round(1).to_string())

# ============================================================
# 8. FINANCIAL & SOCIOECONOMIC FACTORS
# ============================================================
print("\n" + "=" * 70)
print("8. FINANCIAL & SOCIOECONOMIC FACTORS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Debtor status
ax = axes[0]
debtor_target = pd.crosstab(df['Debtor'], df['Target'])
debtor_target.plot(kind='bar', ax=ax, color=[colors[c] for c in debtor_target.columns], edgecolor='white')
ax.set_title('Debtor Status by Outcome', fontweight='bold')
ax.set_xlabel('Debtor (0: No, 1: Yes)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome')

# Unemployment rate
ax = axes[1]
for label, color in colors.items():
    data = df[df['Target'] == label]['Unemployment rate']
    ax.hist(data, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_title('Unemployment Rate by Outcome', fontweight='bold')
ax.set_xlabel('Unemployment Rate (%)')
ax.legend()

# GDP
ax = axes[2]
for label, color in colors.items():
    data = df[df['Target'] == label]['GDP']
    ax.hist(data, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_title('GDP by Outcome', fontweight='bold')
ax.set_xlabel('GDP')
ax.legend()

fig.suptitle('Financial & Macroeconomic Factors', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}05_financial_factors.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 05_financial_factors.png")

print("\nDropout Rate by Debtor Status:")
debtor_dropout = pd.crosstab(df['Debtor'], df['Target'], normalize='index') * 100
print(debtor_dropout.round(1).to_string())

# ============================================================
# 9. BOX PLOTS FOR KEY FEATURES
# ============================================================
print("\n" + "=" * 70)
print("9. BOX PLOTS - KEY FEATURES BY OUTCOME")
print("=" * 70)

key_features = [
    'Admission grade',
    'Age at enrollment',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(key_features):
    ax = axes[idx]
    order = ['Dropout', 'Enrolled', 'Graduate']
    sns.boxplot(data=df, x='Target', y=col, ax=ax, order=order,
               palette=colors, showfliers=True)
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xlabel('')

fig.suptitle('Key Features Distribution by Student Outcome (Box Plots)', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_boxplots.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 06_boxplots.png")

# ============================================================
# 10. COURSE-WISE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("10. COURSE-WISE DROPOUT ANALYSIS")
print("=" * 70)

course_analysis = pd.crosstab(df['Course'], df['Target'])
course_analysis['Total'] = course_analysis.sum(axis=1)
course_analysis['Dropout_Rate'] = (course_analysis.get('Dropout', 0) / course_analysis['Total'] * 100).round(1)
course_analysis = course_analysis.sort_values('Dropout_Rate', ascending=False)

print("\nDropout Rate by Course (sorted):")
print(course_analysis[['Total', 'Dropout_Rate']].to_string())

# Plot top 10 courses by dropout rate
fig, ax = plt.subplots(figsize=(14, 6))
top_courses = course_analysis.head(10)
bars = ax.barh(range(len(top_courses)), top_courses['Dropout_Rate'],
              color=sns.color_palette("Reds_r", len(top_courses)), edgecolor='white')
ax.set_yticks(range(len(top_courses)))
ax.set_yticklabels([f'Course {idx}' for idx in top_courses.index])
ax.set_xlabel('Dropout Rate (%)')
ax.set_title('Top 10 Courses by Dropout Rate', fontweight='bold', fontsize=14)
for i, (rate, total) in enumerate(zip(top_courses['Dropout_Rate'], top_courses['Total'])):
    ax.text(rate + 0.5, i, f'{rate}% (n={total})', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}07_course_dropout.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 07_course_dropout.png")

# ============================================================
# 11. SEMESTER PERFORMANCE COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("11. SEMESTER PERFORMANCE COMPARISON")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sem 1 vs Sem 2 Approved
ax = axes[0]
for label, color in colors.items():
    subset = df[df['Target'] == label]
    ax.scatter(subset['Curricular units 1st sem (approved)'],
              subset['Curricular units 2nd sem (approved)'],
              alpha=0.3, color=color, label=label, s=15)
ax.set_xlabel('1st Semester Approved Units')
ax.set_ylabel('2nd Semester Approved Units')
ax.set_title('Approved Units: Sem 1 vs Sem 2', fontweight='bold')
ax.legend()

# Sem 1 vs Sem 2 Grades
ax = axes[1]
for label, color in colors.items():
    subset = df[df['Target'] == label]
    ax.scatter(subset['Curricular units 1st sem (grade)'],
              subset['Curricular units 2nd sem (grade)'],
              alpha=0.3, color=color, label=label, s=15)
ax.set_xlabel('1st Semester Grade')
ax.set_ylabel('2nd Semester Grade')
ax.set_title('Grades: Sem 1 vs Sem 2', fontweight='bold')
ax.legend()

fig.suptitle('Semester Performance Comparison', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}08_semester_comparison.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 08_semester_comparison.png")

# ============================================================
# 12. MARITAL STATUS ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("12. MARITAL STATUS ANALYSIS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 5))
marital_target = pd.crosstab(df['Marital status'], df['Target'], normalize='index') * 100
marital_target[['Graduate', 'Enrolled', 'Dropout']].plot(kind='bar', stacked=True, ax=ax,
    color=[colors['Graduate'], colors['Enrolled'], colors['Dropout']], edgecolor='white')
ax.set_title('Outcome Distribution by Marital Status', fontweight='bold', fontsize=14)
ax.set_xlabel('Marital Status')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}09_marital_status.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 09_marital_status.png")

# ============================================================
# 13. FEATURE IMPORTANCE PREVIEW - POINT-BISERIAL CORRELATION WITH TARGET
# ============================================================
print("\n" + "=" * 70)
print("13. FEATURE IMPORTANCE (Correlation with Dropout)")
print("=" * 70)

# Create binary dropout indicator
df['is_dropout'] = (df['Target'] == 'Dropout').astype(int)

# Calculate correlation with dropout
dropout_corr = df[numerical_cols + ['is_dropout']].corr()['is_dropout'].drop('is_dropout').sort_values(key=abs, ascending=False)

print("\nTop Features Correlated with Dropout:")
for feat, corr_val in dropout_corr.head(15).items():
    direction = "↑ more dropout" if corr_val > 0 else "↓ less dropout"
    print(f"  {feat}: {corr_val:+.4f} ({direction})")

# Plot feature importance
fig, ax = plt.subplots(figsize=(14, 8))
top_features = dropout_corr.head(15)
colors_bar = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_features.values]
ax.barh(range(len(top_features)), top_features.values, color=colors_bar, edgecolor='white')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features.index)
ax.set_xlabel('Correlation with Dropout')
ax.set_title('Top 15 Features Correlated with Student Dropout', fontweight='bold', fontsize=14)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='↑ Higher value → More Dropout'),
                   Patch(facecolor='#2ecc71', label='↑ Higher value → Less Dropout')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}10_feature_importance.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: 10_feature_importance.png")

# Clean up temp column
df.drop('is_dropout', axis=1, inplace=True)

# ============================================================
# 14. OUTLIER DETECTION
# ============================================================
print("\n" + "=" * 70)
print("14. OUTLIER DETECTION (IQR Method)")
print("=" * 70)

outlier_info = {}
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        outlier_info[col] = outliers

print("\nFeatures with Outliers:")
for feat, count in sorted(outlier_info.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {count} outliers ({count/len(df)*100:.1f}%)")

# ============================================================
# 15. DATA PREPROCESSING SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("15. DATA PREPROCESSING SUMMARY")
print("=" * 70)

print(f"""
📊 Dataset Overview:
   - Total Samples: {len(df)}
   - Total Features: {df.shape[1] - 1}
   - Target Classes: {df['Target'].nunique()} ({', '.join(df['Target'].unique())})
   
🎯 Class Distribution:
   - Graduate: {target_counts.get('Graduate', 0)} ({target_pcts.get('Graduate', 0):.1f}%)
   - Dropout: {target_counts.get('Dropout', 0)} ({target_pcts.get('Dropout', 0):.1f}%)
   - Enrolled: {target_counts.get('Enrolled', 0)} ({target_pcts.get('Enrolled', 0):.1f}%)

✅ Missing Values: {df.isnull().sum().sum()} (No missing values!)
⚠️  Features with Outliers: {len(outlier_info)}
📈 Numerical Features: {len(numerical_cols)}
🏷️  Categorical/Encoded Features: {df.shape[1] - len(numerical_cols) - 1}

🔑 Key Findings:
   1. The dataset is imbalanced (Graduate class is dominant)
   2. Academic performance (grades, approved units) is the strongest predictor
   3. Students who dropout tend to have lower 1st & 2nd semester performance
   4. Financial factors (tuition, debtor, scholarship) affect dropout rates
   5. Older students at enrollment tend to have higher dropout rates
   6. Macroeconomic indicators show some correlation with outcomes
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
