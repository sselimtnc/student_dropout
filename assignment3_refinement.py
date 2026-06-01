"""
Assignment 3: Model Refinement, Test Submission & Deployment
UPV Longitudinal Student Dropout Dataset (MDPI Data 2025)
Samsung Innovation Campus — AI Course

METHODOLOGICAL CORRECTION:
1. SMOTE + StandardScaler run strictly INSIDE imblearn.Pipeline per fold.
2. Group-based splitting on student ID (dni_hash) using StratifiedGroupKFold
   and GroupShuffleSplit to prevent student identity leakage across training
   and validation/test sets.
3. Pipelines are trained on raw features, so the scaler is saved inside the
   pipeline, simplifying deployment to only the 20 selected features.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, "..", "student_dropout-main", "dataset_2022_hash.csv")

print(f"Working Directory: {OUT_DIR}")
print(f"Dataset Path: {DATA_PATH}")

# 1. LOAD DATA
print("\n" + "="*60)
print("1. Loading Dataset")
print("="*60)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please make sure dataset_2022_hash.csv is in student_dropout-main.")

df = pd.read_csv(DATA_PATH, sep=";", decimal=",", low_memory=False)
print(f"Full Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Target and ID Columns
TARGET_COL = "abandono_hash"
ID_COLS = [
    "dni_hash", "tit_hash", "asi_hash", "campus_hash",
    "grupos_por_tipocredito_hash", "baja_fecha", "fecha_datos",
]

# Leakage columns to drop
LMS_PREFIXES = ("pft_", "resource_", "n_wifi_")
TARGET_LEAKAGE_COLS = [
    "matricula_activa",
    "rendimiento_total",
    "rend_total_ultimo",
    "rend_total_penultimo",
    "rend_total_antepenultimo",
]

# Separate raw features, target, and group (student ID)
print("\nCleaning features and encoding target...")
y_raw = df[TARGET_COL].astype(str).str.strip()
le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
class_names = [f"{c} ({'abandoned' if c == 'A' else 'continuing'})" for c in le.classes_]
print(f"Target classes: {class_names} -> [0, 1]")

# Extract groups (student ID) and convert to string
groups_all = df["dni_hash"].astype(str)

# Drop ID, target, and leakage columns (including dni_hash) from X_raw
drop_cols = [TARGET_COL] + [c for c in ID_COLS if c in df.columns]
X_raw = df.drop(columns=drop_cols)

lms_cols = [c for c in X_raw.columns if c.startswith(LMS_PREFIXES)]
X_raw = X_raw.drop(columns=lms_cols)
print(f"  Dropped {len(lms_cols)} LMS/Wi-Fi monthly log columns (temporal leakage prevention)")

leak_cols = [c for c in TARGET_LEAKAGE_COLS if c in X_raw.columns]
X_raw = X_raw.drop(columns=leak_cols)
print(f"  Dropped {len(leak_cols)} target leakage columns: {leak_cols}")

# Label encode object columns
for col in X_raw.select_dtypes(include=["object", "string"]).columns:
    X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

# Handle NaNs
X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

print(f"Remaining feature count for selection: {X_raw.shape[1]}")

# 2. GROUP-BASED SAMPLING & SPLIT
print("\n" + "="*60)
print("2. Group-Based Sampling & Split (Prevent Identity Leakage)")
print("="*60)

# Get unique students and their labels
# (All records for the same student have the same target value)
student_df = pd.DataFrame({
    "dni_hash": groups_all,
    "target": y_enc
})
unique_students = student_df.groupby("dni_hash")["target"].first()
print(f"Total unique students in full dataset: {len(unique_students):,}")

# Subsample students to keep runtime reasonable (Grid Search on ~60k rows)
MAX_STUDENTS = 8_000
if len(unique_students) > MAX_STUDENTS:
    print(f"Subsampling {MAX_STUDENTS:,} unique students...")
    unique_students = unique_students.sample(n=MAX_STUDENTS, random_state=42)

# Perform stratified train-test split at the student level
train_students, test_students = train_test_split(
    unique_students.index,
    test_size=0.20,
    random_state=42,
    stratify=unique_students.values
)

print(f"Unique students in Train: {len(train_students):,}")
print(f"Unique students in Test: {len(test_students):,}")

# Filter the original dataset rows using these student splits
train_mask = groups_all.isin(train_students)
test_mask = groups_all.isin(test_students)

X_train_raw = X_raw[train_mask]
y_train = y_enc[train_mask]
groups_train = groups_all[train_mask].values

X_test_raw = X_raw[test_mask]
y_test = y_enc[test_mask]
groups_test = groups_all[test_mask].values

print(f"Train set: {X_train_raw.shape[0]:,} samples (rows)")
print(f"Test set: {X_test_raw.shape[0]:,} samples (rows)")
print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# 3. FEATURE SELECTION (Using Random Forest Importances on temporary scaled data)
print("\n" + "="*60)
print("3. Feature Selection")
print("="*60)
scaler_temp = StandardScaler()
X_train_scaled_temp = scaler_temp.fit_transform(X_train_raw)

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_scaled_temp, y_train)

importances = rf_selector.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X_train_raw.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_k = 20
selected_features = feature_imp_df.head(top_k)['Feature'].tolist()
print(f"Selected Top {top_k} Features based on Random Forest Importance:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat:<40} (Importance: {feature_imp_df.iloc[i-1]['Importance']:.4f})")

# Subset datasets using RAW features (pipelines will handle scaling inside folds)
X_train_selected = X_train_raw[selected_features]
X_test_selected = X_test_raw[selected_features]

# 4. HYPERPARAMETER TUNING (GroupKFold + SMOTE inside CV)
print("\n" + "="*60)
print("4. Hyperparameter Tuning (Grid Search with imblearn Pipeline & Group CV)")
print("="*60)

# StratifiedGroupKFold ensures that the classes are balanced AND students don't leak across validation folds
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Helper function to build pipeline with scaler inside
def make_tuning_pipeline(estimator):
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', estimator)
    ])

# A. Random Forest Pipeline
print("Tuning Random Forest...")
rf_pipeline = make_tuning_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1))
rf_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5]
}
rf_search = GridSearchCV(rf_pipeline, rf_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
rf_search.fit(X_train_selected, y_train, groups=groups_train)
best_rf_pipe = rf_search.best_estimator_
print(f"  Best RF Params: {rf_search.best_params_}")
print(f"  Best RF CV F1: {rf_search.best_score_:.4f}")

# B. HistGradientBoosting Pipeline
print("\nTuning HistGradientBoosting...")
hgb_pipeline = make_tuning_pipeline(HistGradientBoostingClassifier(random_state=42))
hgb_grid = {
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_iter': [100, 200],
    'model__max_depth': [3, 5, 8],
    'model__l2_regularization': [0.0, 1.0, 10.0]
}
hgb_search = GridSearchCV(hgb_pipeline, hgb_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
hgb_search.fit(X_train_selected, y_train, groups=groups_train)
best_hgb_pipe = hgb_search.best_estimator_
print(f"  Best HGB Params: {hgb_search.best_params_}")
print(f"  Best HGB CV F1: {hgb_search.best_score_:.4f}")

# C. Multi-Layer Perceptron (MLP) Pipeline
print("\nTuning Multi-Layer Perceptron...")
mlp_pipeline = make_tuning_pipeline(MLPClassifier(max_iter=300, random_state=42))
mlp_grid = {
    'model__hidden_layer_sizes': [(64, 32), (32, 16)],
    'model__alpha': [0.1, 1.0, 5.0],
    'model__learning_rate_init': [0.001, 0.01]
}
mlp_search = GridSearchCV(mlp_pipeline, mlp_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
mlp_search.fit(X_train_selected, y_train, groups=groups_train)
best_mlp_pipe = mlp_search.best_estimator_
print(f"  Best MLP Params: {mlp_search.best_params_}")
print(f"  Best MLP CV F1: {mlp_search.best_score_:.4f}")

# D. Voting Ensemble
print("\nBuilding Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf_pipe),
        ('hgb', best_hgb_pipe),
        ('mlp', best_mlp_pipe)
    ],
    voting='soft'
)
voting_clf.fit(X_train_selected, y_train)

# 5. MODEL EVALUATION & COMPARISON
print("\n" + "="*60)
print("5. Model Evaluation on Held-out Test Set (Unseen Students)")
print("="*60)

refined_models = {
    "Tuned Random Forest": best_rf_pipe,
    "Tuned HistGradientBoosting": best_hgb_pipe,
    "Tuned Neural Network (MLP)": best_mlp_pipe,
    "Voting Ensemble": voting_clf
}

test_results = []

for name, model in refined_models.items():
    print(f"\nEvaluating: {name}")
    y_pred = model.predict(X_test_selected)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Predict probabilities for ROC AUC (One-vs-Rest)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_selected)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1], average="macro")
    else:
        roc_auc = np.nan
        
    cv_score = (
        rf_search.best_score_ if "Random Forest" in name else
        hgb_search.best_score_ if "HistGradientBoosting" in name else
        mlp_search.best_score_ if "Neural Network" in name else
        np.nan
    )
        
    test_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Macro)": prec,
        "Recall (Macro)": rec,
        "F1-Score (Macro)": f1,
        "ROC-AUC (Macro)": roc_auc,
        "CV F1 (No Leak)": cv_score
    })
    
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1-Score: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

df_results = pd.DataFrame(test_results)
print("\n" + "="*60)
print("Post-Refinement Performance Summary (Zero student leakage)")
print("="*60)
print(df_results.to_string(index=False))

# Save results summary CSV
results_csv_path = os.path.join(OUT_DIR, "a3_model_refinement_summary.csv")
df_results.to_csv(results_csv_path, index=False)
print(f"\nSaved summary to: {results_csv_path}")

# 5b. ABLATION STUDY (Parental Education Features)
print("\n" + "="*60)
print("5b. Socioeconomic (Parental Education) Feature Ablation Study")
print("="*60)

edu_features = ['estudios_p_hash', 'estudios_m_hash']
# Filter out parental education features if they are in the selected top 20
selected_no_edu = [f for f in selected_features if f not in edu_features]

print(f"Features in Full Model ({len(selected_features)}): {selected_features}")
print(f"Features in Ablated Model ({len(selected_no_edu)}): {selected_no_edu}")

# Train ablated RF pipeline
rf_ablated_pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        n_estimators=best_rf_pipe.named_steps['model'].n_estimators,
        max_depth=best_rf_pipe.named_steps['model'].max_depth,
        min_samples_split=best_rf_pipe.named_steps['model'].min_samples_split,
        random_state=42,
        n_jobs=-1
    ))
])
rf_ablated_pipe.fit(X_train_raw[selected_no_edu], y_train)
y_pred_ablated = rf_ablated_pipe.predict(X_test_raw[selected_no_edu])

acc_ablated = accuracy_score(y_test, y_pred_ablated)
f1_ablated = f1_score(y_test, y_pred_ablated, average='macro')

full_acc = df_results.loc[df_results['Model'] == 'Tuned Random Forest', 'Accuracy'].values[0]
full_f1 = df_results.loc[df_results['Model'] == 'Tuned Random Forest', 'F1-Score (Macro)'].values[0]

print(f"Full Model Accuracy: {full_acc:.4f}")
print(f"Full Model F1-Score (Macro): {full_f1:.4f}")
print(f"Ablated Model (No Parental Education) Accuracy: {acc_ablated:.4f}")
print(f"Ablated Model (No Parental Education) F1-Score (Macro): {f1_ablated:.4f}")

ablation_results = [
    {
        "Experiment": f"Full Model (Tuned RF, {len(selected_features)} features)",
        "Accuracy": full_acc,
        "F1-Score": full_f1
    },
    {
        "Experiment": f"Ablated Model (No Parental Education, {len(selected_no_edu)} features)",
        "Accuracy": acc_ablated,
        "F1-Score": f1_ablated
    }
]
df_ablation = pd.DataFrame(ablation_results)
ablation_csv_path = os.path.join(OUT_DIR, "a3_ablation_results.csv")
df_ablation.to_csv(ablation_csv_path, index=False)
print(f"Saved ablation results to: {ablation_csv_path}")

# 6. SERIALIZATION
print("\n" + "="*60)
print("6. Model Serialization")
print("="*60)
# Find the best model based on F1-Score
best_row = df_results.sort_values(by="F1-Score (Macro)", ascending=False).iloc[0]
best_model_name = best_row["Model"]
best_model = refined_models[best_model_name]
print(f"Best Model Selected for Deployment: {best_model_name} (F1-Score: {best_row['F1-Score (Macro)']:.4f})")

# We save the model (which has StandardScaler inside it), selected features list, class names
model_artifact = {
    "model": best_model,
    "model_name": best_model_name,
    "selected_features": selected_features,
    "class_names": class_names,
    "label_encoder": le
}

artifact_path = os.path.join(OUT_DIR, "best_model.joblib")
joblib.dump(model_artifact, artifact_path)
print(f"Successfully serialized best model artifact to: {artifact_path}")

print("\nREFINEMENT COMPLETED SUCCESSFULLY!")
