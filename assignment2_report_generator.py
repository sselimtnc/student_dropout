"""
Assignment 2 — Report Generator
Produces: Tuncer_Gungoren_assignment2.pdf
Based on the new UPV longitudinal dataset baseline modeling outcomes.
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(OUT_DIR, "Tuncer_Gungoren_assignment2.pdf")
SUMMARY_PATH = os.path.join(OUT_DIR, "a2_model_results_summary.csv")

# ── Page setup ───────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    leftMargin=1.0*cm, rightMargin=1.0*cm,
    topMargin=0.8*cm,  bottomMargin=0.8*cm,
)

W, H = A4
content_width = W - 2.0*cm

# ── Colour palette ───────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1A2D5A")
MID_BLUE   = colors.HexColor("#2E5FA3")
LIGHT_BLUE = colors.HexColor("#D6E4F7")
RED        = colors.HexColor("#C44E52")
GREEN      = colors.HexColor("#55A868")
GREY_BG    = colors.HexColor("#F4F6FA")
WHITE      = colors.white

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

s_title = ParagraphStyle("title",
    fontName="Helvetica-Bold", fontSize=11, textColor=WHITE,
    alignment=TA_CENTER, leading=13, spaceAfter=0)

s_sub = ParagraphStyle("sub",
    fontName="Helvetica", fontSize=7, textColor=colors.HexColor("#BFD3EF"),
    alignment=TA_CENTER, leading=8)

s_h2 = ParagraphStyle("h2",
    fontName="Helvetica-Bold", fontSize=8.5, textColor=DARK_BLUE,
    spaceBefore=2, spaceAfter=1, leading=9)

s_body = ParagraphStyle("body",
    fontName="Helvetica", fontSize=7.2, textColor=colors.HexColor("#2D2D2D"),
    alignment=TA_JUSTIFY, leading=9.5, spaceAfter=1)

s_bullet = ParagraphStyle("bullet",
    fontName="Helvetica", fontSize=7, textColor=colors.HexColor("#333333"),
    leftIndent=8, leading=9, spaceAfter=0.5)

s_table_hdr = ParagraphStyle("thdr",
    fontName="Helvetica-Bold", fontSize=6, textColor=WHITE, alignment=TA_CENTER)

s_table_cell = ParagraphStyle("tcell",
    fontName="Helvetica", fontSize=6, textColor=DARK_BLUE, alignment=TA_CENTER)

# ── Helper: section header ────────────────────────────────────────────────────
def section(title, body_paras):
    out = []
    out.append(HRFlowable(width=content_width, thickness=0.8,
                           color=MID_BLUE, spaceAfter=1))
    out.append(Paragraph(f"▪ {title}", s_h2))
    out.extend(body_paras)
    return out

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# ── Header banner ─────────────────────────────────────────────────────────────
banner_data = [[
    Paragraph("AI-Based Prediction of Student Trajectory Abandonment", s_title),
    Paragraph("Assignment 2 — Model Exploration &amp; Results on UPV Dataset", s_sub),
    Paragraph("Samsung Innovation Campus · Tuncer Güngören · April 2026", s_sub),
]]
banner_table = Table(banner_data, colWidths=[content_width])
banner_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), DARK_BLUE),
    ("TOPPADDING",    (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ("LEFTPADDING",   (0,0), (-1,-1), 2),
    ("RIGHTPADDING",  (0,0), (-1,-1), 2),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [DARK_BLUE]),
]))
story.append(banner_table)
story.append(Spacer(1, 2))

# ── 1. Model Selection & Justification ────────────────────────────────────────
story += section("1. Model Selection & Justification", [
    Paragraph("Four models were selected to evaluate the classification of student abandonment on the Spanish UPV longitudinal dataset, representing distinct learning frameworks:", s_body),
    Paragraph("• <b>Logistic Regression</b> — Transparent baseline model providing easily interpretable weights for educational coordinators.", s_bullet),
    Paragraph("• <b>Random Forest</b> — Non-linear tree ensemble capable of ranking feature importances and resolving complex credit dynamics.", s_bullet),
    Paragraph("• <b>Gradient Boosting</b> — Sequential boosting method optimized to detect subtle year-by-year patterns in academic records.", s_bullet),
    Paragraph("• <b>Neural Network (MLP)</b> — Multi-Layer Perceptron used to set a deep learning baseline for structured tabular records.", s_bullet),
])
story.append(Spacer(1, 2))

# ── 2. Data Preparation ───────────────────────────────────────────────────────
story += section("2. Data Preparation for Modelling", [
    Paragraph("The UPV longitudinal dataset (159,173 enrollment records, 169 columns) was preprocessed and modeled:", s_body),
    Paragraph("• <b>Data Subsampling:</b> Subsampled 60,000 course enrollment rows while preserving class balance (6.8% abandonment rate) for runtime feasibility.", s_bullet),
    Paragraph("• <b>Row-Based Split:</b> Stratified 80% train / 20% test split at the row level. Note: This introduces student identity leakage since courses for the same student are split across sets.", s_bullet),
    Paragraph("• <b>Pre-CV SMOTE &amp; Scaling:</b> StandardScaler and SMOTE were applied globally on the training set <i>before</i> cross-validation folds. This introduces synthetic-data leakage into validation folds.", s_bullet),
])
story.append(Spacer(1, 2))

# ── 3. Model Training & Evaluation ───────────────────────────────────────────
story += section("3. Model Training & Evaluation", [
    Paragraph("Models were trained on the globally SMOTE-resampled training set. Accuracy, Macro Precision, Macro Recall, and Macro F1 scores are evaluated on the held-out test set. 5-fold Cross-Validation F1 scores are obtained directly from the SMOTE-resampled training set:", s_body),
])

# Read summary CSV dynamically
if not os.path.exists(SUMMARY_PATH):
    raise FileNotFoundError("Ödev 2 model sonuçları bulunamadı. Lütfen önce assignment2_modeling.py scriptini çalıştırın.")

df_sum = pd.read_csv(SUMMARY_PATH)

hdr = ["Model", "Accuracy", "Precision", "Recall", "F1-Score (Macro)", "CV F1 ± Std"]
rows_data = []
for idx, r in df_sum.iterrows():
    rows_data.append([
        r["Model"],
        f"{float(r['Accuracy'])*100:.2f}%" if "%" not in str(r['Accuracy']) else r['Accuracy'],
        f"{float(r['Precision'])*100:.2f}%" if "%" not in str(r['Precision']) else r['Precision'],
        f"{float(r['Recall'])*100:.2f}%" if "%" not in str(r['Recall']) else r['Recall'],
        f"{float(r['F1-Score'])*100:.2f}%" if "%" not in str(r['F1-Score']) else r['F1-Score'],
        r["CV F1 (±)"]
    ])

table_data = [[Paragraph(h, s_table_hdr) for h in hdr]]
for i, row in enumerate(rows_data):
    style = s_table_cell
    # Mark Random Forest as best baseline
    is_best = ("Random Forest" in row[0])
    table_data.append([
        Paragraph(f"<b>{row[0]}</b>" if is_best else row[0], style),
        *[Paragraph(f"<b>{v}</b>" if is_best else v, style) for v in row[1:]]
    ])

t = Table(table_data, colWidths=[content_width * f for f in [0.28, 0.12, 0.12, 0.10, 0.20, 0.18]])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), MID_BLUE),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [GREY_BG, WHITE]),
    ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#D4EDDA")),
    ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CCCCCC")),
    ("TOPPADDING", (0,0), (-1,-1), 1),
    ("BOTTOMPADDING", (0,0), (-1,-1), 1),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(t)
story.append(Spacer(1, 2))

# ── Visualisations ────────────────────────────────────────────────────────────
def img(filename, w):
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        im = Image(path)
        aspect = im.imageHeight / float(im.imageWidth)
        # Force height compression
        return Image(path, width=w, height=w*aspect*0.95)
    return Spacer(1, 1)

fig_combo = [
    [img("a2_01_confusion_matrices.png", content_width*0.48), img("a2_02_roc_curves.png", content_width*0.48)],
    [img("a2_03_model_comparison.png", content_width*0.53), img("a2_06_per_class_f1_heatmap.png", content_width*0.43)],
]
t_combo = Table(fig_combo, colWidths=[content_width*0.52, content_width*0.48])
t_combo.setStyle(TableStyle([
    ("ALIGN",  (0,0),(-1,-1),"CENTER"),
    ("VALIGN", (0,0),(-1,-1),"MIDDLE"),
    ("LEFTPADDING",(0,0),(-1,-1),0),
    ("RIGHTPADDING",(0,0),(-1,-1),0),
    ("TOPPADDING",(0,0),(-1,-1),0),
    ("BOTTOMPADDING",(0,0),(-1,-1),0),
]))
story.append(t_combo)
story.append(Spacer(1, 2))

# ── 4. Interpretation of Results ──────────────────────────────────────────────
story += section("4. Interpretation, Limitations & Leakage Analysis", [
    Paragraph("<b>Key findings:</b> Random Forest and HistGradientBoosting achieved exceptionally high test metrics (F1-score ~95% and accuracy ~98%). The 5-fold CV F1-scores were also extremely high (~99%). The top features selected are credit progress (<code>cred_mat_total</code>) and course grades (<code>nota_asig_hash</code>). The ROC curve shows nearly perfect separation (AUC > 0.98).", s_body),
    Paragraph("<b>Methodological Leakage Analysis:</b> These extremely high scores represent severe methodological leakage. First, applying SMOTE globally on the training set <i>before</i> CV splits allows synthetic samples generated from training instances to leak into validation folds, inflating CV metrics. Second, splitting the dataset randomly at the row level (enrolment level) rather than the student level allows course records for the same student to be split between train and test sets (student identity leakage). The model is evaluating on seen students, which inflates the test set F1. These flaws are addressed in Assignment 3 by using Group-based split (GroupShuffleSplit/GroupKFold) and imblearn pipelines.", s_body),
])

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"\n✅ PDF saved → {PDF_PATH}")
