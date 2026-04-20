"""
Assignment 2 — Report Generator
Produces: Tuncer_Gungoren_assignment2.pdf
"""

import os
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
    Paragraph("AI-Based Prediction of Student Dropout and Academic Success", s_title),
    Paragraph("Assignment 2 — Model Exploration &amp; Results", s_sub),
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
    Paragraph("Four models were selected to cover a spectrum of complexity, supporting the SDG 4 objective of predicting dropout:", s_body),
    Paragraph("• <b>Logistic Regression</b> — Linear baseline; transparent coefficients easily explainable to academic advisors.", s_bullet),
    Paragraph("• <b>Random Forest</b> — Robust ensemble handle non-linear interactions; provides reliable feature importance scores.", s_bullet),
    Paragraph("• <b>Gradient Boosting</b> — Boosting of weak learners; captures subtle patterns in semester-by-semester performance.", s_bullet),
    Paragraph("• <b>Neural Network (MLP)</b> — Two-hidden-layer network serving as a deep-learning representation benchmark.", s_bullet),
])
story.append(Spacer(1, 2))

# ── 2. Data Preparation ───────────────────────────────────────────────────────
story += section("2. Data Preparation for Modelling", [
    Paragraph("The UCI dataset (4,424 students, 36 features, 0 missing values) was prepared as follows:", s_body),
    Paragraph("• <b>Train/Test Split:</b> 80% training / 20% test, stratified to preserve the 49.9% Graduate / 32.1% Dropout / 17.9% Enrolled ratio.", s_bullet),
    Paragraph("• <b>Feature Engineering &amp; Normalisation:</b> All features retained; Target label-encoded; StandardScaler applied to metrics.", s_bullet),
    Paragraph("• <b>Class Imbalance:</b> SMOTE applied to training set only, balancing all classes to equal frequency without test leak.", s_bullet),
])
story.append(Spacer(1, 2))

# ── 3. Model Training & Evaluation ───────────────────────────────────────────
story += section("3. Model Training & Evaluation", [
    Paragraph("Models were trained on the SMOTE-balanced training set and evaluated on the held-out test set using 5-fold CV:", s_body),
])

hdr = ["Model", "Accuracy", "Precision", "Recall", "F1-Score (Macro)", "CV F1 ± Std"]
rows_data = [
    ["Logistic Regression",   "0.7356", "0.7062", "0.7065", "0.6969", "0.7425 ± 0.0209"],
    ["Random Forest",         "0.7706", "0.7246", "0.7127", "0.7169", "0.8417 ± 0.0168"],
    ["Gradient Boosting",     "0.7537", "0.7009", "0.6941", "0.6966", "0.8296 ± 0.0121"],
    ["Neural Network (MLP)",  "0.6814", "0.6211", "0.6186", "0.6188", "0.8557 ± 0.0167"],
]
table_data = [[Paragraph(h, s_table_hdr) for h in hdr]]
for i, row in enumerate(rows_data):
    style = s_table_cell
    is_best = (i == 1)
    table_data.append([Paragraph(f"<b>{row[0]}</b>" if is_best else row[0], style), *[Paragraph(f"<b>{v}</b>" if is_best else v, style) for v in row[1:]]])

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
story += section("4. Interpretation, Limitations & Improvements", [
    Paragraph("<b>Key findings:</b> Random Forest achieved the best test performance (Acc=77.1%, F1=0.717). The 'Enrolled' class proved hardest to classify across all models (F1≈0.37–0.52), likely because enrolled students are mid-journey. ROC confirms strong discriminability for Dropout vs. Graduate (AUC&gt;0.88), which is operationally crucial. Top predictors are 2nd-semester performance, tuition status, and scholarship holding.", s_body),
    Paragraph("<b>Limitations &amp; Improvements:</b> MLP showed signs of overfitting to SMOTE (high CV F1, low Test F1). Limitations include single-institution data and outcome ambiguity. Future improvements: Hyperparameter optimization, mutual information feature selection, and dashboard UI integration.", s_body),
])

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"\n✅ PDF saved → {PDF_PATH}")
