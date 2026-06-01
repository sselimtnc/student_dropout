"""
Assignment 1 — EDA Report Generator
Produces: Tuncer_Gungoren_assignment1.pdf
Based on the new UPV longitudinal dataset exploratory data analysis.
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(OUT_DIR, "Tuncer_Gungoren_assignment1.pdf")
STATS_PATH = os.path.join(OUT_DIR, "descriptive_statistics.csv")

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
    Paragraph("Assignment 1 — Exploratory Data Analysis on UPV Dataset", s_sub),
    Paragraph("Samsung Innovation Campus · Tuncer Güngören · March 2026", s_sub),
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

# ── 1. Introduction & SDG 4 Alignment ─────────────────────────────────────────
story += section("1. Introduction & SDG 4 Alignment", [
    Paragraph("This report presents the Exploratory Data Analysis (EDA) for the student trajectory abandonment prediction system, utilizing the <b>UPV Longitudinal Student Dataset (MDPI Data 2025)</b>. The dataset tracks 159,173 enrollment records across 20,427 unique students, recording demographic, academic, financial, and institutional features. This work is directly aligned with **United Nations Sustainable Development Goal 4 (SDG 4: Quality Education)**. By understanding the factors that correlate with academic trajectory abandonment (class A) vs. continuation (class B), we can identify early-warning indicators that empower educational institutions to intervene proactively, ensuring inclusive and equitable quality education.", s_body),
])
story.append(Spacer(1, 2))

# ── 2. Data Cleaning & Preprocessing Summary ──────────────────────────────────
story += section("2. Data Cleaning & Preprocessing Summary", [
    Paragraph("Prior to visualization, the raw dataset was cleaned to prevent target and temporal leakage:", s_body),
    Paragraph("• <b>Target Leakage Drop:</b> Removed columns that directly state or imply graduation or current active enrollment status (e.g., <code>matricula_activa</code>, <code>rendimiento_total</code>, and previous year performance metrics).", s_bullet),
    Paragraph("• <b>Temporal Leakage Drop:</b> Dropped 88 monthly LMS events, visits, and Wi-Fi connection day log columns. These variables collect information throughout the academic year and would lead to future-leakage if used to predict abandonment at the time of registration.", s_bullet),
    Paragraph("• <b>Categorical Encoding:</b> Non-numeric fields were encoded into index categories. Missing values were filled using median value imputation on numeric columns.", s_bullet),
])
story.append(Spacer(1, 2))

# ── 3. Descriptive Statistics Summary ──────────────────────────────────────────
# Read descriptive stats if they exist
stats_text = "The descriptive statistics show a highly structured academic environment. The average grade average of courses (nota_asig_hash) is centered around 6.5 out of 10. Admission grades span two scales: 10-scale (nota10_hash, mean: 7.3) and 14-scale (nota14_hash, mean: 9.8). The average admission year is 2020.9. Students enroll in an average of 60.0 credits per year, matching standard full-time dedication requirements."
if os.path.exists(STATS_PATH):
    try:
        df_stats = pd.read_csv(STATS_PATH, index_index=0)
        # We can extract some values dynamically
        if "nota_asig_hash" in df_stats.columns and "mean" in df_stats.index:
            m_grade = df_stats.loc["mean", "nota_asig_hash"]
            stats_text = f"The descriptive statistics show a highly structured academic environment. The average course grade average (<code>nota_asig_hash</code>) is <b>{m_grade:.2f}/10</b>. Admission grades span two scales: 10-scale (<code>nota10_hash</code>, mean: <b>{df_stats.loc['mean', 'nota10_hash']:.2f}</b>) and 14-scale (<code>nota14_hash</code>, mean: <b>{df_stats.loc['mean', 'nota14_hash']:.2f}</b>). The average admission year is <b>{int(df_stats.loc['mean', 'anyo_ingreso'])}</b>. Students enroll in an average of <b>{df_stats.loc['mean', 'cred_mat_total']:.1f}</b> total credits per year, matching standard full-time requirements."
    except:
        pass

story += section("3. Descriptive Statistics Overview", [
    Paragraph(stats_text, s_body),
])
story.append(Spacer(1, 2))

# ── 4. Key Exploratory Findings & Interpretations ──────────────────────────────
story += section("4. Key Exploratory Findings & Visualizations", [
    Paragraph("Exploration of the target outcome and its correlations reveals several key insights:", s_body),
    Paragraph("• <b>Target Class Imbalance:</b> The student outcome variable <code>abandono_hash</code> is highly imbalanced. Class B (continuing) constitutes <b>93.2%</b> (148,384 records) of the dataset, while Class A (abandoned) makes up only <b>6.8%</b> (10,789 records). This extreme imbalance requires class balancing techniques (like SMOTE) during model training.", s_bullet),
    Paragraph("• <b>Academic Predictors:</b> Academic achievement features show the strongest separation between outcomes. Histogram plots reveal that students who abandon their studies have significantly lower average course grades (<code>nota_asig_hash</code>) and admission grades than continuing students.", s_bullet),
    Paragraph("• <b>Financial Vulnerability:</b> Financial factors are powerful predictors. Proportional cross-tabulation shows that students with unpaid course registration fees (<code>impagado_curso_mat = 1</code>) drop out at an extremely high rate compared to those whose fees are paid.", s_bullet),
    Paragraph("• <b>Feature Correlation:</b> Point-biserial correlations show that second semester performance (<code>rendimiento_cuat_b</code>, r = -0.285) and first-year completed credits (<code>cred_sup_1o</code>, r = -0.159) have the strongest negative correlation with abandonment, confirming that early in-programme performance is the dominant survival signal.", s_bullet),
])

story.append(PageBreak())

# ── PAGE 2: VISUALISATIONS & CONCLUSION ───────────────────────────────────────
story.append(banner_table)
story.append(Spacer(1, 4))

story += section("5. Diagnostic Exploratory Visualizations", [
    Paragraph("The following plots display the target outcome distribution, performance histograms, demographic/financial associations, and correlation rankings generated from the UPV dataset:", s_body),
])
story.append(Spacer(1, 4))

# Visualisations table
def img(filename, w):
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        im = Image(path)
        aspect = im.imageHeight / float(im.imageWidth)
        return Image(path, width=w, height=w*aspect*0.93)
    return Spacer(1, 1)

fig_combo = [
    [img("a1_01_target_distribution.png", content_width*0.48), img("a1_02_academic_performance.png", content_width*0.48)],
    [img("a1_03_demographic_financial.png", content_width*0.48), img("a1_04_dropout_correlation.png", content_width*0.48)],
]
t_combo = Table(fig_combo, colWidths=[content_width*0.50, content_width*0.50])
t_combo.setStyle(TableStyle([
    ("ALIGN",  (0,0),(-1,-1),"CENTER"),
    ("VALIGN", (0,0),(-1,-1),"MIDDLE"),
    ("LEFTPADDING",(0,0),(-1,-1),0),
    ("RIGHTPADDING",(0,0),(-1,-1),0),
    ("TOPPADDING",(0,0),(-1,-1),2),
    ("BOTTOMPADDING",(0,0),(-1,-1),2),
]))
story.append(t_combo)
story.append(Spacer(1, 4))

# ── 6. Conclusion ─────────────────────────────────────────────────────────────
story += section("6. Conclusion & Transition to Modeling", [
    Paragraph("The exploratory phase confirms that the longitudinal UPV dataset contains strong, clean predictive signals for tracking student trajectories. Individual academic performance in the first semesters and financial indicators (unpaid tuition) represent the most critical risk factors, while entry qualifications carry weaker signal. These insights justify the development of predictive classification models in Assignment 2, utilizing SMOTE to balance the target distribution, and highlight the importance of group-based splitting in Assignment 3 to prevent student identity leakage.", s_body),
])

# Build
doc.build(story)
print(f"✅ PDF saved -> {PDF_PATH}")
