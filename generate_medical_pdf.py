import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf():
    os.makedirs("data", exist_ok=True)
    pdf_filename = os.path.join("data", "medical_book.pdf")
    
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, leading=24, spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6)
    
    story = []
    
    # Title
    story.append(Paragraph("Clinical Handbook of Internal Medicine & Patient Diagnostics", title_style))
    story.append(Spacer(1, 12))
    
    # Section 1: Diabetes Mellitus
    story.append(Paragraph("1. Diabetes Mellitus (Type 1 and Type 2)", heading_style))
    story.append(Paragraph(
        "Overview: Diabetes mellitus is a metabolic disorder characterized by persistent hyperglycemia resulting from defects in insulin secretion, insulin action, or both.",
        body_style
    ))
    story.append(Paragraph(
        "Symptoms: Classic symptoms include polyuria (frequent urination), polydipsia (increased thirst), polyphagia (increased hunger), unexplained weight loss, fatigue, blurred vision, and slow-healing sores or frequent infections.",
        body_style
    ))
    story.append(Paragraph(
        "Causes and Pathophysiology: Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells, leading to severe insulin deficiency. Type 2 diabetes is primarily driven by peripheral insulin resistance coupled with relative secretory defect.",
        body_style
    ))
    story.append(Paragraph(
        "Diagnosis: Fasting Plasma Glucose (FPG) >= 126 mg/dL (7.0 mmol/L), 2-hour Oral Glucose Tolerance Test (OGTT) >= 200 mg/dL (11.1 mmol/L), or HbA1c >= 6.5%.",
        body_style
    ))
    story.append(Paragraph(
        "Treatment & Management: Type 1 requires life-long exogenous subcutaneous insulin therapy. Type 2 management includes lifestyle changes, metformin (first-line drug, 500-1000mg twice daily), sulfonylureas, DPP-4 inhibitors, SGLT2 inhibitors, GLP-1 receptor agonists, and insulin when glycemic targets are unmet.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Section 2: Essential Hypertension
    story.append(Paragraph("2. Essential Hypertension", heading_style))
    story.append(Paragraph(
        "Overview: Essential (primary) hypertension is defined as persistent elevated blood pressure (systolic >= 130 mmHg or diastolic >= 80 mmHg) without a clear secondary cause.",
        body_style
    ))
    story.append(Paragraph(
        "Symptoms: Commonly asymptomatic ('the silent killer'). Severe hypertension may cause morning headaches, dizziness, chest tightness, dyspnea, epistaxis (nosebleeds), or visual disturbances.",
        body_style
    ))
    story.append(Paragraph(
        "Causes: Risk factors include sodium-rich diet, physical inactivity, obesity, excessive alcohol consumption, tobacco use, chronic stress, advanced age, and genetic predisposition.",
        body_style
    ))
    story.append(Paragraph(
        "Treatment: Dietary Approaches to Stop Hypertension (DASH diet), sodium restriction (<2,000 mg/day), regular aerobic exercise. First-line antihypertensive medications include ACE inhibitors (Lisinopril 10-40mg daily), ARBs (Losartan 50-100mg daily), Calcium Channel Blockers (Amlodipine 5-10mg daily), and Thiazide diuretics (Hydrochlorothiazide 12.5-25mg daily).",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Section 3: Bronchial Asthma
    story.append(Paragraph("3. Bronchial Asthma", heading_style))
    story.append(Paragraph(
        "Overview: Asthma is a chronic inflammatory disorder of the airways characterized by hyperresponsiveness, airway edema, and bronchoconstriction.",
        body_style
    ))
    story.append(Paragraph(
        "Symptoms: Recurrent episodes of wheezing, shortness of breath (dyspnea), chest tightness, and coughing, particularly at night or early morning.",
        body_style
    ))
    story.append(Paragraph(
        "Triggers: Airborne allergens (pollen, dust mites, pet dander), cold air, exercise, respiratory infections, smoke, and stress.",
        body_style
    ))
    story.append(Paragraph(
        "Treatment: Quick-relief medications include short-acting beta-agonists (SABA: Albuterol inhaler 2 puffs every 4-6 hours as needed). Long-term control medications include inhaled corticosteroids (Fluticasone, Budesonide) and long-acting beta-agonists (Salmeterol, Formoterol).",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 4: Acute Pneumonia
    story.append(Paragraph("4. Acute Community-Acquired Pneumonia", heading_style))
    story.append(Paragraph(
        "Overview: Pneumonia is an acute infection of the lung parenchyma caused by bacteria, viruses, or fungi, leading to alveolar consolidation.",
        body_style
    ))
    story.append(Paragraph(
        "Symptoms: High fever, chills, pleuritic chest pain, productive cough with rust-colored or purulent sputum, dyspnea, tachypnea, and fatigue.",
        body_style
    ))
    story.append(Paragraph(
        "Diagnosis: Chest X-ray demonstrating lobar consolidation or infiltrates, sputum Gram stain and culture, blood cultures, and elevated inflammatory markers (CRP, ESR).",
        body_style
    ))
    story.append(Paragraph(
        "Treatment: Empirical antibiotic therapy for outpatient pneumonia includes Amoxicillin (1g tid) or Macrolides (Azithromycin 500mg day 1, then 250mg daily). In hospitalized cases, IV Ceftriaxone plus Azithromycin or Levofloxacin is administered.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 5: Migraine Headaches
    story.append(Paragraph("5. Migraine Headaches", heading_style))
    story.append(Paragraph(
        "Overview: Migraine is a primary neurological headache disorder characterized by recurrent moderate to severe unilateral throbbing headache attacks lasting 4 to 72 hours.",
        body_style
    ))
    story.append(Paragraph(
        "Symptoms: Pulsating/throbbing unilateral headache, photophobia (sensitivity to light), phonophobia (sensitivity to sound), nausea, vomiting, and visual or sensory aura (scotomas, flashing lights in ~25% of patients).",
        body_style
    ))
    story.append(Paragraph(
        "Treatment: Acute treatment includes NSAIDs (Ibuprofen 400-800mg, Naproxen 500mg) and Triptans (Sumatriptan 50-100mg orally at onset). Prophylactic therapy includes Beta-blockers (Propranolol), Anticonvulsants (Topiramate), or CGRP monoclonal antibodies (Erenumab).",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 6: Body Mass Index (BMI) WHO Clinical Classification
    story.append(Paragraph("6. Body Mass Index (BMI) WHO Clinical Classification", heading_style))
    story.append(Paragraph(
        "Overview: Body Mass Index (BMI) is a simple weight-for-height index used to classify underweight, overweight, and obesity in adults. It is calculated as weight in kilograms divided by height in meters squared (kg/m²).",
        body_style
    ))
    story.append(Paragraph(
        "World Health Organization (WHO) Adult BMI Classification:\n"
        "• Underweight: BMI < 18.5 kg/m²\n"
        "• Normal weight (Healthy Range): BMI 18.5 – 24.9 kg/m²\n"
        "• Overweight (Pre-obesity): BMI 25.0 – 29.9 kg/m²\n"
        "• Obesity Class I: BMI 30.0 – 34.9 kg/m²\n"
        "• Obesity Class II: BMI 35.0 – 39.9 kg/m²\n"
        "• Obesity Class III (Severe/Morbid Obesity): BMI >= 40.0 kg/m²",
        body_style
    ))
    story.append(Paragraph(
        "Clinical Interpretation: Low BMI (<18.5 kg/m²) indicates underweight status and potential nutritional deficiency. Elevated BMI (>=25.0 kg/m²) increases risk for type 2 diabetes, essential hypertension, dyslipidemia, and metabolic syndrome.",
        body_style
    ))
    
    doc.build(story)
    print(f"Generated medical reference PDF at {pdf_filename}")


if __name__ == "__main__":
    create_pdf()
