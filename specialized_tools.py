import re
from typing import Dict, Any, List, Union, Optional, Tuple
from langchain_core.tools import tool

# 1. ENHANCED DETERMINISTIC SAFETY TAXONOMY (Normalized Phrases & Medical Synonyms)
RED_FLAG_TAXONOMY = {
    "CARDIOVASCULAR": [
        "chest pain", "crushing chest pain", "chest tightness", "chest pressure",
        "pain radiating to arm", "radiating to jaw", "heart attack", "cardiac arrest",
        "crushing pressure", "severe palpitations with dizziness"
    ],
    "NEUROLOGICAL": [
        "facial drooping", "face drooping", "arm weakness", "slurred speech",
        "stroke", "sudden numbness", "sudden paralysis", "thunderclap headache",
        "worst headache of my life", "sudden blindness", "loss of vision", "seizure",
        "unconscious", "unresponsive", "collapsed", "fainting episode", "loss of consciousness", "fainted"
    ],
    "RESPIRATORY": [
        "shortness of breath", "severe dyspnea", "can't breathe", "cannot breathe",
        "gasping for air", "cyanosis", "blue lips", "stridor", "coughing blood",
        "hemoptysis", "choking"
    ],
    "ALLERGIC_ANAPHYLAXIS": [
        "anaphylaxis", "throat closing", "tongue swelling", "swollen lips and throat",
        "severe allergic reaction", "difficulty swallowing with rash"
    ],
    "PSYCHIATRIC_CRISIS": [
        "suicidal", "suicide", "want to die", "self harm", "harming myself"
    ]
}






SYNONYM_MAP = {
    "shortness of breath": "dyspnea",
    "breathlessness": "dyspnea",
    "difficulty breathing": "dyspnea",
    "can't breathe": "dyspnea",
    "dizziness": "vertigo",
    "dizzy": "vertigo",
    "lightheadedness": "vertigo",
    "feeling faint": "vertigo",
    "stomach pain": "abdominal_pain",
    "belly ache": "abdominal_pain",
    "stomach ache": "abdominal_pain",
    "abdominal pain": "abdominal_pain",
    "nausea": "nausea_vomiting",
    "vomiting": "nausea_vomiting",
    "throwing up": "nausea_vomiting",
    "upset stomach": "nausea_vomiting",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "weakness": "fatigue",
    "feeling weak": "fatigue",
    "joint pain": "joint_pain",
    "swollen joints": "joint_pain",
    "arthralgia": "joint_pain",
    "racing heart": "palpitations",
    "irregular heartbeat": "palpitations",
    "heart fluttering": "palpitations",
    "skin rash": "rash",
    "hives": "rash",
    "lower back pain": "back_pain",
    "back ache": "back_pain",
}

SYMPTOM_DATABASE: Dict[str, Dict[str, Any]] = {
    "fever": {
        "name": "Fever (Pyrexia)",
        "category": "General / Systemic",
        "description": "An elevation in body temperature above normal (>38.0°C / 100.4°F), typically caused by infection or inflammation.",
        "common_causes": ["Viral infections (Flu, COVID-19)", "Bacterial infections (UTI, Pneumonia)", "Inflammatory disorders"],
        "when_to_seek_care": "Seek care if fever >103°F (39.4°C), lasts >3 days, or is accompanied by stiff neck, rash, or confusion.",
        "red_flags": "Fever with stiff neck, severe headache, confusion, or dark purple rash."
    },
    "headache": {
        "name": "Headache (Cephalgia)",
        "category": "Neurological",
        "description": "Pain arising from head or neck structures. Can be primary (Tension, Migraine) or secondary.",
        "common_causes": ["Tension / Stress", "Migraine", "Dehydration or fatigue", "Sinusitis"],
        "when_to_seek_care": "Seek care if headaches are unusually severe, waking you from sleep, or worsening.",
        "red_flags": "Sudden 'thunderclap' onset (worst headache of life), headache with fever and stiff neck, or focal neurological deficits."
    },
    "chest pain": {
        "name": "Chest Pain (Thoracalgia)",
        "category": "Cardiovascular / Thoracic",
        "description": "Discomfort or pain along the front of the chest, requiring immediate evaluation to rule out cardiac emergencies.",
        "common_causes": ["Musculoskeletal strain", "GERD", "Anxiety / Panic attack", "Pericarditis"],
        "when_to_seek_care": "Any new, unexplained chest pain should be evaluated by a healthcare professional.",
        "red_flags": "Crushing pressure, pain radiating to left arm or jaw, shortness of breath, cold sweat."
    },
    "dyspnea": {
        "name": "Shortness of Breath (Dyspnea)",
        "category": "Respiratory / Cardiovascular",
        "description": "Subjective experience of breathing discomfort or feeling unable to draw a full breath.",
        "common_causes": ["Asthma / COPD", "Respiratory infection", "Anxiety", "Deconditioning"],
        "when_to_seek_care": "Seek prompt medical evaluation if dyspnea occurs at rest or with minimal exertion.",
        "red_flags": "Sudden onset dyspnea, inability to speak full sentences, blue lips/fingernails, or chest pain."
    },
    "priapism": {
        "name": "Priapism",
        "category": "Urological Emergency",
        "description": "A persistent, usually painful penile erection lasting longer than 4 hours, unrelated to sexual stimulation.",
        "common_causes": ["Sickle cell disease", "Medications (PDE5 inhibitors, intracavernosal injections)", "Trauma"],
        "when_to_seek_care": "Priapism is a medical emergency requiring immediate urological treatment to prevent permanent tissue damage.",
        "red_flags": "Erection lasting >4 hours requires urgent Emergency Room evaluation."
    },
    "abdominal_pain": {
        "name": "Abdominal Pain",
        "category": "Gastrointestinal",
        "description": "Pain localized to the stomach or abdominal region.",
        "common_causes": ["Gastroenteritis", "Gastritis / GERD", "IBS", "Constipation / Gas"],
        "when_to_seek_care": "Seek care if pain is severe, persistent (>24 hrs), or accompanied by persistent vomiting.",
        "red_flags": "Severe right lower quadrant pain (Appendicitis), rigid abdomen, high fever, or vomiting blood."
    },
    "vertigo": {
        "name": "Dizziness & Vertigo",
        "category": "Neurological / Otological",
        "description": "A sensation of spinning, motion, or lightheadedness affecting balance.",
        "common_causes": ["BPPV", "Vestibular neuritis", "Dehydration / Orthostatic hypotension"],
        "when_to_seek_care": "Seek medical evaluation if vertigo is persistent, impairs walking, or causes recurrent falls.",
        "red_flags": "Vertigo accompanied by double vision, slurred speech, facial numbness, or sudden hearing loss."
    },
    "fatigue": {
        "name": "Fatigue & Lethargy",
        "category": "General / Metabolic",
        "description": "A state of persistent physical or mental exhaustion not relieved by sleep.",
        "common_causes": ["Sleep deprivation / Sleep apnea", "Anemia", "Hypothyroidism", "Chronic stress"],
        "when_to_seek_care": "Seek evaluation if fatigue persists >2 weeks despite adequate sleep.",
        "red_flags": "Fatigue accompanied by unexplained weight loss, night sweats, or swollen lymph nodes."
    },
    "nausea_vomiting": {
        "name": "Nausea & Vomiting",
        "category": "Gastrointestinal / Systemic",
        "description": "Sensation of an urge to vomit, frequently accompanied by involuntary forceful expulsion of stomach contents.",
        "common_causes": ["Acute Gastroenteritis", "Food Poisoning", "Motion Sickness", "Migraine", "Medication Side Effect", "Early Pregnancy"],
        "when_to_seek_care": "Maintain hydration with oral electrolyte solutions. See a doctor if vomiting persists beyond 24 hours or leads to inability to retain fluids.",
        "red_flags": "🚨 Emergency Red Flags: Coffee-ground or bloody vomit, severe intractable headache with stiff neck, signs of severe dehydration (no urine for >8h, confusion)."
    },
    "fever_pyrexia": {
        "name": "Fever (Pyrexia)",
        "category": "Infectious / Immunological",
        "description": "Temporary elevation in body temperature above normal baseline (>38.0°C or 100.4°F), typically part of an immune response.",
        "common_causes": ["Viral Upper Respiratory Infection (Flu/Cold)", "Urinary Tract Infection (UTI)", "Bacterial Pneumonia", "Ear Infection", "Post-Vaccination Reaction"],
        "when_to_seek_care": "Monitor temperature and rest. Seek medical guidance if fever exceeds 39.5°C (103°F) or lasts >3 consecutive days.",
        "red_flags": "🚨 Emergency Red Flags: Fever accompanied by stiff neck, confusion, petechial skin rash, difficulty breathing, or lethargy."
    },
    "headache_cephalalgia": {
        "name": "Headache (Cephalalgia)",
        "category": "Neurological",
        "description": "Pain in the head or upper neck caused by disturbance of pain-sensitive structures.",
        "common_causes": ["Tension-Type Headache", "Migraine", "Sinusitis", "Dehydration / Fatigue", "Caffeine Withdrawal"],
        "when_to_seek_care": "Use standard OTC analgesics if appropriate. Consult a doctor for headaches increasing in frequency, severity, or interfering with work.",
        "red_flags": "🚨 Emergency Red Flags: Thunderclap headache (instant peak severity), headache following head trauma, fever with neck stiffness, focal neurological deficits."
    }
}


@tool
def symptom_information(symptom_name: str) -> str:
    """
    Retrieves structured, non-diagnostic clinical reference information for a specified symptom.
    Supports normalized phrase matching and handles invalid arguments safely.
    """
    if not symptom_name or not isinstance(symptom_name, str):
        return "Error: Please provide a valid symptom name string."
        
    s_clean = symptom_name.lower().strip()
    normalized_key = SYNONYM_MAP.get(s_clean, s_clean)
    
    symptom_data = None
    if normalized_key in SYMPTOM_DATABASE:
        symptom_data = SYMPTOM_DATABASE[normalized_key]
    else:
        for key, data in SYMPTOM_DATABASE.items():
            if key in s_clean or s_clean in key:
                symptom_data = data
                break
                
    if not symptom_data:
        available = ", ".join([d["name"] for d in SYMPTOM_DATABASE.values()])
        return (
            f"ℹ️ Structured symptom guide for '{symptom_name}' is not pre-indexed.\n"
            f"Available pre-indexed symptom guides: {available}.\n"
            f"For detailed medical guidance on this symptom, use `medical_knowledge_search`."
        )
        
    return (
        f"📋 **Structured Symptom Guide: {symptom_data['name']}**\n"
        f"• **Category**: {symptom_data['category']}\n"
        f"• **Clinical Description**: {symptom_data['description']}\n"
        f"• **Common Non-Diagnostic Causes**: {', '.join(symptom_data['common_causes'])}\n"
        f"• **When to Seek Care**: {symptom_data['when_to_seek_care']}\n"
        f"• **🚨 Critical Red Flags**: {symptom_data['red_flags']}\n\n"
        f"*(Note: Educational and non-diagnostic. Always consult a physician for clinical assessment.)*"
    )


import json

@tool
def medical_calculator(
    metric_type: str = "BMI",
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
    age_years: Optional[int] = None,
    gender: Optional[str] = None
) -> str:


    """
    Performs deterministic clinical metric calculations (BMI, BMR, IBW).
    
    Requirements per metric:
    - BMI: Requires weight_kg and height_cm.
    - BMR: Requires weight_kg, height_cm, age_years, AND gender ('male'/'female'). Do NOT guess or default missing age or gender.
    - IBW: Requires height_cm AND gender ('male'/'female'). Do NOT guess or default missing gender.
    
    If required parameters are missing from the user query, pass None for missing parameters so the tool returns an explicit prompt requesting them.
    """
    if not metric_type or not isinstance(metric_type, str):
        return json.dumps({"status": "error", "message": "Please specify metric_type as 'BMI', 'BMR', or 'IBW'."})
        
    m_type = str(metric_type).upper().strip()

    def parse_float(val):
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def parse_int(val):
        if val is None or str(val).strip() == "":
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    w = parse_float(weight_kg)
    h = parse_float(height_cm)
    a = parse_int(age_years)
    g_clean = str(gender).lower().strip() if gender is not None and str(gender).strip() else None

    if "BMI" in m_type:
        missing = []
        if w is None:
            missing.append("weight_kg")
        if h is None:
            missing.append("height_cm")
        if missing:
            res = {
                "calculation_type": "BMI",
                "status": "missing_inputs",
                "message": "⚠️ Missing Information: Please provide your weight in kg and height in cm so I can calculate your BMI."
            }
            return json.dumps(res)
            
        if w <= 0 or h <= 0:
            res = {
                "calculation_type": "BMI",
                "status": "error",
                "message": "Error: Weight and height must be positive non-zero numbers."
            }
            return json.dumps(res)
            
        height_m = round(h / 100.0, 2)
        bmi_raw = w / ((h / 100.0) ** 2)
        bmi = round(bmi_raw, 1)
        
        res = {
            "calculation_type": "BMI",
            "weight_kg": w,
            "height_cm": h,
            "height_m": height_m,
            "bmi": bmi,
            "status": "success"
        }
        return json.dumps(res)
        
    elif "BMR" in m_type:
        missing = []
        if w is None:
            missing.append("weight_kg")
        if h is None:
            missing.append("height_cm")
        if a is None:
            missing.append("age_years")
        if g_clean is None:
            missing.append("gender ('male' or 'female')")
        if missing:
            res = {
                "calculation_type": "BMR",
                "status": "missing_inputs",
                "message": f"⚠️ Missing Information: BMR calculation requires {', '.join(missing)}. Please provide those details."
            }
            return json.dumps(res)
            
        if w <= 0 or h <= 0 or a <= 0:
            res = {
                "calculation_type": "BMR",
                "status": "error",
                "message": "Error: Weight, height, and age must be positive non-zero numbers."
            }
            return json.dumps(res)

        if g_clean.startswith("f"):
            bmr = (10 * w) + (6.25 * h) - (5 * a) - 161
        else:
            bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
            
        res = {
            "calculation_type": "BMR",
            "bmr": round(bmr, 0),
            "status": "success"
        }
        return json.dumps(res)
        
    elif "IBW" in m_type:
        missing = []
        if h is None:
            missing.append("height_cm")
        if g_clean is None:
            missing.append("gender ('male' or 'female')")
        if missing:
            res = {
                "calculation_type": "IBW",
                "status": "missing_inputs",
                "message": f"⚠️ Missing Information: Ideal Body Weight (IBW) calculation requires {', '.join(missing)}. Please specify those details."
            }
            return json.dumps(res)
            
        if h <= 0:
            res = {
                "calculation_type": "IBW",
                "status": "error",
                "message": "Error: Height must be a positive non-zero number."
            }
            return json.dumps(res)

        height_inches = h / 2.54
        inches_over_5ft = max(0, height_inches - 60)
        
        if g_clean.startswith("f"):
            ibw = 45.5 + (2.3 * inches_over_5ft)
        else:
            ibw = 50.0 + (2.3 * inches_over_5ft)
            
        res = {
            "calculation_type": "IBW",
            "ibw_kg": round(ibw, 1),
            "status": "success"
        }
        return json.dumps(res)
        
    return json.dumps({"status": "error", "message": f"Unsupported metric type '{metric_type}'. Supported metrics are 'BMI', 'BMR', and 'IBW'."})


