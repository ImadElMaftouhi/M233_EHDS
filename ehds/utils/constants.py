"""Constantes et dictionnaires de référence pour le projet EHDS."""

# Tests de laboratoire LOINC (mini-dictionnaire)
LOINC_TESTS = [
    {"code": "718-7", "name": "Hemoglobin", "unit": "g/dL", "low": 12.0, "high": 17.0},
    {"code": "2160-0", "name": "Creatinine", "unit": "mg/dL", "low": 0.5, "high": 1.5},
    {"code": "2345-7", "name": "Glucose", "unit": "mg/dL", "low": 70.0, "high": 140.0},
    {"code": "6690-2", "name": "WBC", "unit": "10^3/uL", "low": 4.0, "high": 11.0},
]

# Codes ICD-10 (mini-dictionnaire)
ICD10 = {
    "I10": "Hypertension essentielle",
    "E11": "Diabète de type 2",
    "J45": "Asthme",
}

# Mapping médicament -> famille thérapeutique
DRUG_FAMILY = {
    "Amoxicillin": "Penicillins",
    "Penicillin V": "Penicillins",
    "Ibuprofen": "NSAIDs",
    "Aspirin": "NSAIDs",
    "Paracetamol": "Analgesics",
}

# Mapping allergie -> famille affectée (pour démonstration de contre-indication)
ALLERGY_FAMILY = {
    "Penicillin": "Penicillins",
    "Aspirin": "NSAIDs",
}
