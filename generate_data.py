# generate_team_data.py
from pathlib import Path
from ehds.data import DataGenerator

data_dir = Path("data")
generator = DataGenerator(data_dir=data_dir, seed=42)  # Same seed as teammates

# Generate with same parameters (120 patients, 600 labs)
print("Generating EHR patients CSV...")
generator.generate_ehr_patients_csv(n_patients=120)

print("Generating Lab results JSON...")
generator.generate_lab_results_json(n_records=600, n_patients=120)

print("Generating FHIR-like NDJSON (if not using real Synthea)...")

# Only if you're not using real Synthea data (madirohach ila drto clone l repo dyal synthea)
# generator.generate_fhir_like_ndjson(n_patients=120)

print("Generating DICOM series...")
generator.generate_dicom_series(n_patients=120, n_studies=150)

print("✓ All data generated!")