"""Générateurs de données simulées pour le projet EHDS."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from ..utils import (
    LOINC_TESTS,
    ICD10,
    DRUG_FAMILY,
    ALLERGY_FAMILY,
    deterministic_uid,
    ensure_dir,
)


@dataclass
class DataGenerator:
    """Générateur de données simulées pour tests et démonstrations."""

    data_dir: Path = Path("data")
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise les répertoires et la graine aléatoire."""
        random.seed(self.seed)
        ensure_dir(self.data_dir)
        ensure_dir(self.data_dir / "source_ehr_csv")
        ensure_dir(self.data_dir / "source_lab_json")
        ensure_dir(self.data_dir / "source_fhir_ndjson")
        ensure_dir(self.data_dir / "source_dicom")
        ensure_dir(self.data_dir / "integrated")
        ensure_dir(self.data_dir / "rdf")

    def generate_ehr_patients_csv(self, n_patients: int = 120) -> Path:
        """Génère un fichier CSV de patients EHR simulés."""
        genders = ["male", "female"]
        rows = []
        for i in range(1, n_patients + 1):
            pid = f"P{i:04d}"
            birth = datetime.now() - timedelta(days=random.randint(18 * 365, 90 * 365))
            rows.append(
                {
                    "patient_id": pid,
                    "first_name": random.choice(["Ahmed", "Leila", "Sara", "Youssef", "Nora", "Omar"]),
                    "last_name": random.choice(["Benali", "El Amrani", "Khaldi", "Ait", "Maaquili"]),
                    "gender": random.choice(genders),
                    "birthDate": birth.date().isoformat(),
                    "country": random.choice(["FR", "MA", "ES", "DE"]),
                }
            )
        df = pd.DataFrame(rows)
        out = self.data_dir / "source_ehr_csv" / "ehr_patients.csv"
        df.to_csv(out, index=False)
        print(f"✓ Generated EHR patients CSV: {out} ({len(df)} rows)")
        return out

    def generate_lab_results_json(self, n_records: int = 600, n_patients: int = 120) -> Path:
        """Génère un fichier JSON de résultats de laboratoire simulés."""
        labs: List[dict] = []

        # Add unit heterogeneity for Creatinine (mg/dL vs µmol/L)
        for i in range(n_records):
            test = random.choice(LOINC_TESTS)
            pid = f"P{random.randint(1, n_patients):04d}"

            value = round(random.uniform(test["low"], test["high"]), 2)
            unit = test["unit"]

            # 25% of creatinine as µmol/L to demonstrate unit conversion
            if test["name"] == "Creatinine" and random.random() < 0.25:
                # 1 mg/dL = 88.4 µmol/L (approx)
                value = round(value * 88.4, 1)
                unit = "µmol/L"

            labs.append(
                {
                    "lab_id": f"LAB{i:06d}",
                    "patient_id": pid,
                    "test_code_loinc": test["code"],
                    "test_name": test["name"],
                    "value": value,
                    "unit": unit,
                    "date": (datetime.now() - timedelta(days=random.randint(0, 365))).date().isoformat(),
                    "status": random.choice(["final", "preliminary"]),
                    "source_system": random.choice(["Lab_A", "Lab_B"]),
                }
            )

        out = self.data_dir / "source_lab_json" / "lab_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(labs, f, indent=2, ensure_ascii=False)
        print(f"✓ Generated Lab JSON: {out} ({len(labs)} rows)")
        return out

    def generate_fhir_like_ndjson(self, n_patients: int = 120) -> Path:
        """Génère un fichier NDJSON avec des ressources FHIR simulées."""
        out = self.data_dir / "source_fhir_ndjson" / "bundle.ndjson"

        conditions = list(ICD10.keys())
        allergies = list(ALLERGY_FAMILY.keys())
        drugs = list(DRUG_FAMILY.keys())

        with open(out, "w", encoding="utf-8") as f:
            for i in range(1, n_patients + 1):
                pid = f"P{i:04d}"

                # Patient
                f.write(
                    json.dumps(
                        {
                            "resourceType": "Patient",
                            "id": pid,
                            "gender": random.choice(["male", "female"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Condition (ICD-10)
                if random.random() < 0.65:
                    code = random.choice(conditions)
                    f.write(
                        json.dumps(
                            {
                                "resourceType": "Condition",
                                "id": f"C{i:04d}",
                                "subject": {"reference": f"Patient/{pid}"},
                                "code": {"coding": [{"system": "ICD-10", "code": code}]},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                # AllergyIntolerance
                if random.random() < 0.35:
                    al = random.choice(allergies)
                    f.write(
                        json.dumps(
                            {
                                "resourceType": "AllergyIntolerance",
                                "id": f"A{i:04d}",
                                "patient": {"reference": f"Patient/{pid}"},
                                "code": {"text": al},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                # MedicationRequest (prescription)
                if random.random() < 0.55:
                    drug = random.choice(drugs)
                    f.write(
                        json.dumps(
                            {
                                "resourceType": "MedicationRequest",
                                "id": f"M{i:04d}",
                                "subject": {"reference": f"Patient/{pid}"},
                                "medicationCodeableConcept": {"text": drug},
                                "authoredOn": (datetime.now() - timedelta(days=random.randint(0, 90))).date().isoformat(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        print(f"✓ Generated FHIR-like NDJSON: {out}")
        return out

    def generate_dicom_series(self, n_patients: int = 120, n_studies: int = 150) -> Path:
        """Génère des fichiers DICOM simulés organisés en études."""
        base = self.data_dir / "source_dicom"
        ensure_dir(base)

        modalities = ["CT", "MR"]
        body_parts = ["HEAD", "CHEST", "ABDOMEN", "KNEE", "SPINE"]
        rng = np.random.default_rng(self.seed)

        for study_idx in range(1, n_studies + 1):
            pid = f"P{random.randint(1, n_patients):04d}"
            study_uid = deterministic_uid(f"study-{study_idx}-{pid}")
            series_uid = deterministic_uid(f"series-{study_idx}-{pid}")
            study_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y%m%d")
            study_dir = base / f"patient_{pid}" / f"study_{study_uid}"
            ensure_dir(study_dir)

            for instance_idx in range(1, random.randint(1, 3) + 1):
                sop_uid = deterministic_uid(f"sop-{study_idx}-{instance_idx}-{pid}")
                modality = random.choice(modalities)
                rows = 128
                cols = 128
                pixel = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)

                filename = study_dir / f"{sop_uid}.dcm"
                file_meta = Dataset()
                file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
                file_meta.MediaStorageSOPInstanceUID = sop_uid
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                file_meta.ImplementationClassUID = deterministic_uid("impl")

                ds = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)
                ds.SOPClassUID = SecondaryCaptureImageStorage
                ds.PatientID = pid
                ds.StudyInstanceUID = study_uid
                ds.SeriesInstanceUID = series_uid
                ds.SOPInstanceUID = sop_uid
                ds.Modality = modality
                ds.StudyDate = study_date
                ds.BodyPartExamined = random.choice(body_parts)
                ds.Rows = rows
                ds.Columns = cols
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
                ds.PixelRepresentation = 0
                ds.PixelData = pixel.tobytes()
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                ds.save_as(str(filename), write_like_original=False)

        print(f"✓ Generated DICOM series: {base}")
        return base
