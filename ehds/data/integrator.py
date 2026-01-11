"""Intégration et transformation de données multi-sources."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pydicom

from ..utils import (
    LOINC_TESTS,
    ICD10,
    DRUG_FAMILY,
    sha256_pseudo,
    ensure_dir,
)


class DataIntegrator:
    """Intègre et unifie les données de sources hétérogènes."""

    def __init__(self, data_dir: Path = Path("data")) -> None:
        """Initialise l'intégrateur avec le répertoire de données."""
        self.data_dir = data_dir
        ensure_dir(self.data_dir / "integrated")

    @staticmethod
    def convert_creatinine_to_mgdl(value: float, unit: str) -> Tuple[float, str]:
        """Normalise la créatinine de µmol/L vers mg/dL."""
        if unit == "µmol/L":
            # 1 mg/dL = 88.4 µmol/L
            return round(value / 88.4, 3), "mg/dL"
        return float(value), unit

    def load_ehr_csv(self) -> pd.DataFrame:
        """Charge les données EHR depuis un fichier CSV."""
        path = self.data_dir / "source_ehr_csv" / "ehr_patients.csv"
        df = pd.read_csv(path)
        df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)
        return df

    def load_lab_json(self) -> pd.DataFrame:
        """Charge les résultats de laboratoire depuis un fichier JSON."""
        path = self.data_dir / "source_lab_json" / "lab_results.json"
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
        df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)

        # Normalisation des unités pour la créatinine
        mask = df["test_name"].eq("Creatinine")
        if mask.any():
            converted = df.loc[mask].apply(
                lambda r: self.convert_creatinine_to_mgdl(float(r["value"]), str(r["unit"])),
                axis=1,
                result_type="expand",
            )
            df.loc[mask, "value"] = converted[0]
            df.loc[mask, "unit"] = converted[1]

        return df

    def load_fhir_ndjson(self) -> Dict[str, pd.DataFrame]:
        """Charge les données FHIR depuis un fichier NDJSON."""
        # Chercher dans bronze/fhir d'abord, puis source_fhir_ndjson comme fallback
        path = self.data_dir / "bronze" / "fhir" / "bundle.ndjson"
        if not path.exists():
            path = self.data_dir / "source_fhir_ndjson" / "bundle.ndjson"
        
        patients, conds, alls, meds = [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rt = obj.get("resourceType")
                if rt == "Patient":
                    pid = obj["id"]
                    patients.append(
                        {
                            "patient_id": pid,
                            "gender_fhir": obj.get("gender"),
                        }
                    )
                elif rt == "Condition":
                    pid = obj["subject"]["reference"].split("/")[-1]
                    code = obj["code"]["coding"][0]["code"]
                    conds.append({"patient_id": pid, "icd10_code": code})
                elif rt == "AllergyIntolerance":
                    pid = obj["patient"]["reference"].split("/")[-1]
                    alls.append({"patient_id": pid, "allergy": obj["code"]["text"]})
                elif rt == "MedicationRequest":
                    pid = obj["subject"]["reference"].split("/")[-1]
                    meds.append(
                        {
                            "patient_id": pid,
                            "drug": obj["medicationCodeableConcept"]["text"],
                            "date": obj.get("authoredOn"),
                        }
                    )

        out = {
            "fhir_patients": pd.DataFrame(patients),
            "conditions": pd.DataFrame(conds),
            "allergies": pd.DataFrame(alls),
            "prescriptions": pd.DataFrame(meds),
        }

        for k, df in out.items():
            if not df.empty and "patient_id" in df.columns:
                df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)

        return out

    def load_dicom_metadata(self) -> pd.DataFrame:
        """Extrait les métadonnées DICOM depuis les fichiers."""
        dicom_dir = self.data_dir / "source_dicom"
        columns = [
            "dicom_id",
            "patient_id",
            "patient_id_pseudo",
            "study_uid",
            "series_uid",
            "modality",
            "study_date",
            "file_path",
            "rows",
            "cols",
        ]
        if not dicom_dir.exists():
            return pd.DataFrame(columns=columns)

        rows = []
        for path in dicom_dir.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception:
                continue

            patient_id = getattr(ds, "PatientID", None)
            rows.append(
                {
                    "dicom_id": getattr(ds, "SOPInstanceUID", None),
                    "patient_id": patient_id,
                    "patient_id_pseudo": sha256_pseudo(patient_id) if patient_id else None,
                    "study_uid": getattr(ds, "StudyInstanceUID", None),
                    "series_uid": getattr(ds, "SeriesInstanceUID", None),
                    "modality": getattr(ds, "Modality", None),
                    "study_date": getattr(ds, "StudyDate", None),
                    "file_path": str(path),
                    "rows": getattr(ds, "Rows", None),
                    "cols": getattr(ds, "Columns", None),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df

    def integrate(self) -> Dict[str, pd.DataFrame]:
        """Intègre toutes les sources de données en un schéma unifié."""
        ehr = self.load_ehr_csv()
        lab = self.load_lab_json()
        fhir = self.load_fhir_ndjson()
        dicom_images = self.load_dicom_metadata()

        # Unifier la table des patients
        patients = ehr.merge(
            fhir["fhir_patients"][["patient_id", "gender_fhir"]] if not fhir["fhir_patients"].empty else ehr[["patient_id"]],
            on="patient_id",
            how="left",
        )
        patients["gender_unified"] = patients["gender_fhir"].fillna(patients["gender"])
        patients = patients.drop(columns=[c for c in ["gender_fhir"] if c in patients.columns])

        # Conditions -> ajouter label depuis dictionnaire ICD-10
        conditions = fhir["conditions"].copy()
        if not conditions.empty:
            conditions["icd10_label"] = conditions["icd10_code"].map(ICD10).fillna("Unknown ICD-10")
        else:
            conditions = pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "icd10_code", "icd10_label"])

        # Allergies et prescriptions
        allergies = fhir["allergies"].copy()
        if allergies.empty:
            allergies = pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "allergy"])

        prescriptions = fhir["prescriptions"].copy()
        if not prescriptions.empty:
            prescriptions["drug_family"] = prescriptions["drug"].map(DRUG_FAMILY).fillna("UnknownFamily")
        else:
            prescriptions = pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "drug", "date", "drug_family"])

        # Flags d'anomalie pour les labos en utilisant les plages de référence
        ref = {t["name"]: (t["low"], t["high"]) for t in LOINC_TESTS}
        lab["ref_low"] = lab["test_name"].apply(lambda n: ref.get(n, (None, None))[0])
        lab["ref_high"] = lab["test_name"].apply(lambda n: ref.get(n, (None, None))[1])
        lab["is_abnormal"] = (lab["value"] < lab["ref_low"]) | (lab["value"] > lab["ref_high"])

        return {
            "patients": patients,
            "lab_results": lab,
            "conditions": conditions,
            "allergies": allergies,
            "prescriptions": prescriptions,
            "dicom_images": dicom_images,
        }

    def export_to_sqlite(self, tables: Dict[str, pd.DataFrame], db_path: Path) -> None:
        """Exporte les tables intégrées vers SQLite."""
        ensure_dir(db_path.parent)
        conn = sqlite3.connect(str(db_path))
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"✓ Exported integrated DB: {db_path}")
