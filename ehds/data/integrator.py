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
        """
        Charge les données EHR depuis un fichier CSV (généré).
        
        NOTE: Cette méthode est conservée pour usage futur mais n'est plus utilisée
        automatiquement. Utilisez load_mimic_patients_csv() pour les données MIMIC-III.
        """
        path = self.data_dir / "source_ehr_csv" / "ehr_patients.csv"
        if not path.exists():
            return pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "first_name", "last_name", "gender", "birthDate", "country"])
        df = pd.read_csv(path)
        if "patient_id" in df.columns:
            df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)
        return df

    def load_mimic_patients_csv(self) -> pd.DataFrame:
        """
        Charge les données patients depuis MIMIC-III PATIENTS.csv.
        
        MIMIC-III PATIENTS.csv contient:
        - SUBJECT_ID: identifiant patient
        - GENDER: genre (M/F)
        - DOB: date de naissance
        - DOD: date de décès (optionnel)
        """
        mimic_source = self.data_dir / "source_mimic_csv"
        if not mimic_source.exists():
            return pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "gender", "birthDate", "country"])
        
        # Find PATIENTS.CSV (case insensitive)
        patients_file = None
        for f in mimic_source.glob("*.csv"):
            if f.name.upper() == "PATIENTS.CSV":
                patients_file = f
                break
        
        if patients_file is None:
            return pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "gender", "birthDate", "country"])
        
        try:
            # MIMIC-III CSV files may have encoding issues, try multiple encodings
            encodings = ["utf-8", "latin-1", "cp1252"]
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(patients_file, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                # Last resort: use replace strategy
                df = pd.read_csv(patients_file, encoding="utf-8", errors="replace", low_memory=False)
            
            # Map MIMIC-III columns to our schema
            # SUBJECT_ID -> patient_id
            if "SUBJECT_ID" in df.columns:
                df = df.rename(columns={"SUBJECT_ID": "patient_id"})
            elif "subject_id" in df.columns:
                df = df.rename(columns={"subject_id": "patient_id"})
            else:
                # Use first column as fallback
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "patient_id"})
            
            # Ensure patient_id is string
            df["patient_id"] = df["patient_id"].astype(str)
            
            # GENDER -> gender (normalize to lowercase: M->male, F->female)
            if "GENDER" in df.columns:
                df["gender"] = df["GENDER"].map({"M": "male", "F": "female", "m": "male", "f": "female"}).fillna(df["GENDER"].str.lower())
            elif "gender" in df.columns:
                df["gender"] = df["gender"].map({"M": "male", "F": "female", "m": "male", "f": "female"}).fillna(df["gender"].str.lower())
            else:
                df["gender"] = None
            
            # DOB -> birthDate
            if "DOB" in df.columns:
                df["birthDate"] = pd.to_datetime(df["DOB"], errors="coerce").dt.date
            elif "dob" in df.columns:
                df["birthDate"] = pd.to_datetime(df["dob"], errors="coerce").dt.date
            else:
                df["birthDate"] = None
            
            # Add missing columns for compatibility
            if "first_name" not in df.columns:
                df["first_name"] = None
            if "last_name" not in df.columns:
                df["last_name"] = None
            if "country" not in df.columns:
                df["country"] = "US"  # MIMIC-III is from US hospitals
            
            # Pseudonymize
            df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)
            
            # Select and reorder columns for consistency
            columns_order = ["patient_id", "patient_id_pseudo", "first_name", "last_name", "gender", "birthDate", "country"]
            available_columns = [c for c in columns_order if c in df.columns]
            df = df[available_columns]
            
            print(f"✓ Loaded {len(df)} MIMIC-III patients from {patients_file}")
            return df
            
        except Exception as e:
            print(f"[WARNING] Error loading MIMIC-III patients: {e}")
            return pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "gender", "birthDate", "country"])

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

        # Try reading the whole file as binary and decode with multiple encoding fallbacks
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        skipped_lines = 0
        total_lines = 0
        
        with open(path, "rb") as f:
            lineno = 0
            for rawline in f:
                lineno += 1
                total_lines += 1
                line = None
                
                # Try multiple encodings
                for encoding in encodings:
                    try:
                        line = rawline.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings failed, try with error handling
                if line is None:
                    try:
                        # Use 'replace' to replace invalid bytes with replacement character
                        line = rawline.decode("utf-8", errors="replace")
                        if skipped_lines == 0:  # Only warn once
                            print(f"[WARNING] Some lines in {path} contain invalid UTF-8 bytes. Using replacement characters ().")
                    except Exception as e:
                        skipped_lines += 1
                        if skipped_lines <= 5:
                            print(f"[WARNING] Could not decode line {lineno} of {path}: {e}. Skipping line.")
                        continue
                
                if not line.strip():
                    continue
                    
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    if skipped_lines <= 5:
                        print(f"[WARNING] JSON decoding error in line {lineno} of {path}: {e}. Skipping line.")
                    continue
                rt = obj.get("resourceType")
                if rt == "Patient":
                    pid = obj.get("id")
                    patients.append(
                        {
                            "patient_id": pid,
                            "gender_fhir": obj.get("gender"),
                        }
                    )
                elif rt == "Condition":
                    try:
                        pid = obj.get("subject", {}).get("reference", "").split("/")[-1]
                        coding = obj.get("code", {}).get("coding", [{}])
                        code = coding[0].get("code", "") if coding else ""
                        if pid and code:
                            conds.append({"patient_id": pid, "icd10_code": code})
                    except (KeyError, IndexError, AttributeError, TypeError):
                        pass
                elif rt == "AllergyIntolerance":
                    try:
                        pid = obj.get("patient", {}).get("reference", "").split("/")[-1]
                        allergy = obj.get("code", {}).get("text", "") or obj.get("code", {}).get("coding", [{}])[0].get("display", "")
                        if pid and allergy:
                            alls.append({"patient_id": pid, "allergy": allergy})
                    except (KeyError, IndexError, AttributeError, TypeError):
                        pass
                elif rt == "MedicationRequest":
                    try:
                        pid = obj.get("subject", {}).get("reference", "").split("/")[-1]
                        # Try different locations for medication info
                        drug = (
                            obj.get("medicationCodeableConcept", {}).get("text", "")
                            or obj.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "")
                            or obj.get("medicationReference", {}).get("display", "")
                            or "Unknown"
                        )
                        if pid:
                            meds.append(
                                {
                                    "patient_id": pid,
                                    "drug": drug,
                                    "date": obj.get("authoredOn"),
                                }
                            )
                    except (KeyError, IndexError, AttributeError, TypeError):
                        pass
        
        # Report summary
        if skipped_lines > 0:
            print(f"[INFO] Processed {total_lines} lines, skipped {skipped_lines} lines with errors.")
        else:
            print(f"[INFO] Successfully processed {total_lines} lines from {path}")

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
        """
        Intègre toutes les sources de données en un schéma unifié.
        
        Utilise MIMIC-III comme source principale de données patients.
        L'EHR généré (load_ehr_csv) est conservé mais non utilisé automatiquement.
        """
        # Try MIMIC-III first, fallback to EHR if not available
        mimic_patients = self.load_mimic_patients_csv()
        if mimic_patients.empty or len(mimic_patients) == 0:
            print("[INFO] MIMIC-III patients not found, trying EHR CSV...")
            ehr = self.load_ehr_csv()
            patients = ehr.copy()
        else:
            patients = mimic_patients.copy()
        
        lab = self.load_lab_json()
        fhir = self.load_fhir_ndjson()
        dicom_images = self.load_dicom_metadata()

        # Unifier la table des patients avec données FHIR si disponibles
        if not fhir["fhir_patients"].empty and "patient_id" in patients.columns:
            # Merge FHIR gender data if available
            fhir_patients_subset = fhir["fhir_patients"][["patient_id", "gender_fhir"]].copy()
            patients = patients.merge(
                fhir_patients_subset,
                on="patient_id",
                how="left",
            )
            # Use FHIR gender if available, otherwise use existing gender
            if "gender_fhir" in patients.columns:
                patients["gender_unified"] = patients["gender_fhir"].fillna(patients.get("gender", ""))
                patients = patients.drop(columns=["gender_fhir"])
            else:
                patients["gender_unified"] = patients.get("gender", "")
        else:
            # No FHIR data, use existing gender
            patients["gender_unified"] = patients.get("gender", "")

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
