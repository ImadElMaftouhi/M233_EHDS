"""
EHDS Integration Prototype
- Partie II: Data Integration (>= 3 sources, heterogeneous, ETL, quality)
- Partie III: Semantic Interoperability (RDF + SKOS + OWL-ish + SPARQL)

Run:
  python ehds_integration.py --run-all

Then dashboard:
  streamlit run dashboard.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL, SKOS


# -----------------------------
# Utils
# -----------------------------
def sha256_pseudo(value: str, keep: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:keep]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


# -----------------------------
# Reference dictionaries (toy but realistic)
# -----------------------------
LOINC_TESTS = [
    {"code": "718-7", "name": "Hemoglobin", "unit": "g/dL", "low": 12.0, "high": 17.0},
    {"code": "2160-0", "name": "Creatinine", "unit": "mg/dL", "low": 0.5, "high": 1.5},
    {"code": "2345-7", "name": "Glucose", "unit": "mg/dL", "low": 70.0, "high": 140.0},
    {"code": "6690-2", "name": "WBC", "unit": "10^3/uL", "low": 4.0, "high": 11.0},
]

# ICD-10 mini dictionary (code -> label)
ICD10 = {
    "I10": "Hypertension essentielle",
    "E11": "Diabète de type 2",
    "J45": "Asthme",
}

# Drug -> family (toy)
DRUG_FAMILY = {
    "Amoxicillin": "Penicillins",
    "Penicillin V": "Penicillins",
    "Ibuprofen": "NSAIDs",
    "Aspirin": "NSAIDs",
    "Paracetamol": "Analgesics",
}

# Allergy -> affected family (toy, for contraindication demo)
ALLERGY_FAMILY = {
    "Penicillin": "Penicillins",
    "Aspirin": "NSAIDs",
}


# -----------------------------
# PART 1: Data generation (3 heterogeneous sources)
# -----------------------------
@dataclass
class EHDSDataPreparation:
    data_dir: Path = Path("data")
    seed: int = 42

    def __post_init__(self) -> None:
        random.seed(self.seed)
        ensure_dir(self.data_dir)
        ensure_dir(self.data_dir / "source_ehr_csv")
        ensure_dir(self.data_dir / "source_lab_json")
        ensure_dir(self.data_dir / "source_fhir_ndjson")
        ensure_dir(self.data_dir / "integrated")
        ensure_dir(self.data_dir / "rdf")

    def generate_ehr_patients_csv(self, n_patients: int = 120) -> Path:
        # Source 1: EHR (CSV)
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
        print(f"OK Generated EHR patients CSV: {out} ({len(df)} rows)")
        return out

    def generate_lab_results_json(self, n_records: int = 600, n_patients: int = 120) -> Path:
        # Source 2: Labs (JSON)
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
        print(f"OK Generated Lab JSON: {out} ({len(labs)} rows)")
        return out

    def generate_fhir_like_ndjson(self, n_patients: int = 120) -> Path:
        # Source 3: FHIR-like NDJSON (Patient + Condition + AllergyIntolerance + MedicationRequest)
        # (We keep it simple: just enough to show heterogeneity and semantic linking)
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

        print(f"OK Generated FHIR-like NDJSON: {out}")
        return out


# -----------------------------
# PART 2: Integration pipeline (ETL -> SQLite)
# -----------------------------
class EHDSDataIntegration:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir
        ensure_dir(self.data_dir / "integrated")

    @staticmethod
    def convert_creatinine_to_mgdl(value: float, unit: str) -> Tuple[float, str]:
        # normalize creatinine unit heterogeneity
        if unit == "µmol/L":
            # 1 mg/dL = 88.4 µmol/L
            return round(value / 88.4, 3), "mg/dL"
        return float(value), unit

    def load_ehr_csv(self) -> pd.DataFrame:
        path = self.data_dir / "source_ehr_csv" / "ehr_patients.csv"
        df = pd.read_csv(path)
        df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)
        return df

    def load_lab_json(self) -> pd.DataFrame:
        path = self.data_dir / "source_lab_json" / "lab_results.json"
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
        df["patient_id_pseudo"] = df["patient_id"].apply(sha256_pseudo)

        # unit normalization only for creatinine rows
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

    def integrate(self) -> Dict[str, pd.DataFrame]:
        ehr = self.load_ehr_csv()
        lab = self.load_lab_json()
        fhir = self.load_fhir_ndjson()

        # unify patient table
        patients = ehr.merge(
            fhir["fhir_patients"][["patient_id", "gender_fhir"]] if not fhir["fhir_patients"].empty else ehr[["patient_id"]],
            on="patient_id",
            how="left",
        )
        patients["gender_unified"] = patients["gender_fhir"].fillna(patients["gender"])
        patients = patients.drop(columns=[c for c in ["gender_fhir"] if c in patients.columns])

        # conditions -> add label from ICD-10 dict (translation dictionary)
        conditions = fhir["conditions"].copy()
        if not conditions.empty:
            conditions["icd10_label"] = conditions["icd10_code"].map(ICD10).fillna("Unknown ICD-10")
        else:
            conditions = pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "icd10_code", "icd10_label"])

        # allergies and prescriptions
        allergies = fhir["allergies"].copy()
        if allergies.empty:
            allergies = pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "allergy"])

        prescriptions = fhir["prescriptions"].copy()
        if not prescriptions.empty:
            prescriptions["drug_family"] = prescriptions["drug"].map(DRUG_FAMILY).fillna("UnknownFamily")
        else:
            prescriptions = pd.DataFrame(columns=["patient_id", "patient_id_pseudo", "drug", "date", "drug_family"])

        # lab abnormal flag using reference ranges
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
        }

    def export_to_sqlite(self, tables: Dict[str, pd.DataFrame], db_path: Path) -> None:
        ensure_dir(db_path.parent)
        conn = sqlite3.connect(str(db_path))
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"OK Exported integrated DB: {db_path}")


# -----------------------------
# PART 3: Semantic layer (RDF + SKOS + OWL-ish + SPARQL)
# -----------------------------
class EHDSSemanticLayer:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir
        ensure_dir(self.data_dir / "rdf")
        self.g = Graph()

        self.EHDS = Namespace("http://ehds.eu/ontology#")
        self.RES = Namespace("http://ehds.eu/resource/")
        self.g.bind("ehds", self.EHDS)
        self.g.bind("skos", SKOS)
        self.g.bind("owl", OWL)

        # Ontology header
        ont = URIRef(self.EHDS)
        self.g.add((ont, RDF.type, OWL.Ontology))
        self.g.add((ont, RDFS.label, Literal("EHDS Mini Ontology")))

        # Classes
        for cls in ["Patient", "LabResult", "Test", "Condition", "Drug", "DrugFamily", "Allergy", "Alert"]:
            self.g.add((self.EHDS[cls], RDF.type, OWL.Class))

        # Properties
        props = [
            ("hasPatient", OWL.ObjectProperty),
            ("hasTest", OWL.ObjectProperty),
            ("hasCondition", OWL.ObjectProperty),
            ("hasAllergy", OWL.ObjectProperty),
            ("hasPrescription", OWL.ObjectProperty),
            ("drug", OWL.ObjectProperty),
            ("belongsToFamily", OWL.ObjectProperty),
            ("affectsFamily", OWL.ObjectProperty),
            ("triggersAlert", OWL.ObjectProperty),
            ("value", OWL.DatatypeProperty),
            ("unit", OWL.DatatypeProperty),
            ("loincCode", OWL.DatatypeProperty),
            ("icd10Code", OWL.DatatypeProperty),
            ("label", OWL.DatatypeProperty),
            ("date", OWL.DatatypeProperty),
        ]
        for p, t in props:
            self.g.add((self.EHDS[p], RDF.type, t))

        # SKOS ConceptSchemes (for “dictionary / translation” part)
        self.loinc_scheme = self.RES["scheme/loinc"]
        self.icd_scheme = self.RES["scheme/icd10"]
        self.g.add((self.loinc_scheme, RDF.type, SKOS.ConceptScheme))
        self.g.add((self.icd_scheme, RDF.type, SKOS.ConceptScheme))
        self.g.add((self.loinc_scheme, SKOS.prefLabel, Literal("LOINC mini scheme")))
        self.g.add((self.icd_scheme, SKOS.prefLabel, Literal("ICD-10 mini scheme")))

    def add_reference_concepts(self) -> None:
        # LOINC concepts
        for t in LOINC_TESTS:
            c = self.RES[f"loinc/{t['code']}"]
            self.g.add((c, RDF.type, SKOS.Concept))
            self.g.add((c, SKOS.inScheme, self.loinc_scheme))
            self.g.add((c, SKOS.notation, Literal(t["code"])))
            self.g.add((c, SKOS.prefLabel, Literal(t["name"])))

        # ICD-10 concepts
        for code, label in ICD10.items():
            c = self.RES[f"icd10/{code}"]
            self.g.add((c, RDF.type, SKOS.Concept))
            self.g.add((c, SKOS.inScheme, self.icd_scheme))
            self.g.add((c, SKOS.notation, Literal(code)))
            self.g.add((c, SKOS.prefLabel, Literal(label)))

        # Drug families (simple taxonomy with SKOS broader/narrower demo)
        fam_root = self.RES["drugfamily/Medicines"]
        self.g.add((fam_root, RDF.type, SKOS.Concept))
        self.g.add((fam_root, SKOS.prefLabel, Literal("Medicines")))
        for fam in sorted(set(DRUG_FAMILY.values())):
            furi = self.RES[f"drugfamily/{fam}"]
            self.g.add((furi, RDF.type, SKOS.Concept))
            self.g.add((furi, SKOS.prefLabel, Literal(fam)))
            self.g.add((furi, SKOS.broader, fam_root))
            self.g.add((fam_root, SKOS.narrower, furi))

    def build_graph(
        self,
        patients: pd.DataFrame,
        lab_results: pd.DataFrame,
        conditions: pd.DataFrame,
        allergies: pd.DataFrame,
        prescriptions: pd.DataFrame,
        sample_labs: int = 200,
    ) -> None:
        self.add_reference_concepts()

        # Patients
        for _, r in patients.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            self.g.add((p, RDF.type, self.EHDS.Patient))
            self.g.add((p, self.EHDS.label, Literal(f"{r.get('first_name','')} {r.get('last_name','')}".strip())))

        # Conditions (ICD-10)
        for _, r in conditions.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            cnode = self.RES[f"condition/{sha256_pseudo(r['patient_id'] + r['icd10_code'], 12)}"]
            self.g.add((cnode, RDF.type, self.EHDS.Condition))
            self.g.add((cnode, self.EHDS.icd10Code, Literal(r["icd10_code"])))
            self.g.add((cnode, self.EHDS.label, Literal(r["icd10_label"])))
            self.g.add((p, self.EHDS.hasCondition, cnode))

        # Allergies
        for _, r in allergies.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            anode = self.RES[f"allergy/{sha256_pseudo(r['patient_id'] + r['allergy'], 12)}"]
            self.g.add((anode, RDF.type, self.EHDS.Allergy))
            self.g.add((anode, self.EHDS.label, Literal(r["allergy"])))
            self.g.add((p, self.EHDS.hasAllergy, anode))

            fam = ALLERGY_FAMILY.get(r["allergy"])
            if fam:
                fam_uri = self.RES[f"drugfamily/{fam}"]
                self.g.add((anode, self.EHDS.affectsFamily, fam_uri))

        # Drugs + prescriptions
        for _, r in prescriptions.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            drug_uri = self.RES[f"drug/{r['drug'].replace(' ', '_')}"]
            self.g.add((drug_uri, RDF.type, self.EHDS.Drug))
            self.g.add((drug_uri, self.EHDS.label, Literal(r["drug"])))

            fam = r.get("drug_family", "UnknownFamily")
            fam_uri = self.RES[f"drugfamily/{fam}"]
            self.g.add((drug_uri, self.EHDS.belongsToFamily, fam_uri))

            pr_uri = self.RES[f"prescription/{sha256_pseudo(r['patient_id'] + r['drug'] + str(r.get('date','')), 12)}"]
            self.g.add((pr_uri, RDF.type, self.EHDS.Prescription if hasattr(self.EHDS, "Prescription") else self.EHDS.Drug))
            self.g.add((pr_uri, self.EHDS.drug, drug_uri))
            if r.get("date"):
                self.g.add((pr_uri, self.EHDS.date, Literal(r["date"], datatype=XSD.date)))

            self.g.add((p, self.EHDS.hasPrescription, pr_uri))

        # Lab results (sample to keep TTL small)
        for _, r in lab_results.head(sample_labs).iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            lr = self.RES[f"lab/{r['lab_id']}"]
            self.g.add((lr, RDF.type, self.EHDS.LabResult))
            self.g.add((lr, self.EHDS.hasPatient, p))
            self.g.add((lr, self.EHDS.value, Literal(float(r["value"]), datatype=XSD.float)))
            self.g.add((lr, self.EHDS.unit, Literal(r["unit"])))
            self.g.add((lr, self.EHDS.date, Literal(r["date"], datatype=XSD.date)))
            self.g.add((lr, self.EHDS.loincCode, Literal(r["test_code_loinc"])))
            self.g.add((lr, self.EHDS.label, Literal(r["test_name"])))

            test_uri = self.RES[f"loinc/{r['test_code_loinc']}"]
            self.g.add((test_uri, RDF.type, self.EHDS.Test))
            self.g.add((lr, self.EHDS.hasTest, test_uri))

        print(f"OK RDF graph built: {len(self.g)} triples")

    def save(self, filename: str = "ehds_data.ttl") -> Path:
        out = self.data_dir / "rdf" / filename
        self.g.serialize(destination=str(out), format="turtle")
        print(f"OK Saved TTL: {out}")
        return out

    def run_predefined_queries(self) -> Dict[str, list]:
        # Real SPARQL queries (not simulated)
        queries = {
            "Count patients": """
                PREFIX ehds: <http://ehds.eu/ontology#>
                PREFIX res:  <http://ehds.eu/resource/>
                SELECT (COUNT(DISTINCT ?p) AS ?count)
                WHERE { ?p a ehds:Patient . }
            """,
            "Abnormal glucose (>140 mg/dL)": """
                PREFIX ehds: <http://ehds.eu/ontology#>
                SELECT ?patient ?val ?date
                WHERE {
                    ?lab a ehds:LabResult ;
                         ehds:hasPatient ?patient ;
                         ehds:label "Glucose" ;
                         ehds:value ?val ;
                         ehds:date ?date .
                    FILTER (?val > 140)
                }
                ORDER BY DESC(?val)
                LIMIT 10
            """,
            "Contraindication alerts (allergy <-> drug family)": """
                PREFIX ehds: <http://ehds.eu/ontology#>
                SELECT ?patient ?allergyLabel ?drugLabel ?family
                WHERE {
                    ?patient a ehds:Patient ;
                             ehds:hasAllergy ?a ;
                             ehds:hasPrescription ?pr .
                    ?a ehds:label ?allergyLabel ;
                       ehds:affectsFamily ?family .
                    ?pr ehds:drug ?d .
                    ?d ehds:label ?drugLabel ;
                       ehds:belongsToFamily ?family .
                }
                LIMIT 20
            """,
        }

        results = {}
        for name, q in queries.items():
            rows = list(self.g.query(q))
            results[name] = rows
            print(f"\n[Query] {name}: {len(rows)} rows")
        return results


# -----------------------------
# DEMO / CLI
# -----------------------------
def run_all(data_dir: Path, n_patients: int, n_labs: int) -> None:
    prep = EHDSDataPreparation(data_dir=data_dir)
    prep.generate_ehr_patients_csv(n_patients=n_patients)
    prep.generate_lab_results_json(n_records=n_labs, n_patients=n_patients)
    prep.generate_fhir_like_ndjson(n_patients=n_patients)

    integrator = EHDSDataIntegration(data_dir=data_dir)
    tables = integrator.integrate()
    db_path = data_dir / "integrated" / "ehds.db"
    integrator.export_to_sqlite(tables, db_path=db_path)

    semantic = EHDSSemanticLayer(data_dir=data_dir)
    semantic.build_graph(
        patients=tables["patients"],
        lab_results=tables["lab_results"],
        conditions=tables["conditions"],
        allergies=tables["allergies"],
        prescriptions=tables["prescriptions"],
        sample_labs=min(200, len(tables["lab_results"])),
    )
    semantic.save("ehds_data.ttl")
    semantic.run_predefined_queries()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-patients", type=int, default=120)
    parser.add_argument("--n-labs", type=int, default=600)
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all(Path(args.data_dir), args.n_patients, args.n_labs)
    else:
        print("Use: python ehds_integration.py --run-all")


if __name__ == "__main__":
    main()
