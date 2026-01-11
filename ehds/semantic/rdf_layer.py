"""Couche RDF pour transformation sémantique des données."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL, SKOS

from ..utils import LOINC_TESTS, ICD10, DRUG_FAMILY, ALLERGY_FAMILY, sha256_pseudo, ensure_dir
from .namespaces import EHDS


class RDFLayer:
    """Couche RDF pour transformation sémantique des données intégrées."""

    def __init__(self, data_dir: Path = Path("data")) -> None:
        """Initialise la couche RDF avec l'ontologie EHDS."""
        self.data_dir = data_dir
        ensure_dir(self.data_dir / "rdf")
        self.g = Graph()

        self.RES = Namespace("http://ehds.eu/resource/")
        self.g.bind("ehds", EHDS)
        self.g.bind("skos", SKOS)
        self.g.bind("owl", OWL)

        # Ontology header
        ont = URIRef(EHDS)
        self.g.add((ont, RDF.type, OWL.Ontology))
        self.g.add((ont, RDFS.label, Literal("EHDS Mini Ontology")))

        # Classes
        for cls in [
            "Patient",
            "LabResult",
            "Test",
            "Condition",
            "Drug",
            "DrugFamily",
            "Allergy",
            "Alert",
            "ImagingStudy",
            "DicomInstance",
        ]:
            self.g.add((EHDS[cls], RDF.type, OWL.Class))

        # Properties
        props = [
            ("hasPatient", OWL.ObjectProperty),
            ("hasTest", OWL.ObjectProperty),
            ("hasCondition", OWL.ObjectProperty),
            ("hasAllergy", OWL.ObjectProperty),
            ("hasPrescription", OWL.ObjectProperty),
            ("hasImagingStudy", OWL.ObjectProperty),
            ("hasDicomInstance", OWL.ObjectProperty),
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
            ("modality", OWL.DatatypeProperty),
            ("studyUID", OWL.DatatypeProperty),
            ("seriesUID", OWL.DatatypeProperty),
            ("sopUID", OWL.DatatypeProperty),
            ("filePath", OWL.DatatypeProperty),
        ]
        for p, t in props:
            self.g.add((EHDS[p], RDF.type, t))

        # SKOS ConceptSchemes
        self.loinc_scheme = self.RES["scheme/loinc"]
        self.icd_scheme = self.RES["scheme/icd10"]
        self.g.add((self.loinc_scheme, RDF.type, SKOS.ConceptScheme))
        self.g.add((self.icd_scheme, RDF.type, SKOS.ConceptScheme))
        self.g.add((self.loinc_scheme, SKOS.prefLabel, Literal("LOINC mini scheme")))
        self.g.add((self.icd_scheme, SKOS.prefLabel, Literal("ICD-10 mini scheme")))

    def add_reference_concepts(self) -> None:
        """Ajoute les concepts de référence (LOINC, ICD-10, familles de médicaments)."""
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
        dicom_images: pd.DataFrame,
        sample_labs: int = 200,
    ) -> None:
        """Construit le graphe RDF à partir des données intégrées."""
        self.add_reference_concepts()

        # Patients
        for _, r in patients.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            self.g.add((p, RDF.type, EHDS.Patient))
            self.g.add((p, EHDS.label, Literal(f"{r.get('first_name','')} {r.get('last_name','')}".strip())))

        # DICOM imaging (study + instance)
        if not dicom_images.empty:
            for _, r in dicom_images.iterrows():
                patient_pseudo = r.get("patient_id_pseudo")
                study_uid = r.get("study_uid")
                sop_uid = r.get("dicom_id")
                if not patient_pseudo or not study_uid or not sop_uid:
                    continue
                p = self.RES[f"patient/{patient_pseudo}"]
                if (p, RDF.type, EHDS.Patient) not in self.g:
                    self.g.add((p, RDF.type, EHDS.Patient))

                study_uid = str(study_uid)
                study_uri = self.RES[f"imagingstudy/{study_uid}"]
                self.g.add((study_uri, RDF.type, EHDS.ImagingStudy))
                self.g.add((study_uri, EHDS.studyUID, Literal(study_uid)))
                study_date = r.get("study_date")
                if study_date:
                    if isinstance(study_date, str) and len(study_date) == 8:
                        study_date = f"{study_date[0:4]}-{study_date[4:6]}-{study_date[6:8]}"
                    self.g.add((study_uri, EHDS.date, Literal(study_date, datatype=XSD.date)))
                self.g.add((p, EHDS.hasImagingStudy, study_uri))

                inst_uid = str(sop_uid)
                inst_uri = self.RES[f"dicom/{inst_uid}"]
                self.g.add((inst_uri, RDF.type, EHDS.DicomInstance))
                self.g.add((inst_uri, EHDS.hasPatient, p))
                if r.get("series_uid"):
                    self.g.add((inst_uri, EHDS.seriesUID, Literal(str(r["series_uid"]))))
                if r.get("dicom_id"):
                    self.g.add((inst_uri, EHDS.sopUID, Literal(str(r["dicom_id"]))))
                if r.get("modality"):
                    self.g.add((inst_uri, EHDS.modality, Literal(r["modality"])))
                if r.get("file_path"):
                    self.g.add((inst_uri, EHDS.filePath, Literal(r["file_path"])))
                self.g.add((study_uri, EHDS.hasDicomInstance, inst_uri))

        # Conditions (ICD-10)
        for _, r in conditions.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            cnode = self.RES[f"condition/{sha256_pseudo(r['patient_id'] + r['icd10_code'], 12)}"]
            self.g.add((cnode, RDF.type, EHDS.Condition))
            self.g.add((cnode, EHDS.icd10Code, Literal(r["icd10_code"])))
            self.g.add((cnode, EHDS.label, Literal(r["icd10_label"])))
            self.g.add((p, EHDS.hasCondition, cnode))

        # Allergies
        for _, r in allergies.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            anode = self.RES[f"allergy/{sha256_pseudo(r['patient_id'] + r['allergy'], 12)}"]
            self.g.add((anode, RDF.type, EHDS.Allergy))
            self.g.add((anode, EHDS.label, Literal(r["allergy"])))
            self.g.add((p, EHDS.hasAllergy, anode))

            fam = ALLERGY_FAMILY.get(r["allergy"])
            if fam:
                fam_uri = self.RES[f"drugfamily/{fam}"]
                self.g.add((anode, EHDS.affectsFamily, fam_uri))

        # Drugs + prescriptions
        for _, r in prescriptions.iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            # URL-encode drug name to handle special characters (e.g., {, }, (, ), /, +, [, ])
            drug_name_encoded = quote(str(r['drug']), safe='')
            drug_uri = self.RES[f"drug/{drug_name_encoded}"]
            self.g.add((drug_uri, RDF.type, EHDS.Drug))
            self.g.add((drug_uri, EHDS.label, Literal(r["drug"])))

            fam = r.get("drug_family", "UnknownFamily")
            fam_uri = self.RES[f"drugfamily/{fam}"]
            self.g.add((drug_uri, EHDS.belongsToFamily, fam_uri))

            pr_uri = self.RES[f"prescription/{sha256_pseudo(r['patient_id'] + r['drug'] + str(r.get('date','')), 12)}"]
            self.g.add((pr_uri, RDF.type, EHDS.Drug))  # Simplified
            self.g.add((pr_uri, EHDS.drug, drug_uri))
            if r.get("date"):
                self.g.add((pr_uri, EHDS.date, Literal(r["date"], datatype=XSD.date)))

            self.g.add((p, EHDS.hasPrescription, pr_uri))

        # Lab results (sample to keep TTL small)
        for _, r in lab_results.head(sample_labs).iterrows():
            p = self.RES[f"patient/{r['patient_id_pseudo']}"]
            lr = self.RES[f"lab/{r['lab_id']}"]
            self.g.add((lr, RDF.type, EHDS.LabResult))
            self.g.add((lr, EHDS.hasPatient, p))
            self.g.add((lr, EHDS.value, Literal(float(r["value"]), datatype=XSD.float)))
            self.g.add((lr, EHDS.unit, Literal(r["unit"])))
            self.g.add((lr, EHDS.date, Literal(r["date"], datatype=XSD.date)))
            self.g.add((lr, EHDS.loincCode, Literal(r["test_code_loinc"])))
            self.g.add((lr, EHDS.label, Literal(r["test_name"])))

            test_uri = self.RES[f"loinc/{r['test_code_loinc']}"]
            self.g.add((test_uri, RDF.type, EHDS.Test))
            self.g.add((lr, EHDS.hasTest, test_uri))

        print(f"✓ RDF graph built: {len(self.g)} triples")

    def save(self, filename: str = "ehds_data.ttl") -> Path:
        """Sauvegarde le graphe RDF au format Turtle."""
        out = self.data_dir / "rdf" / filename
        self.g.serialize(destination=str(out), format="turtle")
        print(f"✓ Saved TTL: {out}")
        return out

    def run_predefined_queries(self) -> Dict[str, List]:
        """Exécute des requêtes SPARQL prédéfinies sur le graphe."""
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
            "Count DICOM instances per modality": """
                PREFIX ehds: <http://ehds.eu/ontology#>
                SELECT ?modality (COUNT(?dicom) AS ?count)
                WHERE {
                    ?dicom a ehds:DicomInstance ;
                           ehds:modality ?modality .
                }
                GROUP BY ?modality
                ORDER BY DESC(?count)
            """,
            "Patients with imaging + abnormal glucose": """
                PREFIX ehds: <http://ehds.eu/ontology#>
                SELECT DISTINCT ?patient ?val ?study
                WHERE {
                    ?patient a ehds:Patient ;
                             ehds:hasImagingStudy ?study .
                    ?lab a ehds:LabResult ;
                         ehds:hasPatient ?patient ;
                         ehds:label "Glucose" ;
                         ehds:value ?val .
                    FILTER (?val > 140)
                }
                LIMIT 20
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
