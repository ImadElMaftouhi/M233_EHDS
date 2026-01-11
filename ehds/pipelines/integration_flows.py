"""Flux d'intégration ELT pour différentes sources de données."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rdflib import Literal, URIRef
from rdflib.namespace import RDF, XSD

from ..data import DataGenerator, DataIntegrator
from ..semantic import CATALOG, PROV_O, SemanticDataLake
from ..utils import ensure_dir


class IntegrationFlow:
    """Classe de base pour les flux d'intégration ELT."""

    def __init__(self, lake: SemanticDataLake, flow_name: str):
        """Initialise un flux d'intégration."""
        self.lake = lake
        self.flow_name = flow_name

    def extract(self) -> Path:
        """Extract: Génération/extraction des données brutes."""
        raise NotImplementedError

    def load_bronze(self, source_path: Path) -> URIRef:
        """Load Bronze: Copie brute + métadonnées RDF (PROV-O)."""
        bronze_uri = self.lake.ingest_raw(
            source_name=f"{self.flow_name}_raw",
            file_path=source_path,
            domain=self.flow_name,
            format_type=self._detect_format(source_path),
        )
        return bronze_uri

    def transform_silver(self, bronze_uri: URIRef) -> URIRef:
        """Transform Silver: Enrichissement, normalisation, qualité."""
        raise NotImplementedError

    def transform_gold(self, silver_uris: List[URIRef]) -> URIRef:
        """Transform Gold: Schéma unifié."""
        raise NotImplementedError

    def transform_rdf(self, gold_uri: URIRef):
        """Transform RDF: Transformation sémantique."""
        raise NotImplementedError

    def _detect_format(self, path: Path) -> str:
        """Détecte le format d'un fichier."""
        if path.is_dir():
            return "dicom"
        ext = path.suffix.lower()
        return ext[1:] if ext else "unknown"

    def run_full_flow(self) -> Dict[str, URIRef]:
        """Exécute le flux complet ELT."""
        print(f"\n{'='*60}")
        print(f"Flux: {self.flow_name}")
        print(f"{'='*60}")

        # Extract
        print(f"[1/5] Extract...")
        source_path = self.extract()

        # Load Bronze
        print(f"[2/5] Load Bronze...")
        bronze_uri = self.load_bronze(source_path)

        # Transform Silver
        print(f"[3/5] Transform Silver...")
        silver_uri = self.transform_silver(bronze_uri)

        # Transform Gold
        print(f"[4/5] Transform Gold...")
        gold_uri = self.transform_gold([silver_uri])

        # Transform RDF
        print(f"[5/5] Transform RDF...")
        self.transform_rdf(gold_uri)

        print(f"✓ Flux {self.flow_name} terminé\n")

        return {
            "bronze": bronze_uri,
            "silver": silver_uri,
            "gold": gold_uri,
        }


class SyntheaFHIRFlow(IntegrationFlow):
    """Flux 1 – Synthea (FHIR, usage primaire)."""

    def __init__(self, lake: SemanticDataLake, n_patients: int = 100):
        super().__init__(lake, "synthea_fhir")
        self.n_patients = n_patients

    def extract(self) -> Path:
        """Extract: Utilise données FHIR existantes dans data/source_fhir_ndjson/."""
        path = self.lake.data_dir / "source_fhir_ndjson" / "bundle.ndjson"
        if not path.exists():
            raise FileNotFoundError(
                f"FHIR data not found at {path}. "
                "Please copy Synthea FHIR bundle.ndjson to data/source_fhir_ndjson/"
                "Check the documentation for Synthea Guide"
            )
        return path

    def transform_silver(self, bronze_uri: URIRef) -> URIRef:
        """
        Silver: Pseudonymisation, normalisation dates ISO, terminologies LOINC, flags qualité.
        
        Note: NDJSON FHIR ne peut pas être directement converti en Parquet car il contient
        des structures imbriquées. On utilise DataIntegrator pour extraire les données structurées.
        """
        from datetime import datetime
        from rdflib import Literal
        from rdflib.namespace import RDF
        
        # Utiliser DataIntegrator pour charger et transformer les données FHIR
        integrator = DataIntegrator(data_dir=self.lake.data_dir)
        fhir_data = integrator.load_fhir_ndjson()
        
        # Créer un DataFrame unifié à partir des données FHIR extraites
        import pandas as pd
        
        silver_records = []
        
        # Patients
        if not fhir_data["fhir_patients"].empty:
            for _, row in fhir_data["fhir_patients"].iterrows():
                silver_records.append({
                    "patient_id": row.get("patient_id", ""),
                    "patient_id_pseudo": row.get("patient_id_pseudo", ""),
                    "gender": row.get("gender_fhir", ""),
                    "resource_type": "Patient",
                })
        
        # Conditions
        if not fhir_data["conditions"].empty:
            for _, row in fhir_data["conditions"].iterrows():
                silver_records.append({
                    "patient_id": row.get("patient_id", ""),
                    "patient_id_pseudo": row.get("patient_id_pseudo", ""),
                    "icd10_code": row.get("icd10_code", ""),
                    "resource_type": "Condition",
                })
        
        # Allergies
        if not fhir_data["allergies"].empty:
            for _, row in fhir_data["allergies"].iterrows():
                silver_records.append({
                    "patient_id": row.get("patient_id", ""),
                    "patient_id_pseudo": row.get("patient_id_pseudo", ""),
                    "allergy": row.get("allergy", ""),
                    "resource_type": "AllergyIntolerance",
                })
        
        # Prescriptions
        if not fhir_data["prescriptions"].empty:
            for _, row in fhir_data["prescriptions"].iterrows():
                silver_records.append({
                    "patient_id": row.get("patient_id", ""),
                    "patient_id_pseudo": row.get("patient_id_pseudo", ""),
                    "drug": row.get("drug", ""),
                    "date": row.get("date", ""),
                    "resource_type": "MedicationRequest",
                })
        
        if not silver_records:
            # Fallback: retourner l'URI bronze si aucune donnée
            return bronze_uri
        
        # Créer DataFrame et sauvegarder en Silver
        df = pd.DataFrame(silver_records)
        
        # Ajouter tags sémantiques si nécessaire
        if "icd10_code" in df.columns:
            from ..utils import ICD10
            df["icd10_label"] = df["icd10_code"].map(ICD10).fillna("Unknown ICD-10")
        
        # Sauvegarder en Parquet
        silver_domain_dir = self.lake.silver / "synthea_fhir"
        ensure_dir(silver_domain_dir)
        silver_path = silver_domain_dir / f"synthea_fhir_enriched_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(silver_path, index=False, engine="pyarrow")
        
        # Enregistrer dans le catalogue
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        silver_uri = CATALOG[f"dataset/synthea_fhir_silver_{timestamp}"]
        self.lake.catalog.add((silver_uri, RDF.type, CATALOG.Dataset))
        self.lake.catalog.add((silver_uri, CATALOG.zone, Literal("silver")))
        self.lake.catalog.add((silver_uri, CATALOG.domain, Literal("synthea_fhir")))
        self.lake.catalog.add((silver_uri, CATALOG.location, Literal(str(silver_path))))
        self.lake.catalog.add((silver_uri, PROV_O.wasDerivedFrom, bronze_uri))
        self.lake.catalog.add((silver_uri, CATALOG["format"], Literal("parquet")))
        
        # Métadonnées qualité
        completeness = float((df.notnull().sum() / len(df) * 100).mean()) if len(df) > 0 else 0.0
        self.lake.catalog.add((silver_uri, CATALOG.completeness, Literal(completeness, datatype=XSD.float)))
        self.lake.catalog.add((silver_uri, CATALOG.rowCount, Literal(len(df), datatype=XSD.integer)))
        
        print(f"✓ Enriched to Silver → {silver_path} ({len(df)} rows, {completeness:.1f}% completeness)")
        return silver_uri

    def transform_gold(self, silver_uris: List[URIRef]) -> URIRef:
        """Gold: Jointure Patient+Observation, schéma unifié."""
        integrator = DataIntegrator(data_dir=self.lake.data_dir)
        fhir_data = integrator.load_fhir_ndjson()

        unified_schema = {
            "patient_id_pseudo": "string",
            "resource_type": "string",
            "observation_date": "date",
            "code": "string",
            "value": "float",
        }

        # Construire dataset depuis FHIR
        import pandas as pd
        from ..utils import sha256_pseudo

        gold_records = []
        fhir_patients_df = fhir_data.get("fhir_patients", pd.DataFrame())
        if not fhir_patients_df.empty:
            for _, patient_row in fhir_patients_df.iterrows():
                pid_pseudo = sha256_pseudo(patient_row.get("patient_id", ""))
                gold_records.append(
                    {
                        "patient_id_pseudo": pid_pseudo,
                        "resource_type": "Patient",
                        "resource_date": None,
                        "code": None,
                        "value": None,
                    }
                )

        if gold_records:
            gold_df = pd.DataFrame(gold_records)
            gold_path = self.lake.gold / f"synthea_fhir_unified_{datetime.now().strftime('%Y%m%d')}.parquet"
            gold_df.to_parquet(gold_path, index=False)

            # Enregistrer dans catalogue
            gold_uri = CATALOG[f"dataset/synthea_fhir_gold_{datetime.now().strftime('%Y%m%d')}"]
            self.lake.catalog.add((gold_uri, RDF.type, CATALOG.Dataset))
            self.lake.catalog.add((gold_uri, CATALOG.zone, Literal("gold")))
            self.lake.catalog.add((gold_uri, CATALOG.location, Literal(str(gold_path))))
            return gold_uri

        return silver_uris[0]  # Fallback

    def transform_rdf(self, gold_uri: URIRef):
        """RDF: Convert FHIR→RDF avec ontologies HL7 FHIR RDF."""
        # Le graphe RDF est déjà construit dans le pipeline principal
        pass


class MIMICIIIFlow(IntegrationFlow):
    """Flux 2 – MIMIC-III (CSV, usage secondaire)."""

    def __init__(self, lake: SemanticDataLake, mimic_dir: Optional[Path] = None):
        super().__init__(lake, "mimic")
        self.mimic_dir = mimic_dir or lake.data_dir / "bronze" / "mimic"

    def extract(self) -> Path:
        """
        Extract: Utilise les données MIMIC-III existantes dans data/source_mimic_csv/.

        Quatre fichiers CSV doivent être présents (copiés manuellement) :
        - ADMISSIONS.csv
        - LABEVENTS.csv
        - PATIENTS.csv
        - PRESCRIPTIONS.csv
        """
        mimic_source = self.lake.data_dir / "source_mimic_csv"
        if not mimic_source.exists():
            raise FileNotFoundError(
                f"MIMIC-III data directory not found at {mimic_source}. "
                "Please copy the following CSV files (ADMISSIONS.csv, LABEVENTS.csv, PATIENTS.csv, PRESCRIPTIONS.csv) to data/source_mimic_csv/"
            )

        required_files = [
            "ADMISSIONS.CSV",
            "LABEVENTS.CSV",
            "PATIENTS.CSV",
            "PRESCRIPTIONS.CSV",
        ]
        actual_files = {f.name.upper(): f for f in mimic_source.glob("*.csv") if f.is_file()}
        missing = [f for f in required_files if f not in actual_files]
        if missing:
            raise FileNotFoundError(
                f"Missing MIMIC-III files in {mimic_source}: {', '.join(missing)}. "
                f"Found files: {list(actual_files.keys())}"
            )

        # On retourne le chemin de PATIENTS.csv (par convention, peut être utilisé dans la suite du pipeline)
        return actual_files["PATIENTS.CSV"]

    def transform_silver(self, bronze_uri: URIRef) -> URIRef:
        """Silver: Nettoyage, pseudonymisation subject_id, mapping LOINC/ICD-10, normalisation unités."""
        enrichment_rules = {
            "pseudonymize": "patient_id",
            "clean_data": True,
            "map_terminologies": {"loinc": True, "icd10": True},
            "normalize_units": {"creatinine": "mg/dL"},
            "quality_flags": True,
        }
        return self.lake.enrich_to_silver(bronze_uri, enrichment_rules)

    def transform_gold(self, silver_uris: List[URIRef]) -> URIRef:
        """Gold: Jointure tables, schéma unifié (admission_id, lab_itemid, value, flag)."""
        return self.lake.curate_to_gold(
            silver_uris,
            {
                "patient_id_pseudo": "string",
                "admission_id": "string",
                "lab_itemid": "string",
                "value": "float",
                "flag": "string",
            },
        )

    def transform_rdf(self, gold_uri: URIRef):
        """RDF: Triples EHDS avec ontologies LOINC, SPARQL pour glucose >140."""
        pass


class LabSimulatedFlow(IntegrationFlow):
    """Flux 3 – Données Labo Simulées (CSV/JSON)."""

    def __init__(self, lake: SemanticDataLake, n_records: int = 600):
        super().__init__(lake, "lab_simulated")
        self.n_records = n_records
        self.generator = DataGenerator(data_dir=lake.data_dir)

    def extract(self) -> Path:
        """Extract: Crée fichiers simulés (random data)."""
        path = self.generator.generate_lab_results_json(n_records=self.n_records, n_patients=120)
        return path

    def transform_silver(self, bronze_uri: URIRef) -> URIRef:
        """Silver: Normalise unités, tags SKOS, impute manquants, calcul complétude."""
        enrichment_rules = {
            "pseudonymize": "patient_id",
            "normalize_units": {"creatinine": "mg/dL"},
            "semantic_tags": {"skos": True, "loinc": True},
            "impute_missing": True,
            "quality_checks": True,
        }
        return self.lake.enrich_to_silver(bronze_uri, enrichment_rules)

    def transform_gold(self, silver_uris: List[URIRef]) -> URIRef:
        """Gold: Intégration à schéma unifié (merge avec autres sources)."""
        return self.lake.curate_to_gold(
            silver_uris,
            {
                "patient_id_pseudo": "string",
                "test_name": "string",
                "value": "float",
                "unit": "string",
            },
        )

    def transform_rdf(self, gold_uri: URIRef):
        """RDF: Ajout graphe (ehds:hasTest), liens SKOS/OWL, SPARQL cross-source."""
        pass


class DICOMFlow(IntegrationFlow):
    """Flux 4 – Imagerie DICOM (Synthea Coherent)."""

    def __init__(self, lake: SemanticDataLake, n_studies: int = 150):
        super().__init__(lake, "dicom")
        self.n_studies = n_studies
        self.generator = DataGenerator(data_dir=lake.data_dir)

    def extract(self) -> Path:
        """Extract: Génère .dcm (Synthea coherent)."""
        path = self.generator.generate_dicom_series(n_patients=120, n_studies=self.n_studies)
        return path

    def transform_silver(self, bronze_uri: URIRef) -> URIRef:
        """Silver: Extract métadonnées pydicom, pseudonymisation, tags DICOM→LOINC."""
        integrator = DataIntegrator(data_dir=self.lake.data_dir)
        dicom_metadata = integrator.load_dicom_metadata()

        # Sauvegarder en Silver
        silver_path = (
            self.lake.silver / "dicom" / f"dicom_metadata_enriched_{datetime.now().strftime('%Y%m%d')}.parquet"
        )
        ensure_dir(silver_path.parent)
        dicom_metadata.to_parquet(silver_path, index=False)

        # Enregistrer dans catalogue
        silver_uri = CATALOG[f"dataset/dicom_silver_{datetime.now().strftime('%Y%m%d')}"]
        self.lake.catalog.add((silver_uri, RDF.type, CATALOG.Dataset))
        self.lake.catalog.add((silver_uri, CATALOG.zone, Literal("silver")))
        self.lake.catalog.add((silver_uri, CATALOG.location, Literal(str(silver_path))))
        self.lake.catalog.add((silver_uri, PROV_O.wasDerivedFrom, bronze_uri))
        self.lake.catalog.add((silver_uri, CATALOG["format"], Literal("parquet")))

        return silver_uri

    def transform_gold(self, silver_uris: List[URIRef]) -> URIRef:
        """Gold: Liaison patients, schéma (study_id, modality, date)."""
        return self.lake.curate_to_gold(
            silver_uris,
            {
                "patient_id_pseudo": "string",
                "study_id": "string",
                "modality": "string",
                "date": "date",
            },
        )

    def transform_rdf(self, gold_uri: URIRef):
        """RDF: Métadonnées triples (:image a ehds:ImagingStudy), SPARQL pour modality."""
        pass


def run_all_flows(data_dir: Path = Path("data"), flow_names: Optional[List[str]] = None) -> Dict:
    """
    Exécute tous les flux d'intégration.

    Args:
        data_dir: Répertoire de données
        flow_names: Liste des flux à exécuter (None = tous)

    Returns:
        Dictionnaire avec résultats par flux
    """
    lake = SemanticDataLake(data_dir=data_dir)
    results = {}

    flows = {
        "synthea": SyntheaFHIRFlow(lake, n_patients=100),
        "mimic": MIMICIIIFlow(lake),
        "lab": LabSimulatedFlow(lake, n_records=600),
        "dicom": DICOMFlow(lake, n_studies=150),
    }

    if flow_names:
        flows = {k: v for k, v in flows.items() if k in flow_names}

    print("=" * 60)
    print("Exécution des Flux d'Intégration EHDS")
    print("=" * 60)

    for flow_name, flow in flows.items():
        try:
            flow_results = flow.run_full_flow()
            results[flow_name] = flow_results
        except Exception as e:
            print(f"❌ Erreur dans flux {flow_name}: {e}")
            results[flow_name] = {"error": str(e)}

    # Finaliser: sauvegarder catalogue et graphe RDF
    print("\n[Finalisation] Sauvegarde couche sémantique...")
    lake.save_semantic_catalog()

    # Construire graphe RDF depuis Gold
    for flow_name, flow_results in results.items():
        if "gold" in flow_results and not isinstance(flow_results.get("gold"), str):
            try:
                lake.build_data_graph_from_gold(flow_results["gold"], sample_size=200)
            except Exception as e:
                print(f"Warning: Could not build RDF for {flow_name}: {e}")

    lake.save_data_graph()

    print("\n" + "=" * 60)
    print("Tous les flux terminés")
    print("=" * 60)
    print(f"\nRésultats:")
    for flow_name, flow_results in results.items():
        if "error" not in flow_results:
            print(f"  ✓ {flow_name}: Bronze + Silver + Gold + RDF")
        else:
            print(f"  ❌ {flow_name}: {flow_results['error']}")

    return results
