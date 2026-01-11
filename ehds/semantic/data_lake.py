"""Data Lake sémantique avec architecture Bronze/Silver/Gold."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, RDF, RDFS, SKOS, XSD

from ..utils import ICD10, LOINC_TESTS, ensure_dir, sha256_pseudo
from .namespaces import CATALOG, EHDS, FHIR, ICD10_NS, LOINC, PROV_O


class SemanticDataLake:
    """
    Data Lake sémantique avec architecture Bronze/Silver/Gold
    + Couche sémantique (Catalogue + Data Graph RDF)
    """

    def __init__(self, data_dir: Path = Path("data")):
        """Initialise le Data Lake avec les zones et graphes RDF."""
        self.data_dir = Path(data_dir)
        self.bronze = self.data_dir / "bronze"
        self.silver = self.data_dir / "silver"
        self.gold = self.data_dir / "gold"
        self.semantic = self.data_dir / "semantic"

        # Initialize zones
        for zone in [self.bronze, self.silver, self.gold, self.semantic]:
            ensure_dir(zone)

        # Semantic catalog graph (métadonnées)
        self.catalog = Graph()
        self.catalog.bind("ehds", EHDS)
        self.catalog.bind("catalog", CATALOG)
        self.catalog.bind("prov", PROV_O)
        self.catalog.bind("dct", DCTERMS)
        self.catalog.bind("fhir", FHIR)

        # Data graph (données RDF transformées)
        self.data_graph = Graph()
        self.data_graph.bind("ehds", EHDS)
        self.data_graph.bind("fhir", FHIR)
        self.data_graph.bind("loinc", LOINC)
        self.data_graph.bind("icd10", ICD10_NS)
        self.data_graph.bind("skos", SKOS)
        self.data_graph.bind("owl", OWL)
        self.data_graph.bind("rdfs", RDFS)
        self.data_graph.bind("rdf", RDF)
        self.data_graph.bind("xsd", XSD)

        # Initialize ontology in data graph
        self._init_ontology()

    def _init_ontology(self):
        """Initialise l'ontologie EHDS dans le graphe de données."""
        ont = URIRef(EHDS)
        self.data_graph.add((ont, RDF.type, OWL.Ontology))
        self.data_graph.add((ont, RDFS.label, Literal("EHDS Semantic Data Lake Ontology")))
        self.data_graph.add(
            (ont, DCTERMS.description, Literal("Ontology for European Health Data Space semantic integration"))
        )

        # Classes principales
        classes = [
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
        ]
        for cls in classes:
            self.data_graph.add((EHDS[cls], RDF.type, OWL.Class))

        # Properties principales
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
            ("value", OWL.DatatypeProperty),
            ("unit", OWL.DatatypeProperty),
            ("loincCode", OWL.DatatypeProperty),
            ("icd10Code", OWL.DatatypeProperty),
            ("label", OWL.DatatypeProperty),
            ("date", OWL.DatatypeProperty),
        ]
        for p, t in props:
            self.data_graph.add((EHDS[p], RDF.type, t))

    # ==================== BRONZE ZONE ====================

    def ingest_raw(self, source_name: str, file_path: Path, domain: str, format_type: str = None) -> URIRef:
        """
        Ingère des données brutes dans Bronze (pas de transformation).
        Schema-on-read: les données gardent leur format original.
        Ne copie pas si la destination existe déjà (données déjà présentes).
        """
        bronze_domain_dir = self.bronze / domain
        ensure_dir(bronze_domain_dir)

        bronze_path = bronze_domain_dir / file_path.name

        # Si le fichier source est déjà dans le répertoire bronze du domaine, utiliser directement
        # Sinon, copier seulement si la destination n'existe pas déjà
        if file_path.parent.resolve() == bronze_domain_dir.resolve():
            bronze_path = file_path
            print(f"  Using existing file in bronze: {bronze_path}")
        else:
            if file_path.is_dir():
                if not bronze_path.exists():
                    shutil.copytree(file_path, bronze_path, dirs_exist_ok=True)
                    print(f"  Copied directory to bronze: {bronze_path}")
                else:
                    print(f"  Using existing directory in bronze: {bronze_path}")
            else:
                if not bronze_path.exists():
                    shutil.copy2(file_path, bronze_path)
                    print(f"  Copied file to bronze: {bronze_path}")
                else:
                    print(f"  Using existing file in bronze: {bronze_path}")

        # Détecter le format si non spécifié
        if format_type is None:
            if file_path.suffix == ".csv":
                format_type = "csv"
            elif file_path.suffix == ".json":
                format_type = "json"
            elif file_path.suffix == ".ndjson":
                format_type = "ndjson"
            elif file_path.suffix == ".dcm" or file_path.is_dir():
                format_type = "dicom"
            else:
                format_type = file_path.suffix[1:] if file_path.suffix else "unknown"

        # Enregistrer dans le catalogue sémantique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_uri = CATALOG[f"dataset/{source_name}_{timestamp}"]

        self.catalog.add((dataset_uri, RDF.type, CATALOG.Dataset))
        self.catalog.add((dataset_uri, DCTERMS.title, Literal(source_name)))
        self.catalog.add((dataset_uri, CATALOG.zone, Literal("bronze")))
        self.catalog.add((dataset_uri, CATALOG.domain, Literal(domain)))
        self.catalog.add((dataset_uri, CATALOG["format"], Literal(format_type)))
        self.catalog.add((dataset_uri, CATALOG.location, Literal(str(bronze_path))))
        self.catalog.add((dataset_uri, DCTERMS.created, Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))
        self.catalog.add((dataset_uri, CATALOG.schemaOnRead, Literal(True, datatype=XSD.boolean)))

        # Métadonnées techniques
        if bronze_path.is_file():
            size = bronze_path.stat().st_size
            self.catalog.add((dataset_uri, CATALOG.sizeBytes, Literal(size, datatype=XSD.integer)))
        elif bronze_path.is_dir():
            total_size = sum(f.stat().st_size for f in bronze_path.rglob("*") if f.is_file())
            self.catalog.add((dataset_uri, CATALOG.sizeBytes, Literal(total_size, datatype=XSD.integer)))

        print(f"✓ Ingested {source_name} → {bronze_path}")
        return dataset_uri

    # ==================== SILVER ZONE ====================

    def enrich_to_silver(self, bronze_dataset_uri: URIRef, enrichment_rules: Dict[str, Any]) -> URIRef:
        """
        Enrichit les données Bronze → Silver.
        - Pseudonymisation
        - Normalisation d'unités
        - Ajout de tags sémantiques (LOINC, ICD-10)
        - Format Parquet pour performance
        """
        # Récupérer la localisation depuis le catalogue
        location = None
        domain = None
        for obj in self.catalog.objects(bronze_dataset_uri, CATALOG.location):
            location = Path(str(obj))
        for obj in self.catalog.objects(bronze_dataset_uri, CATALOG.domain):
            domain = str(obj)

        if not location or not location.exists():
            raise FileNotFoundError(f"Bronze dataset not found: {location}")

        # Lire selon le format (schema-on-read)
        df = self._read_with_schema(location)

        # Enrichissements
        if "pseudonymize" in enrichment_rules:
            col = enrichment_rules["pseudonymize"]
            if col in df.columns:
                df["patient_id_pseudo"] = df[col].apply(lambda x: sha256_pseudo(str(x)))

        if "normalize_units" in enrichment_rules:
            df = self._normalize_units(df, enrichment_rules["normalize_units"])

        if "semantic_tags" in enrichment_rules:
            df = self._add_semantic_tags(df, enrichment_rules["semantic_tags"])

        if "quality_checks" in enrichment_rules and enrichment_rules["quality_checks"]:
            df = self._add_quality_flags(df, enrichment_rules.get("quality_metadata", {}))

        # Sauvegarder en Parquet dans Silver
        silver_domain_dir = self.silver / domain
        ensure_dir(silver_domain_dir)
        silver_path = silver_domain_dir / f"{location.stem}_enriched.parquet"
        df.to_parquet(silver_path, index=False, engine="pyarrow")

        # Enregistrer dans le catalogue
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        silver_uri = CATALOG[f"dataset/{location.stem}_silver_{timestamp}"]
        self.catalog.add((silver_uri, RDF.type, CATALOG.Dataset))
        self.catalog.add((silver_uri, CATALOG.zone, Literal("silver")))
        self.catalog.add((silver_uri, CATALOG.domain, Literal(domain)))
        self.catalog.add((silver_uri, CATALOG.location, Literal(str(silver_path))))
        self.catalog.add((silver_uri, PROV_O.wasDerivedFrom, bronze_dataset_uri))
        self.catalog.add((silver_uri, CATALOG["format"], Literal("parquet")))
        self.catalog.add((silver_uri, DCTERMS.created, Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))

        # Métadonnées qualité
        completeness = float((df.notnull().sum() / len(df) * 100).mean()) if len(df) > 0 else 0.0
        self.catalog.add((silver_uri, CATALOG.completeness, Literal(completeness, datatype=XSD.float)))
        self.catalog.add((silver_uri, CATALOG.rowCount, Literal(len(df), datatype=XSD.integer)))
        self.catalog.add((silver_uri, CATALOG.columnCount, Literal(len(df.columns), datatype=XSD.integer)))

        # Métadonnées de taille
        size = silver_path.stat().st_size
        self.catalog.add((silver_uri, CATALOG.sizeBytes, Literal(size, datatype=XSD.integer)))

        print(f"✓ Enriched to Silver → {silver_path} ({len(df)} rows, {completeness:.1f}% completeness)")
        return silver_uri

    # ==================== GOLD ZONE ====================

    def curate_to_gold(
        self, silver_uris: List[URIRef], unified_schema: Dict[str, Any], gold_dataset_name: str = "unified"
    ) -> URIRef:
        """Crée la zone Gold avec schéma unifié."""
        # Charger tous les datasets Silver
        dfs = []
        domains = []

        for uri in silver_uris:
            location = None
            domain = None
            for obj in self.catalog.objects(uri, CATALOG.location):
                location = Path(str(obj))
            for obj in self.catalog.objects(uri, CATALOG.domain):
                domain = str(obj)

            if location and location.exists():
                df = pd.read_parquet(location)
                df["_source_domain"] = domain  # Conserver l'info de domaine
                dfs.append(df)
                domains.append(domain)

        if not dfs:
            raise ValueError("No Silver datasets found to curate")

        # Unifier selon le schéma
        unified_df = self._unify_schema(dfs, unified_schema)

        # Sauvegarder en Gold
        ensure_dir(self.gold)
        timestamp = datetime.now().strftime("%Y%m%d")
        gold_path = self.gold / f"{gold_dataset_name}_{timestamp}.parquet"
        unified_df.to_parquet(gold_path, index=False, engine="pyarrow")

        # Catalogue
        gold_uri = CATALOG[f"dataset/{gold_dataset_name}_gold_{timestamp}"]
        self.catalog.add((gold_uri, RDF.type, CATALOG.Dataset))
        self.catalog.add((gold_uri, CATALOG.zone, Literal("gold")))
        self.catalog.add((gold_uri, CATALOG.location, Literal(str(gold_path))))
        self.catalog.add((gold_uri, CATALOG["format"], Literal("parquet")))
        self.catalog.add((gold_uri, DCTERMS.created, Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))

        # Lignage vers Silver
        for silver_uri in silver_uris:
            self.catalog.add((gold_uri, PROV_O.wasDerivedFrom, silver_uri))

        # Métadonnées
        completeness = float((unified_df.notnull().sum() / len(unified_df) * 100).mean()) if len(unified_df) > 0 else 0.0
        self.catalog.add((gold_uri, CATALOG.completeness, Literal(completeness, datatype=XSD.float)))
        self.catalog.add((gold_uri, CATALOG.rowCount, Literal(len(unified_df), datatype=XSD.integer)))
        self.catalog.add((gold_uri, CATALOG.columnCount, Literal(len(unified_df.columns), datatype=XSD.integer)))

        size = gold_path.stat().st_size
        self.catalog.add((gold_uri, CATALOG.sizeBytes, Literal(size, datatype=XSD.integer)))

        print(f"✓ Curated to Gold → {gold_path} ({len(unified_df)} rows)")
        return gold_uri

    # ==================== SEMANTIC LAYER ====================

    def build_data_graph_from_gold(self, gold_uri: URIRef, sample_size: Optional[int] = None):
        """
        Construit le graphe RDF de données depuis Gold.
        Transformation complète selon ontologies standards.
        """
        location = None
        for obj in self.catalog.objects(gold_uri, CATALOG.location):
            location = Path(str(obj))

        if not location or not location.exists():
            raise FileNotFoundError(f"Gold dataset not found: {location}")

        df = pd.read_parquet(location)

        if sample_size:
            df = df.head(sample_size)

        # Ajouter les concepts de référence (LOINC, ICD-10) dans le graphe
        self._add_reference_concepts()

        # Transformer les données en RDF
        # Patients
        if "patient_id_pseudo" in df.columns:
            for _, row in df.iterrows():
                patient_uri = EHDS[f"patient/{row['patient_id_pseudo']}"]
                self.data_graph.add((patient_uri, RDF.type, EHDS.Patient))
                self.data_graph.add((patient_uri, RDF.type, FHIR.Patient))

                if "first_name" in row and pd.notna(row["first_name"]):
                    self.data_graph.add((patient_uri, FHIR["name.given"], Literal(str(row["first_name"]))))
                if "last_name" in row and pd.notna(row["last_name"]):
                    self.data_graph.add((patient_uri, FHIR["name.family"], Literal(str(row["last_name"]))))

        # Lab Results
        if "test_name" in df.columns and "value" in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get("test_name")) and pd.notna(row.get("value")):
                    lab_uri = EHDS[f"lab/{row.get('lab_id', f'lab_{idx}')}"]
                    self.data_graph.add((lab_uri, RDF.type, EHDS.LabResult))

                    # Lien vers patient
                    if "patient_id_pseudo" in row and pd.notna(row["patient_id_pseudo"]):
                        patient_uri = EHDS[f"patient/{row['patient_id_pseudo']}"]
                        self.data_graph.add((lab_uri, EHDS.hasPatient, patient_uri))

                    # Valeur et unité
                    if pd.notna(row.get("value")):
                        self.data_graph.add((lab_uri, EHDS.value, Literal(float(row["value"]), datatype=XSD.float)))
                    if pd.notna(row.get("unit")):
                        self.data_graph.add((lab_uri, EHDS.unit, Literal(str(row["unit"]))))

                    # LOINC code
                    if pd.notna(row.get("test_code_loinc")):
                        loinc_code = str(row["test_code_loinc"])
                        loinc_uri = LOINC[loinc_code]
                        self.data_graph.add((lab_uri, EHDS.loincCode, Literal(loinc_code)))
                        self.data_graph.add((lab_uri, EHDS.hasTest, loinc_uri))

        print(f"✓ Data graph built: {len(self.data_graph)} triples")

    def save_semantic_catalog(self, filename: str = "catalog.ttl") -> Path:
        """Sauvegarde le catalogue sémantique (métadonnées)."""
        catalog_path = self.semantic / filename
        self.catalog.serialize(destination=str(catalog_path), format="turtle")
        print(f"✓ Semantic catalog saved: {catalog_path}")
        return catalog_path

    def save_data_graph(self, filename: str = "ehds_data_graph.ttl") -> Path:
        """Sauvegarde le graphe RDF de données."""
        graph_path = self.semantic / filename
        self.data_graph.serialize(destination=str(graph_path), format="turtle")
        print(f"✓ Data graph saved: {graph_path} ({len(self.data_graph)} triples)")
        return graph_path

    # ==================== HELPERS ====================

    def _read_with_schema(self, path: Path) -> pd.DataFrame:
        """Schema-on-read: lire selon le format."""
        if path.is_dir():
            return self._extract_dicom_metadata(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        elif path.suffix == ".ndjson":
            data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            return pd.DataFrame(data)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _extract_dicom_metadata(self, dicom_dir: Path) -> pd.DataFrame:
        """Extrait les métadonnées DICOM (lightweight)."""
        import pydicom

        metadata = []

        for dcm_file in dicom_dir.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(dcm_file))
                metadata.append(
                    {
                        "patient_id": getattr(ds, "PatientID", None),
                        "study_uid": getattr(ds, "StudyInstanceUID", None),
                        "series_uid": getattr(ds, "SeriesInstanceUID", None),
                        "sop_uid": getattr(ds, "SOPInstanceUID", None),
                        "modality": getattr(ds, "Modality", None),
                        "study_date": getattr(ds, "StudyDate", None),
                        "file_path": str(dcm_file.relative_to(dicom_dir)),
                    }
                )
            except Exception as e:
                print(f"Warning: Could not read {dcm_file}: {e}")

        return pd.DataFrame(metadata)

    def _normalize_units(self, df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
        """Normalisation des unités."""
        df = df.copy()

        if "creatinine" in rules and "test_name" in df.columns:
            mask = df["test_name"].eq("Creatinine") & df["unit"].eq("µmol/L")
            if mask.any():
                df.loc[mask, "value"] = df.loc[mask, "value"] / 88.4
                df.loc[mask, "unit"] = "mg/dL"

        return df

    def _add_semantic_tags(self, df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
        """Ajoute des tags sémantiques."""
        df = df.copy()

        if "loinc" in rules and "test_code_loinc" in df.columns:
            df["loinc_uri"] = df["test_code_loinc"].apply(lambda x: f"http://loinc.org/rdf#{x}" if pd.notna(x) else None)

        if "icd10" in rules and "icd10_code" in df.columns:
            df["icd10_label"] = df["icd10_code"].map(ICD10).fillna("Unknown")

        return df

    def _add_quality_flags(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Ajoute des flags de qualité."""
        df = df.copy()

        # Flags d'anomalie pour lab results
        if "test_name" in df.columns and "value" in df.columns:
            ref_ranges = {t["name"]: (t["low"], t["high"]) for t in LOINC_TESTS}
            df["ref_low"] = df["test_name"].map(lambda n: ref_ranges.get(n, (None, None))[0])
            df["ref_high"] = df["test_name"].map(lambda n: ref_ranges.get(n, (None, None))[1])
            df["is_abnormal"] = (df["value"] < df["ref_low"]) | (df["value"] > df["ref_high"])

        return df

    def _unify_schema(self, dfs: List[pd.DataFrame], schema: Dict) -> pd.DataFrame:
        """Unifie les schémas selon les règles définies."""
        if not dfs:
            return pd.DataFrame()

        # Trouver les colonnes communes
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)

        # Ajouter patient_id_pseudo si présent dans au moins un
        patient_id_cols = ["patient_id_pseudo", "patient_id"]
        for col in patient_id_cols:
            if any(col in df.columns for df in dfs):
                common_cols.add(col)

        # Standardiser et unifier
        unified_dfs = []
        for df in dfs:
            df_copy = df.copy()
            # Renommer patient_id en patient_id_pseudo si nécessaire
            if "patient_id" in df_copy.columns and "patient_id_pseudo" not in df_copy.columns:
                df_copy["patient_id_pseudo"] = df_copy["patient_id"].apply(sha256_pseudo)
            unified_dfs.append(df_copy)

        return pd.concat(unified_dfs, ignore_index=True, sort=False)

    def _add_reference_concepts(self):
        """Ajoute les concepts de référence (LOINC, ICD-10) dans le graphe."""
        # LOINC concepts
        loinc_scheme = EHDS["scheme/loinc"]
        self.data_graph.add((loinc_scheme, RDF.type, SKOS.ConceptScheme))
        self.data_graph.add((loinc_scheme, SKOS.prefLabel, Literal("LOINC Test Codes")))

        for test in LOINC_TESTS:
            concept_uri = LOINC[test["code"]]
            self.data_graph.add((concept_uri, RDF.type, SKOS.Concept))
            self.data_graph.add((concept_uri, SKOS.inScheme, loinc_scheme))
            self.data_graph.add((concept_uri, SKOS.notation, Literal(test["code"])))
            self.data_graph.add((concept_uri, SKOS.prefLabel, Literal(test["name"])))

        # ICD-10 concepts
        icd_scheme = EHDS["scheme/icd10"]
        self.data_graph.add((icd_scheme, RDF.type, SKOS.ConceptScheme))
        self.data_graph.add((icd_scheme, SKOS.prefLabel, Literal("ICD-10 Codes")))

        for code, label in ICD10.items():
            concept_uri = ICD10_NS[code]
            self.data_graph.add((concept_uri, RDF.type, SKOS.Concept))
            self.data_graph.add((concept_uri, SKOS.inScheme, icd_scheme))
            self.data_graph.add((concept_uri, SKOS.notation, Literal(code)))
            self.data_graph.add((concept_uri, SKOS.prefLabel, Literal(label)))

    def query_catalog(self, sparql_query: str) -> List[Dict]:
        """Requête SPARQL sur le catalogue."""
        results = self.catalog.query(sparql_query)
        return [dict(row) for row in results]

    def query_data_graph(self, sparql_query: str) -> List[Dict]:
        """Requête SPARQL sur le graphe de données."""
        results = self.data_graph.query(sparql_query)
        return [dict(row) for row in results]
