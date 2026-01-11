"""Pipeline principal pour le Data Lake sémantique EHDS."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rdflib import Literal
from rdflib.namespace import RDF, XSD

from ..data import DataIntegrator
from ..semantic import CATALOG, RDFLayer, SemanticDataLake
from ..utils import ensure_dir


def run_data_lake_pipeline(data_dir: Path = Path("data"), n_patients: int = 120, n_labs: int = 600):
    """
    Pipeline complet utilisant l'architecture Data Lake sémantique.
    
    Args:
        data_dir: Répertoire de données
        n_patients: Nombre de patients (pour génération si nécessaire)
        n_labs: Nombre de résultats de laboratoire (pour génération si nécessaire)
    """
    print("=" * 60)
    print("EHDS Semantic Data Lake Pipeline")
    print("=" * 60)

    # 1. Vérification des données sources
    print("\n[1/5] Vérification des données sources...")
    print("Note: Les données doivent être déjà présentes dans data/bronze/ ou data/source_*/")
    print("Ce pipeline ne génère pas de données simulées.")

    # 2. Initialiser le Data Lake
    print("\n[2/5] Initialisation du Data Lake sémantique...")
    lake = SemanticDataLake(data_dir=data_dir)

    # 3. Ingestion dans Bronze
    print("\n[3/5] Ingestion dans Bronze (Raw Zone)...")
    bronze_uris = []

    # MIMIC-III CSV (external data - must be manually copied to source_mimic_csv/)
    mimic_source = data_dir / "source_mimic_csv"
    if mimic_source.exists() and list(mimic_source.glob("*.csv")):
        mimic_bronze = lake.ingest_raw("mimic_patients", mimic_source / "PATIENTS.CSV", domain="mimic", format_type="csv")
        bronze_uris.append(mimic_bronze)

    # Lab JSON
    lab_source = data_dir / "source_lab_json" / "lab_results.json"
    if lab_source.exists():
        lab_bronze = lake.ingest_raw("lab_results", lab_source, domain="lab", format_type="json")
        bronze_uris.append(lab_bronze)

    # FHIR NDJSON
    fhir_source = data_dir / "source_fhir_ndjson" / "bundle.ndjson"
    if fhir_source.exists():
        fhir_bronze = lake.ingest_raw("fhir_bundle", fhir_source, domain="fhir", format_type="ndjson")
        bronze_uris.append(fhir_bronze)

    # DICOM
    dicom_source = data_dir / "source_dicom"
    if dicom_source.exists():
        dicom_bronze = lake.ingest_raw("dicom_images", dicom_source, domain="dicom", format_type="dicom")
        bronze_uris.append(dicom_bronze)

    # 4. Enrichissement vers Silver
    print("\n[4/5] Enrichissement vers Silver...")
    silver_uris = []

    for bronze_uri in bronze_uris:
        # Récupérer le domaine depuis le catalogue
        domain = None
        for obj in lake.catalog.objects(bronze_uri, CATALOG.domain):
            domain = str(obj)
            break

        # Règles d'enrichissement selon le domaine
        enrichment_rules = {
            "pseudonymize": "patient_id",
            "quality_checks": True,
        }

        if domain == "lab":
            enrichment_rules["normalize_units"] = {"creatinine": "mg/dL"}
            enrichment_rules["semantic_tags"] = {"loinc": True}
        elif domain == "fhir":
            enrichment_rules["semantic_tags"] = {"icd10": True}

        try:
            silver_uri = lake.enrich_to_silver(bronze_uri, enrichment_rules)
            silver_uris.append(silver_uri)
        except Exception as e:
            print(f"Warning: Could not enrich {bronze_uri}: {e}")

    # 5. Curation vers Gold
    print("\n[5/5] Curation vers Gold (schéma unifié)...")

    # Utiliser l'intégrateur pour créer le schéma unifié
    integrator = DataIntegrator(data_dir=data_dir)
    unified_tables = integrator.integrate()

    # Sauvegarder en Gold (Parquet - format principal)
    gold_path = lake.gold / "ehds_unified.parquet"
    ensure_dir(gold_path.parent)
    patients_gold = unified_tables["patients"]
    patients_gold.to_parquet(gold_path, index=False)

    # Export SQLite (couche de visualisation/requêtes SQL - optionnel)
    sqlite_path = data_dir / "integrated" / "ehds.db"
    integrator.export_to_sqlite(unified_tables, db_path=sqlite_path)
    print(f"✓ SQLite export (dashboard compatibility): {sqlite_path}")

    # Enregistrer Gold dans le catalogue
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")
    gold_uri = CATALOG[f"dataset/ehds_unified_gold_{timestamp}"]
    lake.catalog.add((gold_uri, RDF.type, CATALOG.Dataset))
    lake.catalog.add((gold_uri, CATALOG.zone, Literal("gold")))
    lake.catalog.add((gold_uri, CATALOG.location, Literal(str(gold_path))))
    lake.catalog.add((gold_uri, CATALOG.rowCount, Literal(len(patients_gold), datatype=XSD.integer)))
    lake.catalog.add((gold_uri, CATALOG["format"], Literal("parquet")))

    # 6. Construire la couche sémantique
    print("\n[6/6] Construction de la couche sémantique...")

    # Utiliser RDFLayer pour transformer en RDF
    rdf_layer = RDFLayer(data_dir=data_dir)
    rdf_layer.build_graph(
        patients=unified_tables.get("patients"),
        lab_results=unified_tables.get("lab_results", integrator.load_lab_json()),
        conditions=unified_tables.get("conditions"),
        allergies=unified_tables.get("allergies"),
        prescriptions=unified_tables.get("prescriptions"),
        dicom_images=unified_tables.get("dicom_images"),
    )

    # Copier le graphe dans le lake
    lake.data_graph = rdf_layer.g

    # Sauvegarder le catalogue et le graphe
    lake.save_semantic_catalog()
    lake.save_data_graph()

    # Exécuter quelques requêtes de démonstration
    print("\n    Exécution de requêtes SPARQL de démonstration...")
    rdf_layer.run_predefined_queries()

    print("\n" + "=" * 60)
    print("Pipeline terminé avec succès!")
    print("=" * 60)
    print(f"\nRésumé:")
    print(f"  - Bronze: {len(bronze_uris)} datasets bruts")
    print(f"  - Silver: {len(silver_uris)} datasets enrichis")
    print(f"  - Gold: 1 dataset unifié ({len(patients_gold)} patients)")
    print(f"  - Semantic: {len(lake.data_graph)} triples RDF")
    print(f"\nFichiers générés:")
    print(f"  - Catalogue: {lake.semantic / 'catalog.ttl'}")
    print(f"  - Data Graph: {lake.semantic / 'ehds_data_graph.ttl'}")
