# Architecture Hybride EHDS - Documentation

## Vue d'ensemble

Cette architecture combine les meilleurs aspects de deux approches :

1. **Architecture Data Lake classique** (Bronze/Silver/Gold) - pour la gestion des données structurées
2. **Couche sémantique** (RDF/SPARQL) - pour l'interopérabilité sémantique selon les spécifications EHDS

## Architecture Complète

```
┌─────────────────────────────────────────────────────────────────┐
│                  COUCHE D'ACCÈS                                  │
│  ┌─────────────────┐              ┌──────────────────┐          │
│  │  API FHIR       │              │  SPARQL          │          │
│  │  (Usage         │              │  (Usage          │          │
│  │   primaire)     │              │   secondaire)    │          │
│  └────────┬────────┘              └────────┬─────────┘          │
│           │                                │                    │
│           │        Dashboard Streamlit     │                    │
└───────────┼────────────────────────────────┼────────────────────┘
            │                                │
┌───────────┼────────────────────────────────┼────────────────────┐
│           │    COUCHE SÉMANTIQUE            │                    │
│           │  ┌──────────────────────────┐  │                    │
│           │  │  Catalogue RDF           │  │                    │
│           │  │  - Métadonnées datasets  │  │                    │
│           │  │  - Lignage PROV-O        │  │                    │
│           │  │  - Qualité               │  │                    │
│           │  └──────────┬───────────────┘  │                    │
│           │             │                   │                    │
│           │  ┌──────────▼───────────────┐  │                    │
│           │  │  Data Graph RDF          │  │                    │
│           │  │  (Triplestore)           │  │                    │
│           │  │  - Données transformées  │  │                    │
│           │  │  - Ontologies standards  │  │                    │
│           │  │  - FHIR RDF, LOINC, ICD  │  │                    │
│           │  └──────────────────────────┘  │                    │
└───────────┼────────────────────────────────┼────────────────────┘
            │                                │
┌───────────┼────────────────────────────────┼────────────────────┐
│           │                                │                    │
│  ┌────────▼────────┐  ┌─────────▼──────┐  ┌─────▼──────┐     │
│  │    BRONZE       │  │     SILVER     │  │    GOLD    │     │
│  │   (Raw Zone)    │→ │  (Enriched)    │ →│ (Curated)  │     │
│  │                 │  │                │  │            │     │
│  │ Format original │  │ Parquet        │  │ Parquet    │     │
│  │ - CSV           │  │ + Pseudonym.   │  │ Unified    │     │
│  │ - JSON          │  │ + Normalisation│  │ Schema     │     │
│  │ - NDJSON        │  │ + Tags sémant. │  │ Analytics  │     │
│  │ - DICOM         │  │ + Quality flags│  │ Ready      │     │
│  │                 │  │                │  │            │     │
│  │ Schema-on-read  │  │ Enriched       │  │ Curated    │     │
│  └─────────────────┘  └────────────────┘  └────────────┘     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Zones du Data Lake

### Zone Bronze (Raw)

**Objectif** : Stocker les données dans leur format original, sans transformation.

**Caractéristiques** :
- Format original préservé (CSV, JSON, NDJSON, DICOM)
- Aucun schéma imposé (schema-on-read)
- Organisation par domaine (ehr/, lab/, fhir/, dicom/)
- Traçabilité via catalogue sémantique RDF

**Stockage** :
```
data/bronze/
├── ehr/
│   └── ehr_patients.csv
├── lab/
│   └── lab_results.json
├── fhir/
│   └── bundle.ndjson
└── dicom/
    └── patient_*/
        └── study_*/
            └── *.dcm
```

### Zone Silver (Enriched)

**Objectif** : Données enrichies avec métadonnées, normalisées et nettoyées.

**Opérations** :
- Pseudonymisation (SHA-256)
- Normalisation d'unités (ex: créatinine µmol/L → mg/dL)
- Ajout de tags sémantiques (LOINC, ICD-10)
- Flags de qualité (complétude, anomalies)
- Conversion en Parquet

**Stockage** :
```
data/silver/
├── ehr/
│   └── ehr_patients_enriched.parquet
├── lab/
│   └── lab_results_enriched.parquet
├── fhir/
│   └── bundle_enriched.parquet
└── dicom/
    └── dicom_metadata_enriched.parquet
```

**Métadonnées cataloguées** :
- Complétude
- Nombre de lignes/colonnes
- Taille du fichier
- Lignage PROV-O vers Bronze

### Zone Gold (Curated)

**Objectif** : Schéma unifié, prêt pour l'exploitation analytique.

**Caractéristiques** :
- Schéma harmonisé cross-sources
- Identifiants unifiés (patient_id_pseudo)
- Qualité validée
- Format Parquet optimisé
- Prêt pour ML/Analytics

**Stockage** :
```
data/gold/
└── ehds_unified_YYYYMMDD.parquet
```

## Couche Sémantique

### Catalogue Sémantique (Métadonnées)

**Fichier** : `data/semantic/catalog.ttl`

**Contenu** :
- Métadonnées de tous les datasets (Bronze, Silver, Gold)
- Lignage PROV-O (provenance des données)
- Métriques de qualité
- Localisation et format
- Liens vers ontologies

**Exemple de requête** :
```sparql
PREFIX catalog: <http://ehds.eu/catalog#>
SELECT ?dataset ?zone ?rows ?completeness
WHERE {
    ?dataset catalog:zone ?zone ;
             catalog:rowCount ?rows ;
             catalog:completeness ?completeness .
}
```

### Data Graph RDF (Données Transformées)

**Fichier** : `data/semantic/ehds_data_graph.ttl`

**Contenu** :
- Transformation complète des données en RDF
- Utilisation d'ontologies standards :
  - **FHIR RDF** : http://hl7.org/fhir/
  - **LOINC** : http://loinc.org/rdf#
  - **ICD-10** : http://id.who.int/icd/release/11/
  - **SKOS** : pour les vocabulaires contrôlés

**Triplestore** : Apache Jena Fuseki, GraphDB, ou Blazegraph (optionnel)

**Exemple de requête** :
```sparql
PREFIX ehds: <http://ehds.eu/ontology#>
PREFIX fhir: <http://hl7.org/fhir/>
SELECT ?patient ?glucose ?date
WHERE {
    ?lab a ehds:LabResult ;
         ehds:hasPatient ?patient ;
         ehds:label "Glucose" ;
         ehds:value ?glucose ;
         ehds:date ?date .
    FILTER (?glucose > 140)
}
```

## Utilisation

### Pipeline complet

```bash
python ehds_data_lake_integration.py --run-all
```

### Utilisation programmatique

```python
from ehds_data_lake import SemanticDataLake
from pathlib import Path

# Initialiser le Data Lake
lake = SemanticDataLake(data_dir=Path("data"))

# Ingestion Bronze
ehr_uri = lake.ingest_raw("ehr_patients", 
                          Path("source/ehr.csv"), 
                          domain="ehr")

# Enrichissement Silver
silver_uri = lake.enrich_to_silver(ehr_uri, {
    "pseudonymize": "patient_id",
    "quality_checks": True
})

# Curation Gold
gold_uri = lake.curate_to_gold([silver_uri], 
                               unified_schema={})

# Construire graphe RDF
lake.build_data_graph_from_gold(gold_uri)

# Sauvegarder
lake.save_semantic_catalog()
lake.save_data_graph()

# Requêtes SPARQL
results = lake.query_catalog("""
    PREFIX catalog: <http://ehds.eu/catalog#>
    SELECT ?dataset WHERE {
        ?dataset catalog:zone "silver"
    }
""")
```

## Avantages de cette Architecture

1. **FAIR Compliance** :
   - ✅ **Findable** : Catalogue sémantique RDF
   - ✅ **Accessible** : Zones clairement définies
   - ✅ **Interoperable** : Standards (FHIR, LOINC, ICD-10)
   - ✅ **Reusable** : Données curatées et documentées

2. **EHDS Alignment** :
   - ✅ Usage primaire : API FHIR (à implémenter)
   - ✅ Usage secondaire : SPARQL + exports
   - ✅ Interopérabilité sémantique
   - ✅ Traçabilité des données

3. **Scalabilité** :
   - ✅ Format Parquet (compression, colonnaires)
   - ✅ Schema-on-read (flexibilité)
   - ✅ Lignage (audit, gouvernance)

4. **Faisabilité** :
   - ✅ Technologies standards (Python, Parquet, RDF)
   - ✅ Implémentation progressive
   - ✅ Compatible avec l'existant

## Évolutions Futures

1. **Triplestore dédié** : Jena Fuseki ou GraphDB
2. **API FHIR** : FastAPI avec ressources FHIR
3. **Streaming** : Apache Kafka pour ingestion temps réel
4. **Data Quality** : Great Expectations ou Soda
5. **Lineage avancé** : OpenLineage pour traçabilité complète
