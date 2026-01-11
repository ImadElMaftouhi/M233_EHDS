# Structure modulaire EHDS

Ce package contient une architecture modulaire réorganisée pour le projet EHDS.

## Structure

```
ehds/
├── __init__.py              # Package principal
├── compat.py                # Compatibilité avec anciens scripts
│
├── utils/                   # Utilitaires partagés
│   ├── __init__.py
│   ├── hashing.py          # Pseudonymisation (SHA-256, UID DICOM)
│   ├── file_utils.py       # Gestion fichiers/répertoires
│   └── constants.py        # Constantes (LOINC, ICD-10, etc.)
│
├── data/                    # Génération et intégration de données
│   ├── __init__.py
│   ├── generators.py       # DataGenerator (génération données simulées)
│   └── integrator.py       # DataIntegrator (intégration multi-sources)
│
├── semantic/                # Couche sémantique RDF
│   ├── __init__.py
│   ├── namespaces.py       # Namespaces RDF (EHDS, FHIR, LOINC, etc.)
│   ├── data_lake.py        # SemanticDataLake (Bronze/Silver/Gold)
│   └── rdf_layer.py        # RDFLayer (transformation RDF + SPARQL)
│
└── pipelines/               # Pipelines d'exécution
    ├── __init__.py
    ├── data_lake_pipeline.py    # Pipeline Data Lake complet
    └── integration_flows.py      # Flux d'intégration ELT
```

## Utilisation

### Point d'entrée principal

```bash
python run_pipeline.py --data-lake --run-all
python run_pipeline.py --flows synthea lab
```

### Utilisation programmatique

```python
from ehds.data import DataGenerator, DataIntegrator
from ehds.semantic import SemanticDataLake, RDFLayer
from ehds.pipelines import run_data_lake_pipeline

# Pipeline complet
run_data_lake_pipeline(data_dir=Path("data"))

# Ou utilisation modulaire
lake = SemanticDataLake(data_dir=Path("data"))
generator = DataGenerator(data_dir=Path("data"))
integrator = DataIntegrator(data_dir=Path("data"))
rdf_layer = RDFLayer(data_dir=Path("data"))
```

## Migration depuis l'ancienne structure

Pour compatibilité, utilisez `ehds.compat` :

```python
from ehds.compat import (
    EHDSDataPreparation,  # Alias de DataGenerator
    EHDSDataIntegration,  # Alias de DataIntegrator
    EHDSSemanticLayer,    # Alias de RDFLayer
)
```

## Noms de classes

- `EHDSDataPreparation` → `DataGenerator`
- `EHDSDataIntegration` → `DataIntegrator`
- `EHDSSemanticLayer` → `RDFLayer`
- `SemanticDataLake` → inchangé
