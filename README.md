# EHDS Data Space Integration

A prototype for integrating heterogeneous health data sources according to the European Health Data Space (EHDS) specifications, combining a hybrid Data Lake architecture with semantic interoperability.

## ✨ Key Features

- **Hybrid Data Lake Architecture**: Bronze (raw), Silver (enriched), and Gold (curated) data zones.
- **Semantic Layer**: RDF/SPARQL-based metadata catalog and data graph using FHIR, LOINC, and ICD-10 ontologies.
- **Multi-Source Integration**: Pipelines for Synthea (FHIR), MIMIC-III, Lab (JSON), and DICOM data.
- **Interactive Dashboard**: A Streamlit application to explore data quality, integration status, and run SPARQL queries.
- **FAIR Compliance**: Designed to be Findable, Accessible, Interoperable, and Reusable.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACCESS LAYER                                  │
│  ┌─────────────────┐              ┌──────────────────┐          │
│  │  FHIR API       │              │  SPARQL Endpoint │          │
│  │  (Primary Use)  │              │  (Secondary Use) │          │
│  └────────┬────────┘              └────────┬─────────┘          │
│           │        Streamlit Dashboard     │                    │
└───────────┼────────────────────────────────┼────────────────────┘
            │                                │
┌───────────┼────────────────────────────────┼────────────────────┐
│           │      SEMANTIC LAYER            │                    │
│           │  ┌──────────────────────────┐  │                    │
│           │  │  RDF Catalog & Graph     │  │                    │
│           │  │  (FHIR, LOINC, ICD-10)   │  │                    │
│           │  └──────────────────────────┘  │                    │
└───────────┼────────────────────────────────┼────────────────────┘
            │                                │
┌───────────┴────────────────────────────────┴────────────────────┐
│  ┌─────────────────┐  ┌────────────────┐  ┌────────────┐        │
│  │    BRONZE       │→ │     SILVER     │ →│    GOLD    │        │
│  │   (Raw Data)    │  │  (Enriched)    │  │ (Curated)  │        │
│  │  CSV, JSON,     │  │  Parquet,      │  │  Unified   │        │
│  │  NDJSON, DICOM  │  │  Pseudonymized │  │  Schema    │        │
│  └─────────────────┘  └────────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd M233_EHDS
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the Integration Pipeline

The `run_pipeline.py` script is the main entry point for data processing.

```bash
# Show all available options
python run_pipeline.py --help

# Run the full Data Lake pipeline (generates and processes sample data)
python run_pipeline.py --data-lake --run-all

# Run specific integration flows (e.g., synthea and lab)
python run_pipeline.py --flows synthea lab
```

### Launching the Dashboard

The `dashboard.py` script starts an interactive Streamlit dashboard.

```bash
streamlit run dashboard.py
```

## 📂 Project Structure

```
M233_EHDS/
├── data/                # Data Lake storage (Bronze, Silver, Gold, Semantic)
├── docs/                # Project documentation & architecture diagrams
├── ehds/                # Core Python package (pipelines, transformers, utils)
├── sources/             # Source data connectors (Synthea, MIMIC, Lab, DICOM)
├── dashboard.py         # Streamlit dashboard application
├── run_pipeline.py      # Main CLI for running pipelines
└── requirements.txt     # Python dependencies
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

-   [Architecture Overview](docs/ARCHITECTURE.md)
-   [Data Lake Pipeline](docs/README_DATA_LAKE.md)
-   [Integration Flows](docs/DOC_FLUX_INTEGRATION.md)
-   [Semantic Interoperability](docs/part_3_interoperabilite_semantique.md)

## 🛠️ Technologies

-   **Python**: Core language for pipelines and transformations.
-   **Pandas & Parquet**: Data manipulation and efficient columnar storage.
-   **RDFLib**: Semantic layer (RDF graph, SPARQL queries).
-   **Streamlit & Plotly**: Interactive dashboard and visualizations.
-   **pydicom**: DICOM image metadata extraction.
