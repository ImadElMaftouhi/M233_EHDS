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

## 📋 Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **~2 GB free disk space** (for generated data)

### Optional (for real data sources):
- **Java 11+** (for Synthea FHIR generation)
- **MIMIC-III access** (PhysioNet credentialed access)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd M233_EHDS
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, rdflib, streamlit, pydicom; print('✓ All dependencies installed')"
```

## 🚀 Quick Start

### Option 1: Run with Auto-Generated Data (Recommended for First Run)

This will automatically generate simulated data for Lab and DICOM sources:

```bash
# Run the full Data Lake pipeline
python run_pipeline.py --data-lake --run-all

# Launch the dashboard
streamlit run dashboard.py
```

The pipeline will:
1. Generate simulated Lab (JSON) and DICOM data
2. Process all available sources (MIMIC-III and Synthea FHIR if present)
3. Create Bronze → Silver → Gold zones
4. Build the semantic RDF layer
5. Execute sample SPARQL queries

### Option 2: Run Individual Integration Flows

```bash
# Run all flows (synthea, mimic, lab, dicom)
python run_pipeline.py --run-all

# Run specific flows
python run_pipeline.py --flows lab dicom

# Run only MIMIC-III flow
python run_pipeline.py --flows mimic
```

## 📊 Data Sources Configuration

The pipeline supports multiple data sources with different setup requirements:

| Source | Auto-Generated? | Manual Setup | Location |
|--------|:---------------:|--------------|----------|
| **Lab Results** | ✅ Yes | None | `data/source_lab_json/lab_results.json` |
| **DICOM Images** | ✅ Yes | None | `data/source_dicom/` |
| **EHR (CSV)** | ⚠️ Available but not used | Generator available | `data/source_ehr_csv/ehr_patients.csv` |
| **MIMIC-III** | ❌ No | Download from PhysioNet | `data/source_mimic_csv/` |
| **Synthea FHIR** | ❌ No | Generate with Synthea | `data/source_fhir_ndjson/bundle.ndjson` |

### Using Real MIMIC-III Data (Recommended)

**MIMIC-III** is the primary patient data source. Follow these steps:

1. **Download MIMIC-III** from [PhysioNet](https://physionet.org/content/mimiciii/) (requires credentialed access)
   - You need: `PATIENTS.csv`, `ADMISSIONS.csv`, `LABEVENTS.csv`, `PRESCRIPTIONS.csv`

2. **Create the directory structure:**
   ```bash
   mkdir -p data/source_mimic_csv
   ```

3. **Copy the CSV files** (file names must be **UPPERCASE**):
   ```
   data/source_mimic_csv/
   ├── PATIENTS.csv
   ├── ADMISSIONS.csv
   ├── LABEVENTS.csv
   └── PRESCRIPTIONS.csv
   ```

4. **Run the pipeline:**
   ```bash
   python run_pipeline.py --data-lake --run-all
   ```

The pipeline will automatically detect and use MIMIC-III data instead of generating EHR data.

### Using Real Synthea FHIR Data (Optional)

**Synthea** generates realistic FHIR resources for testing.

1. **Install Synthea** (requires Java 11+):
   ```bash
   git clone https://github.com/synthetichealth/synthea.git
   cd synthea
   ./gradlew build check test
   ```

2. **Generate FHIR data:**
   ```bash
   ./run_synthea -p 100 --exporter.fhir.export=true --exporter.fhir.bulk_data=true
   ```

3. **Merge NDJSON files into a single bundle:**
   
   **Windows (PowerShell):**
   ```powershell
   Get-Content .\output\fhir\*.ndjson | Set-Content .\output\fhir\bundle.ndjson
   ```
   
   **Linux/Mac (Bash):**
   ```bash
   cat output/fhir/*.ndjson > output/fhir/bundle.ndjson
   ```

4. **Copy to project:**
   ```bash
   # Create directory
   mkdir -p data/source_fhir_ndjson
   
   # Copy the bundle
   cp <synthea-path>/output/fhir/bundle.ndjson data/source_fhir_ndjson/
   ```

5. **Run the pipeline:**
   ```bash
   python run_pipeline.py --data-lake --run-all
   ```

## 🔧 Pipeline Commands

### Data Lake Pipeline (Unified Integration)

```bash
# Full pipeline with all available sources
python run_pipeline.py --data-lake --run-all

# Custom data directory
python run_pipeline.py --data-lake --run-all --data-dir ./my_data

# Help
python run_pipeline.py --help
```

**What it does:**
- Detects all available data sources
- Ingests into Bronze zone
- Enriches to Silver zone
- Creates unified Gold schema using `DataIntegrator`
- Builds global RDF semantic layer

### Integration Flows (Modular Approach)

```bash
# Run all flows
python run_pipeline.py --run-all

# Run specific flows
python run_pipeline.py --flows synthea mimic
python run_pipeline.py --flows lab dicom

# Available flows: synthea, mimic, lab, dicom
```

**What it does:**
- Each flow runs independently (Extract → Load → Transform Silver → Transform Gold → Transform RDF)
- Produces separate Gold datasets per flow
- Builds RDF graphs per flow

### Differences

| Aspect | `--data-lake --run-all` | `--run-all` |
|--------|------------------------|-------------|
| **Approach** | Unified (single Gold dataset) | Modular (one Gold per flow) |
| **Integration** | Uses `DataIntegrator.integrate()` | Each flow independent |
| **Use Case** | Final integration | Development/testing |

## 📊 Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard provides:
- **Overview**: System metrics and data counts
- **Data Lake Zones**: Browse Bronze, Silver, Gold, and Semantic layers
- **Data Sources**: Preview raw source files
- **Integrated DB**: Explore unified SQLite database
- **Data Quality**: Completeness and anomaly detection
- **Imaging (DICOM)**: DICOM metadata exploration
- **Semantic Graph**: RDF graph visualization and TTL preview
- **SPARQL Queries**: Execute and test SPARQL queries

Access at: `http://localhost:8501`

## 📂 Project Structure

```
M233_EHDS/
├── data/                    # Data Lake storage (created automatically)
│   ├── bronze/              # Raw ingested data
│   ├── silver/              # Enriched data (Parquet)
│   ├── gold/                # Curated unified schema
│   ├── semantic/            # RDF catalog and data graph
│   ├── integrated/          # SQLite unified database
│   ├── rdf/                 # RDF exports
│   └── source_*/            # Source data directories
│
├── ehds/                    # Core Python package
│   ├── data/                # Data generators and integrators
│   ├── pipelines/           # ELT pipelines
│   ├── semantic/            # RDF layer and data lake
│   └── utils/               # Utilities (hashing, constants)
│
├── docs/                    # Project documentation
│   ├── ARCHITECTURE.md      # Architecture details
│
├── dashboard.py             # Streamlit dashboard
├── run_pipeline.py          # Main CLI entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🔍 Understanding the Output

After running the pipeline, you'll find:

### Data Lake Zones

- **Bronze** (`data/bronze/`): Raw ingested files (CSV, JSON, NDJSON, DICOM)
- **Silver** (`data/silver/`): Enriched Parquet files with pseudonymization and quality flags
- **Gold** (`data/gold/`): Unified schema Parquet files
- **Semantic** (`data/semantic/`):
  - `catalog.ttl`: RDF catalog with PROV-O lineage
  - `ehds_data_graph.ttl`: RDF data graph with all entities

### Integrated Database

- **SQLite** (`data/integrated/ehds.db`): Unified relational database with tables:
  - `patients`
  - `lab_results`
  - `conditions`
  - `allergies`
  - `prescriptions`
  - `dicom_images`

### RDF Semantic Layer

- **Ontologies**: EHDS, FHIR, LOINC, ICD-10, SKOS
- **Triples**: Patient, LabResult, Condition, Drug, Allergy entities
- **SPARQL**: Query interface for semantic queries

## 🧪 Testing SPARQL Queries

The pipeline includes predefined SPARQL queries. You can also run custom queries via the dashboard or programmatically:

```python
from rdflib import Graph

g = Graph()
g.parse("data/semantic/ehds_data_graph.ttl", format="turtle")

query = """
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
LIMIT 10
"""

for row in g.query(query):
    print(row)
```

## 🐛 Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
# Ensure virtual environment is activated
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "MIMIC-III data not found"

**Solution:**
- Verify files are in `data/source_mimic_csv/`
- Check file names are **UPPERCASE**: `PATIENTS.CSV` (not `patients.csv`)
- Ensure files are valid CSV format

### Issue: "UTF-8 decoding error" (Synthea FHIR)

**Solution:**
The pipeline handles encoding errors automatically. If issues persist:
- Check file encoding: `file -i bundle.ndjson` (Linux/Mac)
- Re-generate Synthea data with UTF-8 encoding

### Issue: "Unsupported format" errors

**Solution:**
- Ensure source files are in correct format (CSV, JSON, NDJSON)
- Check file extensions match expected format
- Verify files are not corrupted

### Issue: Dashboard shows 0 values

**Solution:**
```bash
# Run the pipeline first
python run_pipeline.py --data-lake --run-all

# Then launch dashboard
streamlit run dashboard.py
```

### Issue: Port 8501 already in use

**Solution:**
```bash
streamlit run dashboard.py --server.port 8502
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Architecture Overview](docs/ARCHITECTURE.md)**: System architecture and design decisions
- **[Data Lake Pipeline](docs/README_DATA_LAKE.md)**: Data Lake zone details
- **[Integration Flows](docs/DOC_FLUX_INTEGRATION.md)**: Individual flow documentation
- **[Semantic Interoperability](docs/part_3_interoperabilite_semantique.md)**: RDF, SKOS, OWL, SPARQL details

## 🛠️ Technologies

- **Python 3.10+**: Core language
- **Pandas**: Data manipulation
- **PyArrow**: Parquet file format
- **RDFLib**: RDF graph and SPARQL queries
- **Streamlit**: Interactive dashboard
- **Plotly**: Data visualizations
- **pydicom**: DICOM metadata extraction

**Note**: This is a prototype implementation for educational/research purposes. For production use, additional security, performance, and compliance measures would be required.
