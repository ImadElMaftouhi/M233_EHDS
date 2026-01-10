import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from rdflib import Graph

st.set_page_config(page_title="EHDS Data Integration", layout="wide")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "integrated" / "ehds.db"
TTL_PATH = DATA_DIR / "rdf" / "ehds_data.ttl"

st.title("🏥 European Health Data Space (EHDS) — Prototype")
st.markdown("**Partie II: Intégration des données** | **Partie III: Interopérabilité sémantique (RDF/SKOS/OWL/SPARQL)**")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Overview",
        "Data Sources",
        "Integrated DB",
        "Integration Pipeline",
        "Data Quality",
        "Imaging (DICOM)",
        "Semantic Graph",
        "SPARQL Queries",
    ],
)

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


@st.cache_resource
def load_graph() -> Graph:
    g = Graph()
    if TTL_PATH.exists():
        g.parse(str(TTL_PATH), format="turtle")
    return g


def db_ready() -> bool:
    return DB_PATH.exists()


def table_exists(table_name: str) -> bool:
    if not DB_PATH.exists():
        return False
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def ttl_ready() -> bool:
    return TTL_PATH.exists()


# -----------------------
# Pages
# -----------------------
if page == "Overview":
    st.header("📊 System Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    patients = load_table("patients")
    labs = load_table("lab_results")
    prescriptions = load_table("prescriptions")
    allergies = load_table("allergies")
    dicom_images = load_table("dicom_images")

    with col1:
        st.metric("Patients", int(len(patients)) if not patients.empty else 0)
    with col2:
        st.metric("Lab Results", int(len(labs)) if not labs.empty else 0)
    with col3:
        st.metric("Prescriptions", int(len(prescriptions)) if not prescriptions.empty else 0)
    with col4:
        st.metric("Allergies", int(len(allergies)) if not allergies.empty else 0)
    with col5:
        st.metric("DICOM Instances", int(len(dicom_images)) if not dicom_images.empty else 0)

    if db_ready() and not table_exists("dicom_images"):
        st.warning("dicom_images table not found. Re-run: `python ehds_integration.py --run-all`")

    st.info(
        "If values are 0, run the pipeline first:\n"
        "`python ehds_integration.py --run-all`"
    )

    st.subheader("✅ What this prototype demonstrates")
    st.markdown(
        "- **4 heterogeneous sources**: EHR CSV + Lab JSON + FHIR-like NDJSON + DICOM\n"
        "- **ETL integration** into SQLite (unified IDs + unit normalization)\n"
        "- **Semantic layer**: RDF graph + SKOS concept schemes + SPARQL queries\n"
        "- **Course scenario**: unit conversion + allergy↔prescription contraindication\n"
    )

elif page == "Data Sources":
    st.header("Data Sources (Raw)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Source 1 — EHR (CSV)**")
        p = DATA_DIR / "source_ehr_csv" / "ehr_patients.csv"
        st.write(str(p))
        if p.exists():
            st.dataframe(pd.read_csv(p).head(15))

    with col2:
        st.markdown("**Source 2 — Labs (JSON)**")
        p = DATA_DIR / "source_lab_json" / "lab_results.json"
        st.write(str(p))
        if p.exists():
            df = pd.read_json(p)
            st.dataframe(df.head(15))

    with col3:
        st.markdown("**Source 3 - FHIR-like (NDJSON)**")
        p = DATA_DIR / "source_fhir_ndjson" / "bundle.ndjson"
        st.write(str(p))
        if p.exists():
            # Show first lines only
            lines = p.read_text(encoding="utf-8").splitlines()[:10]
            st.code("\n".join(lines), language="json")
    with col4:
        st.markdown("**Source 4 - DICOM**")
        dicom_dir = DATA_DIR / "source_dicom"
        st.write(str(dicom_dir))
        if dicom_dir.exists():
            count = sum(1 for _ in dicom_dir.rglob("*.dcm"))
            st.write(f"{count} .dcm files")
        if not table_exists("dicom_images"):
            st.warning("dicom_images table not found. Re-run: `python ehds_integration.py --run-all`")
        else:
            dicom = load_table("dicom_images")
            st.dataframe(dicom.head(10))

elif page == "Integrated DB":
    st.header("🗄️ Integrated SQLite Database")

    if not db_ready():
        st.warning("DB not found. Run: `python ehds_integration.py --run-all`")
    else:
        table = st.selectbox(
            "Select table",
            ["patients", "lab_results", "conditions", "allergies", "prescriptions", "dicom_images"],
        )
        if table == "dicom_images" and not table_exists("dicom_images"):
            st.warning("dicom_images table not found. Re-run: `python ehds_integration.py --run-all`")
        df = load_table(table)
        st.dataframe(df)

elif page == "Integration Pipeline":
    st.header("🔄 Data Integration Pipeline (ETL)")

    stages = [
        ("1) Extraction", "Read CSV (EHR), JSON (Labs), NDJSON (FHIR-like)"),
        ("2) Pseudonymization", "Hash patient_id -> patient_id_pseudo (SHA-256)"),
        ("3) Standardization", "LOINC codes, ICD-10 labels, normalize creatinine unit µmol/L→mg/dL"),
        ("4) Imaging", "Generate and parse DICOM instances (study/series/instance metadata)"),
        ("5) Integration", "Unified schema in SQLite: patients / labs / conditions / allergies / prescriptions / dicom_images"),
        ("6) Quality checks", "Completeness + abnormal flags based on reference ranges"),
        ("7) Semantic layer", "RDF + SKOS concept schemes + SPARQL queries"),
    ]

    for title, detail in stages:
        with st.expander(title):
            st.write(detail)

    st.success("This matches the integration + semantic interoperability expectations from the course (unit harmonization + dictionary + allergy linking).")

elif page == "Data Quality":
    st.header("✨ Data Quality")

    labs = load_table("lab_results")
    if labs.empty:
        st.warning("No lab_results yet. Run pipeline first.")
    else:
        completeness = (labs.notnull().sum() / len(labs) * 100).mean()
        abnormal_rate = float(labs["is_abnormal"].mean()) * 100 if "is_abnormal" in labs.columns else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Completeness (avg)", f"{completeness:.1f}%")
        c2.metric("Abnormal rate", f"{abnormal_rate:.1f}%")
        c3.metric("Records", str(len(labs)))

        st.subheader("Missing values")
        missing = labs.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("✓ No missing values detected.")
        else:
            st.dataframe(missing)

        st.subheader("Test distribution")
        fig = px.bar(labs["test_name"].value_counts().reset_index(), x="test_name", y="count")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Imaging (DICOM)":
    st.header("🧩 Imaging (DICOM)")

    if not db_ready():
        st.warning("DB not found. Run: `python ehds_integration.py --run-all`")
    else:
        if not table_exists("dicom_images"):
            st.warning("dicom_images table not found. Re-run: `python ehds_integration.py --run-all`")
        dicom_images = load_table("dicom_images")
        total = int(len(dicom_images)) if not dicom_images.empty else 0
        unique_patients = 0
        if not dicom_images.empty and "patient_id_pseudo" in dicom_images.columns:
            unique_patients = int(dicom_images["patient_id_pseudo"].nunique())
        st.metric("DICOM Instances", total)
        st.metric("Patients with Imaging", unique_patients)

        if dicom_images.empty:
            st.warning("No DICOM data yet. Run pipeline first.")
        else:
            st.subheader("Modality distribution")
            modality_counts = dicom_images["modality"].value_counts().reset_index()
            modality_counts.columns = ["modality", "count"]
            fig = px.bar(modality_counts, x="modality", y="count")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("DICOM preview")
            show_paths = st.checkbox("Show file paths", value=False)
            preview = dicom_images.head(50)
            if not show_paths and "file_path" in preview.columns:
                preview = preview.drop(columns=["file_path"])
            st.dataframe(preview)

elif page == "Semantic Graph":
    st.header("🧠 Semantic Layer (RDF/SKOS/OWL-ish)")

    if not ttl_ready():
        st.warning("TTL not found. Run: `python ehds_integration.py --run-all`")
    else:
        g = load_graph()
        st.metric("Triples", len(g))

        st.subheader("Preview TTL")
        ttl_text = TTL_PATH.read_text(encoding="utf-8").splitlines()[:80]
        st.code("\n".join(ttl_text), language="turtle")

        st.subheader("What’s inside")
        st.markdown(
            "- **Ontology**: classes Patient/LabResult/Test/Drug/Allergy…\n"
            "- **SKOS**: concept schemes for LOINC + ICD-10; drug families with broader/narrower\n"
            "- **Relations**: `affectsFamily` (allergy→family) + `belongsToFamily` (drug→family)\n"
            "→ enables contraindication detection with SPARQL."
        )

elif page == "SPARQL Queries":
    st.header("🔍 SPARQL Query Interface")

    if not ttl_ready():
        st.warning("TTL not found. Run: `python ehds_integration.py --run-all`")
    else:
        g = load_graph()

        predefined = {
            "Count Patients": """
PREFIX ehds: <http://ehds.eu/ontology#>
SELECT (COUNT(DISTINCT ?p) AS ?count)
WHERE { ?p a ehds:Patient . }
            """,
            "Abnormal Glucose (>140)": """
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
            "Contraindication Alerts (Allergy ↔ Drug Family)": """
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
            "Count DICOM instances per modality": """
PREFIX ehds: <http://ehds.eu/ontology#>
SELECT ?modality (COUNT(?inst) AS ?count)
WHERE {
  ?inst a ehds:DicomInstance ;
        ehds:modality ?modality .
}
GROUP BY ?modality
ORDER BY DESC(?count)
            """,
            "Patients with imaging + abnormal glucose (>140)": """
PREFIX ehds: <http://ehds.eu/ontology#>
SELECT ?patient ?modality ?val ?date
WHERE {
  ?inst a ehds:DicomInstance ;
        ehds:hasPatient ?patient ;
        ehds:modality ?modality .
  ?lab a ehds:LabResult ;
       ehds:hasPatient ?patient ;
       ehds:label "Glucose" ;
       ehds:value ?val ;
       ehds:date ?date .
  FILTER (?val > 140)
}
LIMIT 20
            """,
        }

        mode = st.radio("Mode", ["Predefined", "Custom"], horizontal=True)

        if mode == "Predefined":
            choice = st.selectbox("Select query", list(predefined.keys()))
            query = predefined[choice]
        else:
            query = st.text_area("Write SPARQL query", height=220)

        st.code(query, language="sparql")

        if st.button("Execute SPARQL"):
            try:
                res = g.query(query)
                rows = [list(r) for r in res]
                cols = [str(v) for v in res.vars] if res.vars else [f"col{i}" for i in range(len(rows[0]))]
                df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
                st.success("✓ Query executed")
                st.dataframe(df)
            except Exception as e:
                st.error(f"SPARQL error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "**EHDS Prototype**\n"
    "- Run pipeline: `python ehds_integration.py --run-all`\n"
    "- Run dashboard: `streamlit run dashboard.py`"
)
