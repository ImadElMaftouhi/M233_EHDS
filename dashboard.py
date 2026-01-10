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
    ["Overview", "Data Sources", "Integrated DB", "Integration Pipeline", "Data Quality", "Semantic Graph", "SPARQL Queries"],
)

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
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


def ttl_ready() -> bool:
    return TTL_PATH.exists()


# -----------------------
# Pages
# -----------------------
if page == "Overview":
    st.header("📊 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    patients = load_table("patients")
    labs = load_table("lab_results")
    prescriptions = load_table("prescriptions")
    allergies = load_table("allergies")

    with col1:
        st.metric("Patients", int(len(patients)) if not patients.empty else 0)
    with col2:
        st.metric("Lab Results", int(len(labs)) if not labs.empty else 0)
    with col3:
        st.metric("Prescriptions", int(len(prescriptions)) if not prescriptions.empty else 0)
    with col4:
        st.metric("Allergies", int(len(allergies)) if not allergies.empty else 0)

    st.info(
        "If values are 0, run the pipeline first:\n"
        "`python ehds_integration.py --run-all`"
    )

    st.subheader("✅ What this prototype demonstrates")
    st.markdown(
        "- **3 heterogeneous sources**: EHR CSV + Lab JSON + FHIR-like NDJSON\n"
        "- **ETL integration** into SQLite (unified IDs + unit normalization)\n"
        "- **Semantic layer**: RDF graph + SKOS concept schemes + SPARQL queries\n"
        "- **Course scenario**: unit conversion + allergy↔prescription contraindication\n"
    )

elif page == "Data Sources":
    st.header("📁 Data Sources (Raw)")

    col1, col2, col3 = st.columns(3)
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
        st.markdown("**Source 3 — FHIR-like (NDJSON)**")
        p = DATA_DIR / "source_fhir_ndjson" / "bundle.ndjson"
        st.write(str(p))
        if p.exists():
            # Show first lines only
            lines = p.read_text(encoding="utf-8").splitlines()[:10]
            st.code("\n".join(lines), language="json")

elif page == "Integrated DB":
    st.header("🗄️ Integrated SQLite Database")

    if not db_ready():
        st.warning("DB not found. Run: `python ehds_integration.py --run-all`")
    else:
        table = st.selectbox("Select table", ["patients", "lab_results", "conditions", "allergies", "prescriptions"])
        df = load_table(table)
        st.dataframe(df)

elif page == "Integration Pipeline":
    st.header("🔄 Data Integration Pipeline (ETL)")

    stages = [
        ("1) Extraction", "Read CSV (EHR), JSON (Labs), NDJSON (FHIR-like)"),
        ("2) Pseudonymization", "Hash patient_id -> patient_id_pseudo (SHA-256)"),
        ("3) Standardization", "LOINC codes, ICD-10 labels, normalize creatinine unit µmol/L→mg/dL"),
        ("4) Integration", "Unified schema in SQLite: patients / labs / conditions / allergies / prescriptions"),
        ("5) Quality checks", "Completeness + abnormal flags based on reference ranges"),
        ("6) Semantic layer", "RDF + SKOS concept schemes + SPARQL queries"),
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
