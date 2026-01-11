# Architecture d’intégration recommandée pour le projet EHDS

## 1. Contexte et objectif

Dans le cadre de ce projet portant sur le European Health Data Space (EHDS), l’objectif principal est de concevoir et d’implémenter une architecture d’intégration de données **réaliste, reproductible et techniquement maîtrisable**, tout en respectant les principes fondamentaux de l’EHDS, notamment l’interopérabilité, la réutilisation des données et la conformité réglementaire.

Les besoins d’intégration identifiés dans la partie 2.1 mettent en évidence :

* une forte hétérogénéité des sources (FHIR, CSV, DICOM, JSON) ;
* la coexistence d’usages primaires et secondaires des données ;
* des contraintes importantes en matière de qualité, de gouvernance et de protection des données ;
* un besoin explicite d’interopérabilité sémantique fondée sur des standards tels que RDF, OWL et SPARQL.

Dans ce contexte, une architecture de type **Data Lake enrichi par une couche sémantique** a été retenue comme solution la plus adaptée.

---

## 2. Justification du choix architectural

Plusieurs modèles d’architecture de données ont été analysés, notamment le data warehouse, le data mesh et le data fabric. Toutefois, ces approches présentent des limites significatives dans le cadre de ce projet.

Le data warehouse impose un schéma rigide et une modélisation en amont peu compatible avec la diversité des formats manipulés (FHIR, DICOM, données semi-structurées). Les approches data mesh et data fabric, quant à elles, reposent sur des transformations organisationnelles et des infrastructures complexes difficilement justifiables dans un cadre académique et expérimental.

À l’inverse, le data lake permet d’absorber des données hétérogènes sans contrainte de schéma initial, tout en offrant une grande flexibilité pour des traitements ultérieurs. En l’enrichissant d’une couche sémantique dédiée, il devient possible de répondre explicitement aux exigences d’interopérabilité sémantique de l’EHDS, tout en conservant une architecture simple et implémentable.

Cette approche constitue ainsi un compromis pertinent entre faisabilité technique, valeur scientifique et alignement avec les principes européens des espaces de données.

---

## 3. Architecture cible proposée

L’architecture retenue est organisée en couches fonctionnelles distinctes, chacune répondant à des besoins spécifiques du cycle d’intégration des données.

### 3.1. Couche d’ingestion – Zone Bronze (Raw/Bronze Zone)

La première couche correspond à la zone d'ingestion brute du data lake, organisée selon le modèle **Bronze/Silver/Gold**. Elle accueille les données telles qu'elles sont produites par les sources, sans transformation majeure (principe du **schema-on-read**).

Cette zone Bronze contient notamment :

* les dossiers médicaux électroniques synthétiques générés par Synthea au format FHIR (JSON / XML) ;
* les données cliniques anonymisées issues de MIMIC-III au format CSV ;
* les données d'imagerie médicale synthétiques au format DICOM ;
* les données de laboratoire simulées ou extraites au format CSV ou JSON.

**Caractéristiques :**
* Conservation des données dans leur format original (CSV, JSON, NDJSON, DICOM)
* Aucun schéma global imposé à ce stade
* Traçabilité via métadonnées enregistrées dans un catalogue sémantique RDF
* Stockage organisé par domaine (ehr/, lab/, fhir/, dicom/)

L'objectif principal de cette couche est de préserver l'intégrité des données sources, d'assurer leur traçabilité et de permettre leur réutilisation ultérieure.

---

### 3.2. Couche de préparation et d'harmonisation – Zone Silver (Enriched/Silver Zone)

La seconde couche (Silver) est dédiée à la préparation et à l'enrichissement des données en vue de leur exploitation. Elle comprend des traitements de nettoyage, de normalisation et d'harmonisation, avec ajout de métadonnées sémantiques.

Les opérations réalisées dans cette couche incluent notamment :

* la pseudonymisation des identifiants patients (SHA-256) ;
* la correction ou l'élimination des valeurs manquantes et incohérentes ;
* la normalisation des unités de mesure et des formats de dates (ex: créatinine µmol/L → mg/dL) ;
* l'alignement des terminologies médicales (diagnostics, examens, actes) avec LOINC, ICD-10 ;
* l'ajout de tags sémantiques et de métadonnées qualité ;
* la structuration progressive des données issues de formats semi-structurés ou relationnels ;
* conversion en format Parquet pour optimiser les performances et la compression.

**Caractéristiques :**
* Format de stockage : Parquet (partitionné par date/source si nécessaire)
* Métadonnées qualité : complétude, cohérence, flags d'anomalie
* Traçabilité : lien PROV-O vers les données Bronze source
* Catalogue RDF : enregistrement des métadonnées d'enrichissement

Cette couche joue un rôle central dans l'amélioration de la qualité des données et prépare leur projection vers des modèles plus structurés, y compris sémantiques.

---

### 3.3. Couche de curation – Zone Gold (Curated/Gold Zone)

La troisième couche (Gold) correspond à la zone curated de l'architecture originale. Elle fournit un schéma unifié et harmonisé, prêt pour l'exploitation analytique et l'usage secondaire.

**Caractéristiques :**
* Schéma unifié : intégration cross-sources avec identifiants harmonisés
* Format : Parquet optimisé avec schéma structuré
* Qualité garantie : données validées, nettoyées et enrichies
* Prêt pour : analytics, machine learning, exports pour usage secondaire

---

### 3.4. Couche sémantique – Semantic Layer

La couche sémantique constitue l'élément clé de l'architecture proposée. Elle est composée de **deux sous-couches complémentaires** :

#### 3.4.1. Catalogue sémantique (Semantic Catalog)

Un catalogue RDF décrit toutes les données des zones Bronze, Silver et Gold :
* **Métadonnées des datasets** : localisation, format, taille, date de création
* **Lignage des données** : provenance, transformations appliquées (PROV-O)
* **Métadonnées qualité** : métriques de complétude, cohérence
* **Mapping sémantique** : liens vers ontologies standardisées

#### 3.4.2. Transformation RDF complète (Data Graph)

Les données sélectionnées des zones Silver/Gold sont transformées vers un modèle RDF complet. Cette transformation repose sur l'utilisation d'ontologies et de vocabulaires standardisés :

* le modèle RDF de HL7 FHIR ;
* les terminologies médicales SNOMED CT, LOINC et ICD-10 ;
* des ontologies transverses pour la gestion du temps et des identifiants ;
* des schémas SKOS pour les vocabulaires contrôlés.

**Stockage :**
* Les données RDF sont stockées dans un **triplestore** (Apache Jena Fuseki, GraphDB ou Blazegraph)
* Interrogation via **SPARQL** pour raisonnement et requêtes complexes
* Export possible en format TTL (Turtle) pour archivage/échange

Cette couche permet de raisonner sur les données, de croiser des informations issues de sources hétérogènes et de faciliter leur réutilisation dans un contexte européen.

---

### 3.5. Couche d'accès et d'exploitation

La couche supérieure de l'architecture est dédiée à l'accès et à l'exploitation des données, en fonction des usages ciblés.

**Usage primaire (soins cliniques) :**
* API REST conforme aux principes HL7 FHIR (endpoints Patient, Observation, Condition, etc.)
* Accès principalement aux données Silver/Gold via transformation FHIR
* Garantie de performance et de disponibilité

**Usage secondaire (recherche, innovation) :**
* **Interrogation sémantique via SPARQL** : requêtes cross-sources sur le triplestore
* **Export de jeux de données harmonisés** : depuis la zone Gold (Parquet/CSV)
* **Catalogue de données** : découverte via requêtes SPARQL sur le catalogue sémantique
* **Exploitation via environnements analytiques** : notebooks, outils de visualisation (dashboard Streamlit)

**Vue d'ensemble de l'architecture :**

```
┌─────────────────────────────────────────────────────────────┐
│              COUCHE D'ACCÈS ET D'EXPLOITATION               │
│  ┌──────────────┐              ┌──────────────────┐        │
│  │  API FHIR    │              │  SPARQL          │        │
│  │ (Usage       │              │ (Usage           │        │
│  │  primaire)   │              │  secondaire)     │        │
│  └──────┬───────┘              └────────┬─────────┘        │
│         │                                │                  │
│         │                                │                  │
└─────────┼────────────────────────────────┼──────────────────┘
          │                                │
┌─────────┼────────────────────────────────┼──────────────────┐
│         │     COUCHE SÉMANTIQUE          │                  │
│         │  ┌──────────────────────────┐  │                  │
│         │  │  Triplestore             │  │                  │
│         │  │  (Jena/GraphDB)          │  │                  │
│         │  │  - Data Graph RDF        │  │                  │
│         │  │  - Catalogue RDF         │  │                  │
│         │  │  - Lignage PROV-O        │  │                  │
│         │  └──────────┬───────────────┘  │                  │
└─────────┼─────────────┼──────────────────┼──────────────────┘
          │             │                  │
┌─────────┼─────────────┼──────────────────┼──────────────────┐
│         │             │                  │                  │
│  ┌──────▼──────┐  ┌───▼──────┐    ┌─────▼──────┐          │
│  │    BRONZE   │  │  SILVER  │    │    GOLD    │          │
│  │   (Raw)     │→ │(Enriched)│ →  │ (Curated)  │          │
│  │             │  │          │    │            │          │
│  │ - CSV       │  │ - Parquet│    │ - Parquet  │          │
│  │ - JSON      │  │ - Tags   │    │ - Unified  │          │
│  │ - NDJSON    │  │ - Quality│    │   Schema   │          │
│  │ - DICOM     │  │ - Normal.│    │            │          │
│  └─────────────┘  └──────────┘    └────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Cette architecture matérialise la séparation logique entre les données brutes, les données enrichies, les données curatées et les données exposées aux utilisateurs, conformément aux principes de gouvernance de l'EHDS et aux bonnes pratiques des Data Lakes modernes.

---

## 4. Alignement avec les principes EHDS et FAIR

L’architecture proposée est alignée avec les objectifs du European Health Data Space et les principes FAIR. Elle favorise la découvrabilité des données par la structuration progressive et la couche sémantique, garantit une accessibilité contrôlée, assure l’interopérabilité grâce aux standards sémantiques et permet la réutilisation dans des contextes variés.

Par ailleurs, l’utilisation de données synthétiques et anonymisées permet de respecter les exigences du RGPD, tout en conservant un niveau de réalisme suffisant pour une expérimentation crédible.

---

## 5. Conclusion

Au regard des besoins d’intégration identifiés, des contraintes techniques et réglementaires, et des objectifs pédagogiques du projet, une architecture de type **data lake enrichi par une couche sémantique** apparaît comme la solution la plus pertinente. Elle offre un équilibre entre simplicité de mise en œuvre, conformité aux exigences de l’EHDS et valeur scientifique, tout en permettant une implémentation concrète et démontrable dans un cadre académique.

Cette architecture constitue ainsi une base solide pour la conception détaillée des flux d’intégration et des choix technologiques présentés dans les sections suivantes.
