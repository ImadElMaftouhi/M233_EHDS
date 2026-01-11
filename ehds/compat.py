"""
Fichier de compatibilité pour maintenir la compatibilité avec les anciens scripts.
Permet une migration progressive vers la nouvelle structure modulaire.
"""

# Import des nouvelles classes avec les anciens noms pour compatibilité
from .data.generators import DataGenerator as EHDSDataPreparation
from .data.integrator import DataIntegrator as EHDSDataIntegration
from .semantic.rdf_layer import RDFLayer as EHDSSemanticLayer

# Import des utilitaires
from .utils import (
    sha256_pseudo,
    deterministic_uid,
    ensure_dir,
    now_iso,
    LOINC_TESTS,
    ICD10,
    DRUG_FAMILY,
    ALLERGY_FAMILY,
)

__all__ = [
    "EHDSDataPreparation",
    "EHDSDataIntegration",
    "EHDSSemanticLayer",
    "sha256_pseudo",
    "deterministic_uid",
    "ensure_dir",
    "now_iso",
    "LOINC_TESTS",
    "ICD10",
    "DRUG_FAMILY",
    "ALLERGY_FAMILY",
]
