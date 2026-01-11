"""Fonctions de hachage pour pseudonymisation et génération d'identifiants."""

import hashlib


def sha256_pseudo(value: str, keep: int = 16) -> str:
    """
    Pseudonymise une valeur en utilisant SHA-256.
    
    Args:
        value: Valeur à pseudonymiser
        keep: Nombre de caractères hexadécimaux à conserver (défaut: 16)
    
    Returns:
        Hash SHA-256 tronqué
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:keep]


def deterministic_uid(seed: str) -> str:
    """
    Génère un UID DICOM déterministe à partir d'une graine.
    
    Args:
        seed: Chaîne de caractères utilisée comme graine
    
    Returns:
        UID DICOM au format 1.2.826.0.1.3680043.10.543.{num}
    """
    num = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return f"1.2.826.0.1.3680043.10.543.{num % 10**18}"
