"""Utilitaires pour la gestion des fichiers et répertoires."""

from datetime import datetime
from pathlib import Path


def ensure_dir(p: Path) -> None:
    """
    Crée un répertoire s'il n'existe pas déjà.
    
    Args:
        p: Chemin du répertoire à créer
    """
    p.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """
    Retourne la date/heure actuelle au format ISO sans microsecondes.
    
    Returns:
        Date/heure au format ISO (YYYY-MM-DDTHH:MM:SS)
    """
    return datetime.now().replace(microsecond=0).isoformat()
