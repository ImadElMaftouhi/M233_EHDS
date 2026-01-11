"""Point d'entrée principal pour exécuter les pipelines EHDS."""

from __future__ import annotations

import argparse
from pathlib import Path

from ehds.pipelines import run_all_flows, run_data_lake_pipeline


def main():
    """Point d'entrée principal avec interface CLI."""
    parser = argparse.ArgumentParser(
        description="EHDS Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python run_pipeline.py --data-lake --run-all
  python run_pipeline.py --flows synthea lab
  python run_pipeline.py --data-lake --data-dir ./data
        """,
    )

    parser.add_argument("--data-dir", type=str, default="data", help="Répertoire de données")
    parser.add_argument("--n-patients", type=int, default=120, help="Nombre de patients (génération)")
    parser.add_argument("--n-labs", type=int, default=600, help="Nombre de résultats labo (génération)")

    # Choix du pipeline
    parser.add_argument("--data-lake", action="store_true", help="Exécuter le pipeline Data Lake")
    parser.add_argument("--flows", nargs="+", choices=["synthea", "mimic", "lab", "dicom"], help="Flux à exécuter")
    parser.add_argument("--run-all", action="store_true", help="Exécuter tous les flux/pipelines")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.data_lake:
        print("Exécution du pipeline Data Lake...")
        run_data_lake_pipeline(data_dir=data_dir, n_patients=args.n_patients, n_labs=args.n_labs)
    elif args.flows or args.run_all:
        print("Exécution des flux d'intégration...")
        run_all_flows(data_dir=data_dir, flow_names=args.flows if args.flows else None)
    else:
        print("Veuillez spécifier un pipeline à exécuter:")
        print("  --data-lake : Pipeline Data Lake complet")
        print("  --flows synthea lab : Flux d'intégration spécifiques")
        print("  --run-all : Tous les flux")
        print("\nExemple: python run_pipeline.py --data-lake --run-all")


if __name__ == "__main__":
    main()
