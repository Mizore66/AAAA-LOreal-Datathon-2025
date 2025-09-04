from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"

# Placeholder for Phase 2 processing and feature engineering steps.

def load_all_parquet() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in PROC_DIR.glob('*.parquet'):
        try:
            out[p.stem] = pd.read_parquet(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return out
