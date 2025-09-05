import os
import zipfile
from pathlib import Path
from typing import Optional, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ZIP_CANDIDATES = [
    ROOT / "dataset (1).zip",
    ROOT / "dataset.zip",
]
DATASET_DESCRIPTION = ROOT / "dataset_description.xlsx"


def find_zip() -> Optional[Path]:
    for p in DATASET_ZIP_CANDIDATES:
        if p.exists():
            return p
    return None


def unzip_dataset(zip_path: Path, dest_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            # normalize paths
            target_path = dest_dir / member.filename
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            extracted.append(target_path)
    return extracted


def catalog_files(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob('*'):
        if p.is_file():
            rows.append({
                'relative_path': str(p.relative_to(root)),
                'suffix': p.suffix.lower(),
                'size_bytes': p.stat().st_size,
            })
    return pd.DataFrame(rows).sort_values('relative_path')


def profile_csv(path: Path, sample_rows: int = 50000) -> dict:
    """Create a lightweight profile for a CSV and write a parquet sample."""
    info = {'path': str(path), 'rows_estimate': None, 'sample_rows_written': 0, 'columns': []}
    # Count rows (could be slow; ok once during ingestion)
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # subtract header
            total_lines = sum(1 for _ in f)
        info['rows_estimate'] = max(total_lines - 1, 0)
    except Exception as e:
        print(f"[WARN] Failed to count lines for {path}: {e}")
    # Read a sample to infer schema and create a parquet sample for fast EDA
    try:
        df_sample = pd.read_csv(path, nrows=sample_rows)
        info['columns'] = [f"{c}:{str(df_sample[c].dtype)}" for c in df_sample.columns]
        out_path = PROC_DIR / f"{path.stem}.sample.parquet"
        df_sample.to_parquet(out_path, index=False)
        info['sample_rows_written'] = len(df_sample)
    except Exception as e:
        print(f"[WARN] Failed to sample/convert {path}: {e}")
    return info

def try_prepare_tabular(root: Path, out_dir: Path) -> List[Path]:
    """For each tabular file, write a parquet sample and collect schema info markdown."""
    markdown_lines: List[str] = ["# Tabular File Profiles", ""]
    produced: List[Path] = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        low = p.suffix.lower()
        if low == '.csv':
            info = profile_csv(p)
            markdown_lines.append(f"## {p.name}")
            markdown_lines.append(f"- Approx rows: {info.get('rows_estimate')}")
            markdown_lines.append(f"- Sample rows written: {info.get('sample_rows_written')}")
            if info.get('columns'):
                markdown_lines.append("- Columns:")
                for col in info['columns']:
                    markdown_lines.append(f"  - {col}")
            markdown_lines.append("")
            out_path = out_dir / f"{p.stem}.sample.parquet"
            if out_path.exists():
                produced.append(out_path)
        elif low in {'.xlsx', '.xls'}:
            # Convert entire sheet(s) to parquet if small
            try:
                df = pd.read_excel(p)
                out_path = out_dir / f"{p.stem}.parquet"
                df.to_parquet(out_path, index=False)
                produced.append(out_path)
                markdown_lines.append(f"## {p.name}")
                markdown_lines.append(f"- Rows: {len(df)}")
                markdown_lines.append(f"- Columns: {len(df.columns)}")
                markdown_lines.append("")
            except Exception as e:
                print(f"[WARN] Failed to convert {p}: {e}")
        else:
            continue
    # Write schema summary
    schema_md = INTERIM_DIR / 'tabular_profiles.md'
    with open(schema_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))
    return produced


def convert_csv_to_parquet_full(csv_path: Path, out_path: Path, chunksize: int = 500_000) -> Optional[Path]:
    """Convert a CSV to a single Parquet file using chunked writes to control memory.
    Returns the output path if written.
    """
    try:
        if out_path.exists():
            return out_path
        writer = None
        total_rows = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
            writer.write_table(table)
            total_rows += len(chunk)
        if writer is not None:
            writer.close()
            print(f"Wrote {out_path} (rows≈{total_rows})")
            return out_path
    except Exception as e:
        print(f"[WARN] Failed full conversion {csv_path} -> {out_path}: {e}")
    return None


def convert_all_csvs_full(root: Path, out_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    for p in root.rglob('*.csv'):
        out_p = out_dir / (p.stem + '.parquet')
        res = convert_csv_to_parquet_full(p, out_p)
        if res:
            out_paths.append(res)
    return out_paths


def write_report(zip_found: Optional[Path], extracted_count: int, catalog: pd.DataFrame, converted: List[Path]):
    report_path = INTERIM_DIR / 'ingest_report.md'
    lines = [
        '# Ingestion Report',
        '',
        f'- Zip found: {zip_found if zip_found else "None"}',
        f'- Files extracted: {extracted_count}',
        f'- Catalog rows: {len(catalog)}',
        f'- Parquet files written: {len(converted)}',
        '',
        '## Sample of catalog',
        '',
    ]
    sample = catalog.head(50)
    if not sample.empty:
        lines.append(sample.to_markdown(index=False))
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Report written to {report_path}")


def main():
    zip_path = find_zip()
    extracted: List[Path] = []
    if zip_path:
        print(f"Extracting {zip_path} -> {RAW_DIR}")
        extracted = unzip_dataset(zip_path, RAW_DIR)
    else:
        print("No dataset zip found in project root or already extracted. Continuing.")

    cat = catalog_files(RAW_DIR)
    converted = try_prepare_tabular(RAW_DIR, PROC_DIR)
    # Also write full parquet files for all CSVs (chunked) for actual pipeline use
    full_converted = convert_all_csvs_full(RAW_DIR, PROC_DIR)
    converted.extend(full_converted)

    if DATASET_DESCRIPTION.exists():
        try:
            desc_df = pd.read_excel(DATASET_DESCRIPTION)
            desc_out = PROC_DIR / 'dataset_description.parquet'
            desc_df.to_parquet(desc_out, index=False)
            converted.append(desc_out)
            print(f"Wrote {desc_out}")
        except Exception as e:
            print(f"[WARN] Failed to process dataset_description.xlsx: {e}")

    write_report(zip_path, len(extracted), cat, converted)


if __name__ == '__main__':
    main()
