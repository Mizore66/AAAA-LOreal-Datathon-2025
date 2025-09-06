# L'Oréal x Monash Datathon 2025: TrendSpotter

A focused, multi-platform trend radar to detect emerging beauty trends across text and audio signals, with a lightweight Streamlit prototype.

## Project structure

```
project_root/
├── assets/               # Images, logos, etc.
├── data/
│   ├── raw/              # Original data (unzipped here)
│   ├── processed/        # Cleaned and preprocessed data (e.g., parquet)
│   └── interim/          # Intermediate outputs & reports
├── models/               # Saved models
├── notebooks/            # EDA & experiments
├── src/                  # Source code
│   ├── ingest_provided_data.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── requirements.txt      # Dependencies
└── README.md
```

## Quickstart (Windows PowerShell)

1) Python environment (recommended but optional)

```
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Install minimal packages to run ingestion (full list in `requirements.txt`)

```
pip install pandas pyarrow openpyxl
```

3) Place provided files in project root (already present):
- `dataset (1).zip`
- `dataset_description.xlsx`

4) Run ingestion to extract the zip, catalog files, and convert tabular data to parquet

```
python .\src\ingest_provided_data.py
```

Outputs:
- Unzipped data under `data/raw/`
- Parquet files under `data/processed/`
- A report at `data/interim/ingest_report.md`

## Pipeline Executor (NEW)

For automated execution and external agent delegation:

```bash
# Run complete pipeline with auto-commit
python pipeline_executor.py

# Run without committing (for testing)
python pipeline_executor.py --no-commit

# Force regenerate all files
python pipeline_executor.py --force-regenerate
```

See `PIPELINE_DOCUMENTATION.md` for detailed documentation.

## Manual Pipeline Execution

### Full dependencies

Install everything for modeling and the Streamlit app:

```
pip install -r requirements.txt
```

### Manual execution steps

```bash
# Step 1: Data ingestion
python src/ingest_provided_data.py

# Step 2: Feature engineering and multi-timeframe aggregation
python src/data_processing.py

# Step 3: Advanced modeling (optional, requires additional dependencies)
python src/modeling.py
```

## Notes on data ethics and platform ToS
- Only collect public, non-personal data and respect each platform's Terms of Service and rate limits.
- For any scraping, prefer official APIs. Avoid storing personal identifiers.

## Next
- Phase 2: preprocessing/feature engineering in `src/data_processing.py`
- Streamlit prototype wiring after baseline features are ready.
