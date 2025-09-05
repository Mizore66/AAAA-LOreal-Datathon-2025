# L'Oréal x Monash Datathon 2025: TrendSpotter

A focused, multi-platform trend radar to detect emerging beauty trends across text and audio signals, with a lightweight Streamlit prototype.

## 🚀 Phase 3: Enhanced Trend Detection

**NEW:** Keyword-independent trend detection system that discovers emerging trends without relying on predefined keywords.

### Key Features
- **Statistical Anomaly Detection**: Identifies unusual spikes in term frequency
- **N-gram Analysis**: Discovers emerging multi-word phrases and expressions
- **Growth Rate Analysis**: Tracks rapidly growing trends with velocity metrics
- **Trend Clustering**: Groups related emerging concepts automatically
- **Caching System**: Optimizes performance for repeated analysis runs

### Innovation Over Traditional Approaches
Traditional keyword-based systems miss emerging trends that don't match predefined terms. Our enhanced system:
- Analyzes ALL terms in the data, not just keywords
- Detects novel beauty concepts and terminology as they emerge
- Uses statistical methods to identify significant trend changes
- Clusters related trends to show broader patterns

## Project structure

```
project_root/
├── assets/               # Images, logos, etc.
├── data/
│   ├── raw/              # Original data (unzipped here)
│   ├── processed/        # Cleaned and preprocessed data (e.g., parquet)
│   ├── interim/          # Intermediate outputs & reports
│   └── cache/            # Cached analysis results
├── models/               # Saved models
├── notebooks/            # EDA & experiments
├── src/                  # Source code
│   ├── ingest_provided_data.py
│   ├── data_processing.py    # Enhanced with Phase 3 features
│   ├── modeling.py
│   └── visualization.py
├── steps.md              # Phase implementation roadmap
├── demo_phase3.py        # Phase 3 capabilities demo
├── test_phase3.py        # Testing utilities
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

## Phase 3: Enhanced Trend Detection

5) Install all dependencies for advanced analysis:

```
pip install -r requirements.txt
```

6) Run enhanced data processing with keyword-independent trend detection:

```
python .\src\data_processing.py
```

Outputs:
- Traditional features: hashtags, keywords, audio
- **NEW:** Emerging terms analysis (`features_emerging_terms_6h.parquet`)
- **NEW:** Statistical anomalies detection
- **NEW:** Trend clusters (`trend_clusters.parquet`)
- Comprehensive reports in `data/interim/`

7) View Phase 3 capabilities demo:

```
python demo_phase3.py
```

## Key Innovation: Answering "Can we detect trends without keywords?"

**Yes!** Our Phase 3 implementation addresses this challenge through:

### 1. Statistical Anomaly Detection
- Identifies terms with unusual frequency spikes using z-score analysis
- Detects emerging trends even if they don't match existing keywords
- Configurable sensitivity thresholds

### 2. N-gram Analysis  
- Extracts bigrams and trigrams to find emerging phrases
- Discovers compound beauty terms like "peptide slug combo"
- Captures evolving beauty language and expressions

### 3. Growth Rate & Velocity Tracking
- Measures how quickly terms are gaining traction
- Identifies trends in their early stages
- Provides velocity metrics for trend monitoring

### 4. Semantic Clustering
- Groups related emerging terms automatically
- Shows broader thematic trends (e.g., barrier repair, slugging methods)
- Reduces noise in trend detection

## Results Example

Recent analysis found emerging trends like:
- "peptide slug combo" (growth rate: 2.2x)
- "barrier repair ceramide" (velocity: 1.10)
- "glass slug method" (clustered with related terms)

These would be missed by traditional keyword-only approaches!

## Full dependencies

Install everything for modeling and the Streamlit app:

```
pip install -r requirements.txt
```

## Notes on data ethics and platform ToS
- Only collect public, non-personal data and respect each platform's Terms of Service and rate limits.
- For any scraping, prefer official APIs. Avoid storing personal identifiers.

## Next Steps
- Phase 4: Streamlit dashboard with real-time trend visualization
- Semantic similarity analysis using embeddings
- Integration with real social media data feeds
- Advanced time series forecasting for trend prediction
