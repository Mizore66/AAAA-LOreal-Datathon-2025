# L'Oréal x Monash Datathon 2025: TrendSpotter

Unified documentation combining development overview and final submission highlights.

## 🎯 Purpose
A focused, multi-platform trend radar to detect emerging beauty & fashion trends across text and audio signals, delivering early insight, lifecycle state (emerging → peaking → decaying), and actionable summaries for L'Oréal brand, product, and category strategy.

## 🏆 Final Submission Highlights (Team AAAA)
| Capability | Summary |
|------------|---------|
| End-to-End Pipeline | Raw ingestion → feature aggregation → text normalization → modeling → decay analysis → interactive exploration |
| Scale Tested | 7M+ records through chunked & vectorized processing (<500MB steady RAM) |
| Performance | ~27K rows/sec ingestion & processing (local benchmarks) |
| Modeling | Category classification, sentiment sampling, semantic validation, TF‑IDF term surfacing |
| Trend Intelligence | Velocity/acceleration derived trend state + decay detection |
| Multilingual | Language detection + optional translation + spelling correction |
| Persistence | Vectorizers, classifiers & registry JSON for reproducibility |
| Visualization | Streamlit prototype with full-term search beyond top-N |

---


## �️ Tech Stack & Infrastructure

**Languages & Frameworks:**
- Python 3.10+ (core pipeline, modeling, dashboard)
- Streamlit (interactive dashboard)
- Jupyter Notebook (EDA, pipeline demonstration)

**Data Processing & Storage:**
- pandas, numpy (tabular manipulation, aggregation)
- pyarrow (Parquet file IO)
- openpyxl (Excel ingestion)
- joblib, pickle (model persistence)

**Feature Engineering & NLP:**
- scikit-learn (TF-IDF, clustering, classification)
- transformers, sentence-transformers (semantic validation, embeddings)
- spacy, autocorrect, langdetect (text cleaning, spell correction, language detection)

**Modeling & Analysis:**
- statsmodels (trend/seasonal decomposition)
- Prophet (optional, time-series forecasting)
- sklearn (classification, clustering)
- Custom scripts for velocity/acceleration/decay logic

**Visualization:**
- plotly (exploratory plots, dashboard charts)

**Infrastructure & Automation:**
- Chunked/batch processing for memory efficiency
- Modular pipeline scripts (ingestion, processing, modeling, decay analysis)
- Directory structure for reproducibility and separation of raw/interim/processed/model assets

**Deployment & Execution:**
- Local execution (Windows PowerShell, bash)
- Streamlit app for local dashboard hosting
- Jupyter notebook for stepwise demonstration and validation
- Automated pipeline runner scripts (full_pipeline.py, run_end_to_end.ipynb)

**Artifacts & Outputs:**
- Parquet files for efficient data storage
- JSON reports for profiling, performance, and modeling results
- Pickled model/vectorizer objects for reproducibility
- Markdown/HTML reports for documentation

**Optional/Advanced:**
- Cloud deployment (future roadmap: containerization, scheduled retraining)
- Real-time ingestion (streaming, append-only parquet)


```
project_root/
├── assets/                      # Images, logos, presentation assets
├── data/
│   ├── raw/                     # Unpacked original source data
│   ├── processed/               # Legacy processed artifacts (subset)
│   ├── processed_features/      # Feature text processing outputs
│   ├── features/                # Raw extracted term vocabularies
│   ├── interim/                 # Intermediate reports & JSON summaries
│   ├── feature_processing_reports/ # Logs / detail reports from text processor
│   ├── cache/                   # Cached pickled intermediate objects
│   └── reports/                 # Analysis / performance reports
├── models/                      # Saved models & analysis JSON (incl. decay + enhanced modeling)
├── final_submission/            # Competition-ready package
│   ├── app.py                   # Streamlit dashboard (final submission context)
│   ├── full_pipeline.py         # Orchestrated end-to-end runner
│   ├── run_end_to_end.ipynb     # Notebook runner (scripted execution)
│   ├── validate_submission.py   # Validation / integrity script
│   └── src/                     # Final optimized modules
│       ├── ingest_provided_data.py
│       ├── data_processing_optimized.py
│       ├── feature_text_processor.py
│       ├── modeling_optimized.py
│       ├── run_enhanced_modeling.py
│       ├── term_decay_analysis.py
│       └── test_full_pipeline.py
├── model_dev/                   # Development / experimental notebooks
├── src/                         # Base (earlier) source folder
│   └── data/                    # May include auxiliary scripts/data (lean after consolidation)
├── models/                      # (duplicate listing above kept intentionally for clarity)
├── requirements.txt             # Dependencies
├── README.md                    # Unified documentation (this file)
└── .gitignore
```

## 🚀 Quickstart (Windows PowerShell)

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

## 🧪 Manual Pipeline Execution

After that, you can run `full_pipeline.py` to properly process the data and feed it to the model

### Full dependencies

Install everything for modeling and the Streamlit app:

```
pip install -r requirements.txt
```

## 🔐 Notes on Data Ethics & Platform ToS
- Only collect public, non-personal data and respect each platform's Terms of Service and rate limits.
- For any scraping, prefer official APIs. Avoid storing personal identifiers.

## 🗺️ Development Phases (Summary)
Phase 0: Environment & roles → Phase 1: Ingestion & augmentation → Phase 2: Cleaning & feature engineering → Phase 3: Modeling & trend detection → Phase 4: Prototype dashboard → Phase 5: Packaging & submission.

---

---

## 🧱 End-to-End Architecture

The system follows a staged architecture aligning with hackathon phases and internal pipeline modules:

| Stage | Module / Script | Purpose |
|-------|-----------------|---------|
| 0 | `ingest_provided_data.py` | Unzip, catalog, profile, and convert source CSV → Parquet with reports |
| 1 | `data_processing.py` / optimized variant | Clean text, aggregate multi-timeframe features (1h–14d), create trend candidate tables |
| 2 | `feature_text_processor.py` (final_submission) | Normalize, correct, translate and de-duplicate textual feature terms |
| 3 | `modeling.py` / `modeling_optimized.py` / `run_enhanced_modeling.py` | Classification, sentiment, semantic validation, trend scoring, temporal/TF‑IDF term surfacing |
| 4 | `term_decay_analysis.py` | Detect growth vs plateau vs decay phases for surfaced trend terms |
| 5 | `app.py` (Streamlit prototype) | Interactive exploration of full term universe & trend metrics |

Enhanced artifacts (e.g. `enhanced_modeling_results.json`) are produced by the enriched modeling runner adding temporal trend windows and model persistence.

## 🏃 Execution Paths

Choose one of three ways to run everything:

1. Minimal (Ingestion Only)
	```powershell
	python src/ingest_provided_data.py
	```
2. Full Manual (core baseline)
	```powershell
	python src/ingest_provided_data.py
	python src/data_processing.py
	python src/modeling.py
	```
3. Enhanced / Final Submission (recommended)
	```powershell
	# Produces cleaned data + enhanced modeling outputs + decay analysis
	python final_submission/src/ingest_provided_data.py
	python final_submission/src/full_pipeline.py   # orchestrated processing
	python final_submission/src/run_enhanced_modeling.py
	python final_submission/src/term_decay_analysis.py
	```

Or open the consolidated notebook runner:
```powershell
jupyter notebook final_submission/run_end_to_end.ipynb
```

## 📦 Key Generated Artifacts

| Location | Example Files | Description |
|----------|---------------|-------------|
| `data/raw/` | original CSVs | Unpacked source data |
| `data/processed/` | `features_hashtags_6h.parquet` | Aggregated multi-horizon features |
| `data/interim/` | `ingest_report.md`, `performance_report.json` | Profiling & intermediate reports |
| `models/` | `*_classifier.pkl`, `*_tfidf_vectorizers.pkl` | Persisted model & vectorizer objects |
| `data/interim/enhanced_modeling_results.json` | summary JSON | High-level modeling & trend metrics |
| `data/interim/*decay*` | decay JSON/CSV | Term lifecycle state outputs |

## 📉 Trend Detection & Decay Logic (Conceptual)

1. Aggregate counts per term across rolling horizons (1h, 3h, 6h, 1d, 3d, 7d, 14d).
2. Normalize volumes and compute velocity / acceleration (first & second derivatives across windows).
3. Surface candidate emerging terms via deviation thresholds or TF‑IDF lift vs historical mean.
4. Validate semantically (embedding clustering) + classify category + sentiment annotate.
5. Determine decay: positive velocity turning negative acceleration over sustained windows ⇒ transition to "Decaying" state.

Simplified state rule:
```
if velocity > 0 and acceleration > 0: Growing
elif velocity > 0 and acceleration <= 0: Peaking
elif velocity <= 0 and acceleration < 0: Decaying
else: Stable / Low Signal
```

## 🔬 Enhanced Modeling Additions

`run_enhanced_modeling.py` augments baseline modeling with:
- Category-wise TF‑IDF trending term extraction (uni + bi-grams)
- Weekly time window segmentation for temporal term evolution
- Sentiment sampling integration per dataset
- Persisted vectorizers + classifier + semantic model metadata (registry JSON)
- Consolidated `enhanced_modeling_results.json` with dataset summaries & temporal windows

## 💻 Streamlit Prototype

Run the interactive exploration UI (ensure dependencies installed):
```powershell
cd src
streamlit run app.py
```
Features:
- Full term search (not just top-N)
- Time horizon selection & trend metrics
- Emerging vs decaying labeling (from decay analysis outputs)

## 🧭 Roadmap / Potential Extensions

- Real-time incremental ingestion (append-only parquet + streaming feature deltas)
- Adaptive anomaly thresholds via STL residual distribution modeling
- Cross-platform correlation scoring (audio ↔ hashtag ↔ keyword coherence)
- Lightweight embedding cache & semantic drift monitoring
- Integration of audio fingerprint similarity for early low-volume signals
- Unified trend score = f(volume lift, sentiment skew, cross-platform coherence, semantic cluster density)

## ✅ Ethical & Compliance Reminders

- Respect platform ToS; prefer official APIs where available
- Avoid collecting personally identifiable information (PII)
- Provide reproducibility scripts instead of raw redistributed proprietary data

## 🆘 Support / Repro Notes

Common setup issues:
- Install `pyarrow` for parquet IO
- If `prophet` installation fails on Windows, ensure build tools or pin version / skip (optional for core pipeline)
- For faster runs during judging, subset via environment flag (e.g., implement `ROW_LIMIT` in ingestion script)

---

**At a glance:** Ingestion → Multi-horizon feature engineering → Text normalization & semantic enrichment → Modeling & term surfacing → Decay lifecycle tagging → Interactive exploration.

---

## 📊 Final Submission Section

### Key Features
- Multi-platform ingestion & relevance filtering
- Scalable chunked processing & profiling
- Multi-horizon (1h–14d) aggregation & velocity metrics
- Multilingual text normalization (spell correction + detection + translation hooks)
- Semantic & sentiment augmentation for contextual ranking
- Enhanced term surfacing (TF‑IDF + temporal windows)
- Decay analysis & lifecycle state assignment

### Business Value
1. Early detection of emergent consumer concepts & product ingredient spikes
2. Sentiment-oriented prioritization for brand positioning
3. Influencer & content vector alignment for strategic activation
4. Cross-platform convergence spotting (hashtags ↔ audio ↔ keywords)
5. Strategic campaign timing via lifecycle state transitions

### Performance Metrics (Indicative)
- Processing Throughput: ~27K rows/sec (local benchmark)
- Sentiment Accuracy (sampled baseline): 75–80%
- Memory Footprint: <500MB steady during batch flow
- Retention after Relevance Filtering: ~10–15%

### Production Next Steps
- Real-time streaming ingestion & incremental feature deltas
- Cloud containerization + scheduled retraining jobs
- Model evaluation dashboard (drift & quality monitoring)
- Cross Platform Validation (Pulling data from external sources [X, Tiktok, Instagram, etc.] to check if trends are accurate)

---

## 📚 Related Final Submission Directory Snapshot
See `final_submission/` for consolidated competition-ready assets (mirrors architecture above with optimized modules & documentation).

`models` directory already has my results from running it.

`app.py` allows you to locally host a dashboard to see what terms are most trending, as well as allows to graph how the term behaves over it's lifespan and a 3 month prediction graph as well

---

## 🙌 Team & Acknowledgements
Built by Team AAAA for L'Oréal x Monash Datathon 2025. Focused on actionable, explainable, and scalable trend intelligence.


