# Pipeline Executor Documentation

The `pipeline_executor.py` script provides external agent delegation capabilities for the L'Oréal Datathon 2025 data processing pipeline.

## Features

- **Automated Pipeline Execution**: Runs the complete data processing pipeline (ingestion → processing → modeling)
- **Existence Checking**: Skips aggregation if parquet files already exist with data
- **Multi-timeframe Aggregation**: Supports 1h, 3h, 6h, 1d, 3d, 7d, 14d, 1m, 3m, 6m timeframes
- **Git Integration**: Automatically commits pipeline updates and execution summaries
- **Error Handling**: Robust error handling with detailed logging
- **Execution Reporting**: Generates comprehensive execution reports

## Usage

### Basic Usage

```bash
# Run pipeline with auto-commit enabled
python pipeline_executor.py

# Run pipeline without committing changes
python pipeline_executor.py --no-commit

# Force regenerate all files (skip existence checking)
python pipeline_executor.py --force-regenerate

# Use custom commit message
python pipeline_executor.py --commit-message "Updated trend detection algorithms"
```

### Command Line Options

- `--no-commit`: Disable automatic git commit of generated files
- `--force-regenerate`: Force regeneration of existing files (skip existence checking)
- `--commit-message TEXT`: Custom commit message for generated files

### Prerequisites

The pipeline executor checks for the following prerequisites before execution:

1. **Git Repository**: Must be run within a git repository
2. **Required Files**: 
   - `src/ingest_provided_data.py`
   - `src/data_processing.py`
   - `src/modeling.py`
3. **Python Dependencies**: Core packages (pandas, numpy, pyarrow)
4. **Directory Structure**: Automatically creates required data directories

## Pipeline Stages

### 1. Data Ingestion
- Extracts and catalogs raw data files
- Converts CSV files to Parquet format
- Handles missing or corrupted data gracefully

### 2. Data Processing & Feature Engineering
- **Multi-timeframe Aggregation**: Processes data at various time granularities
- **Hashtag Analysis**: Extracts and aggregates social media hashtags
- **Keyword Detection**: Identifies beauty-related keywords and trends
- **Audio Processing**: Analyzes audio-related features
- **Emerging Trends**: Detects keyword-independent emerging trends
- **Statistical Anomalies**: Identifies unusual spikes in feature frequency
- **Trend Clustering**: Groups related trending concepts

### 3. Advanced Modeling (Optional)
- Prophet-based time series forecasting
- Anomaly detection using STL decomposition
- Category classification models
- Cross-platform trend validation

## Existence Checking

The pipeline implements intelligent existence checking to avoid redundant processing:

```python
def check_parquet_exists(name: str) -> bool:
    """Check if parquet file already exists with data."""
    out_path = PROC_DIR / name
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            if len(df) > 0:
                print(f"Skipping {name} - file already exists with {len(df)} rows")
                return True
        except Exception as e:
            print(f"[WARN] Failed to read existing {name}: {e}")
    return False
```

This ensures that:
- Files with existing data are not regenerated
- Corrupted files are automatically regenerated
- Processing time is minimized for subsequent runs

## Generated Files

The pipeline generates files in the following structure:

```
data/
├── processed/
│   ├── features_hashtags_{timeframe}.parquet    # Hashtag aggregations
│   ├── features_keywords_{timeframe}.parquet    # Keyword aggregations
│   ├── features_audio_{timeframe}.parquet       # Audio feature aggregations
│   ├── features_emerging_terms_6h.parquet       # Emerging trends
│   ├── features_statistical_anomalies.parquet   # Statistical anomalies
│   └── trend_clusters.parquet                   # Trend clusters
├── interim/
│   ├── phase2_features_report.md                # Feature engineering report
│   ├── phase3_advanced_trends_report.md         # Advanced trends report
│   └── pipeline_execution_{timestamp}.json      # Execution logs
└── models/
    └── *.pkl                                     # Trained models (if available)
```

## Git Integration

The pipeline handles git operations intelligently:

1. **Respects .gitignore**: Data files in `.gitignore` are not committed
2. **Commits Pipeline Updates**: Code changes and configuration updates are committed
3. **Execution Summary**: Creates `execution_summary.json` with pipeline results
4. **Custom Messages**: Supports custom commit messages

Example execution summary:
```json
{
  "execution_timestamp": "2025-09-06T08:02:51.479458",
  "pipeline_version": "1.0",
  "generated_files_count": 28,
  "generated_files_summary": {
    "trend_analysis": 1,
    "feature_aggregation": 23,
    "other": 1,
    "reports": 3
  },
  "feature_timeframes": ["1h", "3h", "6h", "1d", "3d", "7d", "14d", "1m", "3m", "6m"],
  "reports_generated": ["phase2_features_report.md", "phase3_advanced_trends_report.md"]
}
```

## External Agent Integration

The pipeline executor is designed for external agent delegation:

1. **Self-contained**: Single script with all dependencies
2. **Exit Codes**: Returns 0 for success, 1 for failure
3. **Comprehensive Logging**: Detailed logs with timestamps
4. **Error Recovery**: Continues processing even if individual stages fail
5. **Timeout Protection**: Prevents hanging on long-running operations

## Example Integration

```bash
#!/bin/bash
# External agent script

cd /path/to/project
python pipeline_executor.py --commit-message "Automated pipeline run $(date)"

if [ $? -eq 0 ]; then
    echo "Pipeline executed successfully"
    # Additional post-processing steps
else
    echo "Pipeline execution failed"
    # Error handling and notification
fi
```

## Monitoring and Maintenance

The pipeline provides several monitoring capabilities:

1. **Execution Logs**: Detailed logs in `pipeline_execution.log`
2. **Performance Metrics**: Execution time and file counts
3. **Error Tracking**: Comprehensive error reporting
4. **Git History**: Automatic tracking of pipeline changes

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages via `pip install -r requirements.txt`
2. **Git Issues**: Ensure you're in a git repository with proper permissions
3. **Data Directory**: Pipeline automatically creates required directories
4. **Memory Issues**: Large datasets may require increased memory allocation

### Debug Mode

For debugging, you can run individual pipeline components:

```bash
# Test data processing only
python src/data_processing.py

# Test with sample data
python test_pipeline.py
```

## Performance Optimization

The pipeline includes several optimizations:

1. **Existence Checking**: Skips unnecessary recomputation
2. **Chunked Processing**: Handles large datasets efficiently
3. **Parallel Processing**: Utilizes multiple cores where possible
4. **Caching**: Implements intelligent caching for expensive operations
5. **Memory Management**: Efficient memory usage for large datasets

## Future Enhancements

Planned improvements include:

1. **Real-time Processing**: Support for streaming data
2. **Distributed Computing**: Support for cluster-based processing
3. **Advanced Monitoring**: Integration with monitoring systems
4. **API Integration**: REST API for external system integration
5. **Model Versioning**: Advanced model versioning and rollback capabilities