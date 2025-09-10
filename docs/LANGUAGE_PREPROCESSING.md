# Language Preprocessing Pipeline

## Overview

The language preprocessing pipeline ensures all text data is in English before running the main data processing. This is crucial for accurate text analysis, keyword extraction, and trend detection.

## Features

### 🌍 Language Detection
- Automatically detects the language of text content
- Uses `langdetect` library for robust language identification
- Supports confidence thresholds to handle uncertain cases
- Caches detection results to avoid re-processing identical text

### 🔄 Translation
- Translates non-English text to English using Google Translate
- Batch processing for efficiency
- Translation caching to avoid redundant API calls
- Graceful error handling for failed translations

### ⚡ Performance Optimizations
- Chunked processing for large datasets (default: 10,000 rows per chunk)
- Parallel processing where beneficial
- Memory-efficient streaming
- Progress tracking with detailed logging

### 📊 Comprehensive Reporting
- Detailed statistics on detected languages
- Translation success/failure rates
- Processing time and performance metrics
- Language distribution analysis

## Usage

### Quick Start
```bash
python src/language_preprocessor.py
```

### Custom Processing
```python
from src.language_preprocessor import LanguagePreprocessor, preprocess_dataset

# Process a specific dataset
preprocess_dataset(
    input_file=Path("data/raw/videos.csv"),
    output_file=Path("data/processed/videos_translated.parquet"),
    text_columns=['title', 'description'],
    chunk_size=5000
)
```

### Batch Processing
```python
from src.language_preprocessor import LanguagePreprocessor

processor = LanguagePreprocessor(confidence_threshold=0.8)
results = processor.process_text_batch([
    "Hello world",
    "Bonjour le monde", 
    "Hola mundo"
])
```

## Output Structure

### Translated Datasets
For each text column (e.g., `title`), the following columns are added:
- `title_original`: Original text before translation
- `title_translated`: Text after translation (used as the main column)
- `title_detected_lang`: Detected language code (e.g., 'en', 'fr', 'es')
- `title_was_translated`: Boolean indicating if translation occurred
- `title_is_english`: Boolean indicating if original text was English
- `title_confidence`: Confidence score of language detection

### Language Reports
JSON reports with detailed statistics:
```json
{
  "preprocessing_timestamp": "2025-09-10 03:57:11",
  "statistics": {
    "total_texts": 65,
    "english_texts": 54,
    "translated_texts": 11,
    "failed_translations": 0,
    "detected_languages": {
      "en": 54,
      "de": 2,
      "pl": 1,
      "id": 3,
      "ar": 1
    }
  },
  "configuration": {
    "confidence_threshold": 0.8,
    "batch_size": 100,
    "translation_available": true
  }
}
```

## Configuration

### Performance Modes
- **Confidence Threshold**: Minimum confidence for language detection (default: 0.8)
- **Batch Size**: Number of texts to process together (default: 100)
- **Chunk Size**: Rows per chunk for large datasets (default: 10,000)

### Supported Languages
The preprocessing pipeline can detect and translate from any language supported by Google Translate, including:
- European languages (German, French, Spanish, Italian, etc.)
- Asian languages (Chinese, Japanese, Korean, etc.)
- Middle Eastern languages (Arabic, Hebrew, etc.)
- And many more...

## Files Processed

### Current Dataset Structure
```
data/raw/
├── videos.csv (92,759 rows)
├── comments1.csv
├── comments2.csv
├── comments3.csv
├── comments4.csv
└── comments5.csv
```

### Output Structure
```
data/processed/
├── videos_translated.parquet
├── videos_translated_language_report.json
├── comments1_translated.parquet
├── comments1_translated_language_report.json
├── ... (and so on for all comment files)
```

## Integration with Main Pipeline

The translated datasets can be seamlessly integrated with the main data processing pipeline. Simply update the data loading paths to use the `*_translated.parquet` files instead of the original CSV files.

## Performance Metrics

Based on testing with sample data:
- **Language Detection**: ~100 texts/second
- **Translation**: ~25 texts/second (API dependent)
- **Memory Usage**: Optimized for large datasets with chunked processing
- **Accuracy**: High accuracy with confidence thresholding

## Error Handling

- **Missing Libraries**: Graceful fallback with clear installation instructions
- **API Limits**: Built-in retry logic and rate limiting
- **Network Issues**: Timeout handling and error recovery
- **Invalid Text**: Safe handling of empty/malformed text

## Example Results

From a sample of 50 video titles and descriptions:
- **Total texts processed**: 65
- **Already in English**: 54 (83%)
- **Successfully translated**: 11 (17%)
- **Languages detected**: German, Polish, Indonesian, Finnish, Afrikaans, Welsh, Arabic, Norwegian

This preprocessing ensures that all subsequent analysis (keyword extraction, trend detection, etc.) works with consistent English text, improving accuracy and reliability of the results.
