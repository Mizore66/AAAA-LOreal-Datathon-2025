# Modular Data Processing Pipeline

## 🎯 Overview

This modular version replaces the original 1000+ line `data_processing_optimized.py` with a clean, maintainable architecture. The pipeline is broken down into 6 focused modules plus a clean 200-line driver file.

## 📁 Module Structure

```
src/
├── data_processing_modular.py     # Main driver (200 lines)
├── test_modular_imports.py        # Import verification
└── modules/
    ├── __init__.py                # Module initialization
    ├── config.py                  # Configuration & constants
    ├── text_processing.py         # Text cleaning & feature extraction
    ├── data_loading.py            # Dataset loading & preprocessing
    ├── aggregation.py             # Time-series aggregation
    ├── parallel_processing.py     # Concurrent processing utilities
    └── reporting.py               # Enhanced report generation
```

## 🚀 Key Improvements

### ✅ Modular Architecture
- **6 focused modules** instead of one monolithic file
- **Single responsibility** for each module
- **Clean separation** of concerns
- **Easy to maintain** and extend

### ✅ Enhanced Progress Tracking
- **tqdm progress bars** for all timeframe processing
- **Emoji logging** for better visual feedback
- **Detailed progress** for each phase
- **Real-time status** updates

### ✅ Intelligent File Management
- **Skip existing files** to avoid reprocessing
- **Cache loaded data** for efficiency
- **Smart file detection** for parquet files
- **Memory optimization** through caching

### ✅ Enhanced Reporting
- **JSON reports** matching your provided format
- **Comprehensive analysis** with detailed metrics
- **Category breakdowns** and top performer analysis
- **Structured data** for easy programmatic access
- **Overlap analysis** between timeframes and sources

### 7. **Comprehensive Reporting**
- Multi-timeframe analysis
- Performance monitoring
- Clean markdown output

## Usage

### Run the Complete Pipeline
```bash
cd src
python data_processing_modular.py
```

### Import Individual Modules
```python
from modules import config, text_processing, aggregation
from modules.data_loading import load_samples_optimized
from modules.parallel_processing import parallel_process_sources
```

### Customize Configuration
```python
from modules.config import CONFIG, PERFORMANCE_MODE

# Change performance mode
# Options: "OPTIMIZED", "BALANCED", "THOROUGH"
CONFIG = PERF_CONFIGS["OPTIMIZED"]
```

## Configuration Options

### Performance Modes

- **OPTIMIZED**: Fast processing with aggressive sampling
- **BALANCED**: Good performance with reasonable accuracy (default)
- **THOROUGH**: Complete processing with full datasets

### Key Settings

- `max_features_emerging`: Limit emerging terms for performance
- `sample_rows_per_source`: Sample large datasets
- `enable_parallel`: Use concurrent processing
- `enable_chunking`: Split large datasets into chunks
- `chunk_size`: Size of processing chunks

## Output Files

The pipeline generates structured JSON reports matching your provided format:

```
data/
├── processed/
│   ├── features_hashtags_*.parquet      # Hashtag features by timeframe
│   ├── features_keywords_*.parquet      # Keyword features by timeframe
│   ├── features_emerging_terms_*.parquet # Emerging trends by timeframe
│   ├── features_audio_*.parquet          # Audio features (6h only)
│   └── dataset_description.parquet       # Dataset metadata
└── interim/
    ├── phase2_enhanced_features_comprehensive.json    # JSON Phase 2 analysis
    ├── phase3_emerging_trends_comprehensive.json      # JSON emerging trends report
    └── performance_report.json                        # JSON performance metrics
```

## Benefits vs Original

1. **🔧 Maintainability**: Easy to modify individual components
2. **🧪 Testability**: Each module can be tested independently  
3. **📈 Scalability**: Clear separation allows for easy extensions
4. **🔍 Readability**: Much smaller, focused files
5. **🔄 Reusability**: Modules can be used in other projects
6. **🐛 Debugging**: Easier to isolate and fix issues
7. **👥 Collaboration**: Multiple developers can work on different modules

## Migration from Original

The modular pipeline is fully compatible with the original:
- Same input/output formats
- Same configuration options
- Same performance characteristics
- All optimizations preserved

Simply replace calls to `data_processing_optimized.py` with `data_processing_modular.py`.
