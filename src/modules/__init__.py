"""
L'Oréal Datathon 2025 Data Processing Modules

This package contains modular components for the data processing pipeline:

- config: Configuration settings and constants
- text_processing: Text cleaning and feature extraction utilities
- data_loading: Dataset loading and preprocessing functions
- aggregation: Hashtag, keyword, and emerging terms aggregation
- parallel_processing: Concurrent processing utilities
- reporting: Report generation functions
"""

__version__ = "1.0.0"
__author__ = "L'Oréal Datathon 2025 Team"

# Import key functions for easy access
from .config import CONFIG, PERFORMANCE_MODE, TIMEFRAME_LABELS
from .text_processing import clean_text_optimized, categorize_feature, is_trend_relevant
from .data_loading import load_samples_optimized
from .aggregation import (
    aggregate_hashtags_optimized, 
    aggregate_keywords_optimized, 
    aggregate_emerging_terms_optimized
)
from .parallel_processing import parallel_process_sources, process_dataset_with_chunking
from .reporting import write_enhanced_phase2_report, write_phase3_emerging_trends_report

__all__ = [
    'CONFIG', 'PERFORMANCE_MODE', 'TIMEFRAME_LABELS',
    'clean_text_optimized', 'categorize_feature', 'is_trend_relevant',
    'load_samples_optimized',
    'aggregate_hashtags_optimized', 'aggregate_keywords_optimized', 'aggregate_emerging_terms_optimized',
    'parallel_process_sources', 'process_dataset_with_chunking',
    'write_enhanced_phase2_report', 'write_phase3_emerging_trends_report'
]
