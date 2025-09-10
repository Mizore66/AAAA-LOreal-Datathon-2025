#!/usr/bin/env python3
"""
Full Pipeline Usage Examples for L'Oréal Datathon 2025

This file demonstrates how to use the full pipeline with various configurations
and shows the comprehensive progress tracking features.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

def example_basic_usage():
    """
    Example 1: Basic usage with sample data
    Shows how to run the complete pipeline with default settings
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Sample Data")
    print("=" * 60)
    
    print("""
# Import the pipeline
from full_pipeline import FullPipeline

# Initialize with default configuration
pipeline = FullPipeline()

# Run with sample data (creates sample data automatically)
results = pipeline.run_full_pipeline(
    data_sources={'sample': True},
    save_intermediate=True
)

print("Pipeline completed!")
print(f"Results: {results}")
""")

def example_file_based_usage():
    """
    Example 2: Processing real parquet files
    Shows how to process your own comment and video files
    """
    print("=" * 60)
    print("EXAMPLE 2: Processing Real Parquet Files")
    print("=" * 60)
    
    print("""
from full_pipeline import FullPipeline

# Specify your data files
data_sources = {
    'comments': '/path/to/your/comments.parquet',
    'videos': '/path/to/your/videos.parquet',
    'audio': '/path/to/audio/files/'  # Optional
}

# Initialize pipeline
pipeline = FullPipeline()

# Run pipeline with your data
results = pipeline.run_full_pipeline(
    data_sources=data_sources,
    save_intermediate=True
)
""")

def example_large_dataset_usage():
    """
    Example 3: Processing 7M+ records with optimized settings
    Shows configuration for handling large datasets efficiently
    """
    print("=" * 60)
    print("EXAMPLE 3: Large Dataset Processing (7M+ Records)")
    print("=" * 60)
    
    print("""
from full_pipeline import FullPipeline

# Configuration optimized for large datasets
config = {
    'data_sources': {
        'comments': '/path/to/large_comments.parquet',
        'videos': '/path/to/large_videos.parquet'
    },
    'processing': {
        'chunk_size': 50000,  # Process in 50K row chunks
        'max_memory_gb': 4,   # Limit memory usage
        'enable_audio': False  # Skip audio for speed
    },
    'feature_processing': {
        'enable_spell_check': True,
        'enable_translation': True,
        'confidence_threshold': 0.8,
        'batch_size': 2000
    },
    'modeling': {
        'enable_semantic_validation': True,
        'enable_sentiment_analysis': True,
        'enable_decay_detection': True,
        'n_clusters': 15
    }
}

# Initialize with optimized config
pipeline = FullPipeline(config=config)

# Process large dataset with progress tracking
results = pipeline.run_full_pipeline(
    data_sources=config['data_sources'],
    save_intermediate=True
)

# Expected processing time: ~4-5 minutes for 7M records
""")

def example_command_line_usage():
    """
    Example 4: Command line usage
    Shows how to use the pipeline from command line
    """
    print("=" * 60)
    print("EXAMPLE 4: Command Line Usage")
    print("=" * 60)
    
    print("""
# Basic usage with your files
python full_pipeline.py --comments data/comments.parquet --videos data/videos.parquet

# With configuration file
python full_pipeline.py --config pipeline_config.json

# Sample data demo
python full_pipeline.py --sample

# Disable certain features for speed
python full_pipeline.py --comments data/comments.parquet --disable-audio --disable-translation

# Custom output directory
python full_pipeline.py --comments data/comments.parquet --output-dir /custom/output/path
""")

def example_progress_tracking():
    """
    Example 5: Understanding the progress tracking
    Shows what progress bars and logs you'll see
    """
    print("=" * 60)
    print("EXAMPLE 5: Progress Tracking Features")
    print("=" * 60)
    
    print("""
The pipeline provides comprehensive progress tracking at multiple levels:

📊 MAIN PIPELINE PROGRESS:
🔄 Full Pipeline Progress: 4 steps (Data Processing → Feature Processing → Modeling → Report)

📊 DATA PROCESSING PROGRESS:
   Processing chunks: 100%|██████████| 140/140 [00:04<00:00, 35.2chunk/s]
   Input: 7,000,000, Output: 875,000, Relevance: 12.5%

🔤 FEATURE PROCESSING PROGRESS:
   Processing feature files: 100%|██████████| 5/5 [00:02<00:00, 2.1files/s]
   Processing terms: 100%|██████████| 1500/1500 [00:01<00:00, 1200terms/s]
   corrections: 105, translations: 67

🤖 MODELING PROGRESS:
   Training models: 100%|██████████| 3/3 [00:08<00:00, 2.7s/model]
   Classifying posts: 100%|██████████| 875000/875000 [00:12<00:00, 72500post/s]
   Analyzing sentiment: 100%|██████████| 875000/875000 [00:15<00:00, 58333text/s]

📋 FINAL REPORT GENERATION:
   Generating report sections: 100%|██████████| 4/4 [00:01<00:00, 4sections/s]

Each progress bar shows:
- Real-time completion percentage
- Processing rate (items/second)
- Estimated time remaining
- Live statistics (success/failure counts, etc.)
""")

def example_output_structure():
    """
    Example 6: Understanding the output structure
    Shows what files and reports are generated
    """
    print("=" * 60)
    print("EXAMPLE 6: Output Structure")
    print("=" * 60)
    
    print("""
After running the pipeline, you'll find organized outputs:

data/
├── processed/
│   └── dataset/
│       ├── comments_processed_beauty.parquet
│       ├── videos_processed_fashion.parquet
│       ├── hashtags_extracted.parquet
│       └── features_extracted.parquet
├── features/
│   ├── sample_features_processed.parquet
│   └── feature_processing_details.parquet
├── models/
│   ├── semantic_clusters.json
│   ├── sentiment_results.parquet
│   └── decay_detection_results.parquet
├── interim/
│   ├── data_processing_results.json
│   ├── feature_processing_results.json
│   └── modeling_results.json
└── reports/
    └── final_report_20241201_143052.json

The final report contains:
- Pipeline execution summary
- Processing statistics
- Model performance metrics
- Recommendations for improvements
""")

def example_configuration_options():
    """
    Example 7: Configuration options
    Shows all available configuration parameters
    """
    print("=" * 60)
    print("EXAMPLE 7: Configuration Options")
    print("=" * 60)
    
    print("""
Complete configuration example:

config = {
    'data_sources': {
        'comments': 'path/to/comments.parquet',
        'videos': 'path/to/videos.parquet',
        'audio': 'path/to/audio/'
    },
    'processing': {
        'chunk_size': 50000,        # Rows per chunk
        'max_memory_gb': 2,         # Memory limit per chunk
        'enable_audio': True,       # Process audio files
        'enable_translation': True  # Translate non-English
    },
    'feature_processing': {
        'enable_spell_check': True,     # Fix spelling errors
        'enable_translation': True,     # Translate features
        'confidence_threshold': 0.7,    # Translation confidence
        'batch_size': 1000             # Terms per batch
    },
    'modeling': {
        'enable_semantic_validation': True,  # Semantic clustering
        'enable_sentiment_analysis': True,   # Sentiment scoring
        'enable_decay_detection': True,      # Trend decay analysis
        'n_clusters': 10                     # Number of semantic clusters
    },
    'output': {
        'save_intermediate': True,    # Save intermediate results
        'output_format': 'parquet',   # File format
        'compression': 'snappy'       # Compression type
    }
}
""")

def main():
    """Show all usage examples"""
    print("🎭 L'Oréal Datathon 2025 - Full Pipeline Usage Examples")
    print("🔄 Complete Data Processing → Feature Processing → Model Training")
    
    example_basic_usage()
    example_file_based_usage()
    example_large_dataset_usage()
    example_command_line_usage()
    example_progress_tracking()
    example_output_structure()
    example_configuration_options()
    
    print("\n" + "=" * 60)
    print("🚀 GETTING STARTED")
    print("=" * 60)
    print("""
To start using the pipeline:

1. Install dependencies:
   pip install -r requirements.txt

2. Run with sample data:
   python src/full_pipeline.py --sample

3. Or run with your data:
   python src/full_pipeline.py --comments your_comments.parquet --videos your_videos.parquet

4. Check results in data/reports/ folder

The pipeline handles 7,000,000+ records efficiently with comprehensive progress tracking!
""")

if __name__ == "__main__":
    main()