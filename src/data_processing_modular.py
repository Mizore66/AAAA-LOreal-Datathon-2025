#!/usr/bin/env python3
"""
Modular Data Processing Pipeline for L'Oréal Datathon 2025
Clean driver file that orchestrates all processing modules.

Features:
1. Clean separation of concerns
2. Modular design for easy maintenance
3. Parallel and chunked processing
4. Comprehensive reporting
5. Performance monitoring
"""

import time
import logging
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm

# Import our modular components
from modules.config import (
    INTERIM_DIR, PROC_DIR, TIMEFRAME_LABELS, 
    PERFORMANCE_MODE, CONFIG
)
from modules.data_loading import load_samples_optimized
from modules.aggregation import (
    aggregate_hashtags_optimized, aggregate_keywords_optimized,
    aggregate_emerging_terms_optimized, add_rolling_stats_optimized
)
from modules.parallel_processing import parallel_process_sources, process_dataset_with_chunking
from modules.reporting import (
    write_enhanced_phase2_report, write_phase3_emerging_trends_report,
    generate_performance_report
)

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def process_baseline_6h_aggregations(comment_sources, video_sources):
    """Process baseline 6h aggregations for all sources."""
    logger.info("Phase 2: Running dataset-aware baseline 6h aggregations with chunked processing")
    
    def process_hashtags_6h_chunked(name: str, df: pd.DataFrame) -> pd.DataFrame:
        return process_dataset_with_chunking(name, df, 'hashtags', '6h')
    
    def process_keywords_6h_chunked(name: str, df: pd.DataFrame) -> pd.DataFrame:
        return process_dataset_with_chunking(name, df, 'keywords', '6h')
    
    # Check if baseline files already exist
    hashtag_path = PROC_DIR / 'features_hashtags_6h.parquet'
    keyword_path = PROC_DIR / 'features_keywords_6h.parquet'
    
    if hashtag_path.exists() and keyword_path.exists():
        logger.info("✅ Baseline 6h aggregation files already exist, loading from cache...")
        try:
            ts_hashtags = pd.read_parquet(hashtag_path)
            ts_keywords = pd.read_parquet(keyword_path)
            ts_audio = pd.DataFrame(columns=['bin', 'feature', 'count', 'rolling_mean_24h', 'delta_vs_mean', 'category', 'source_type'])
            logger.info(f"Loaded cached hashtags: {len(ts_hashtags):,} records")
            logger.info(f"Loaded cached keywords: {len(ts_keywords):,} records")
            return ts_hashtags, ts_keywords, ts_audio
        except Exception as e:
            logger.warning(f"Failed to load cached files: {e}. Regenerating...")
    
    # Process baseline aggregations for comments and videos separately
    all_sources = comment_sources + video_sources
    
    logger.info("🔄 Processing hashtags for 6h baseline...")
    hashtag_frames = []
    for name, df in tqdm(all_sources, desc="Hashtags 6h", unit="source"):
        result = process_hashtags_6h_chunked(name, df)
        if result is not None:
            hashtag_frames.append(result)
    
    logger.info("🔄 Processing keywords for 6h baseline...")
    keyword_frames = []
    for name, df in tqdm(all_sources, desc="Keywords 6h", unit="source"):
        result = process_keywords_6h_chunked(name, df)
        if result is not None:
            keyword_frames.append(result)
    
    # Combine and process results
    ts_hashtags = pd.concat(hashtag_frames, ignore_index=True) if hashtag_frames else pd.DataFrame()
    ts_keywords = pd.concat(keyword_frames, ignore_index=True) if keyword_frames else pd.DataFrame()
    
    # Add rolling statistics
    if not ts_hashtags.empty:
        logger.info("📊 Adding rolling statistics to hashtags...")
        ts_hashtags = add_rolling_stats_optimized(ts_hashtags, '6h')
    
    if not ts_keywords.empty:
        logger.info("📊 Adding rolling statistics to keywords...")
        ts_keywords = add_rolling_stats_optimized(ts_keywords, '6h')
    
    # Empty audio for this dataset type
    ts_audio = pd.DataFrame(columns=['bin', 'feature', 'count', 'rolling_mean_24h', 'delta_vs_mean', 'category', 'source_type'])
    
    # Save baseline results
    if not ts_hashtags.empty:
        ts_hashtags.to_parquet(hashtag_path, index=False)
        logger.info(f"💾 Saved baseline hashtags: {hashtag_path} ({len(ts_hashtags):,} records)")
    
    if not ts_keywords.empty:
        ts_keywords.to_parquet(keyword_path, index=False)
        logger.info(f"💾 Saved baseline keywords: {keyword_path} ({len(ts_keywords):,} records)")
    
    return ts_hashtags, ts_keywords, ts_audio

def process_multi_timeframe_aggregations(comment_sources, video_sources):
    """Process aggregations for all timeframes."""
    logger.info("Phase 2: Running multi-timeframe dataset-aware aggregations with chunked processing")
    
    all_sources = comment_sources + video_sources
    
    for label in tqdm(TIMEFRAME_LABELS, desc="🕒 Processing timeframes", unit="timeframe"):
        logger.info(f"📅 Processing timeframe: {label}")
        
        # Check if files already exist
        hashtag_path = PROC_DIR / f'features_hashtags_{label}.parquet'
        keyword_path = PROC_DIR / f'features_keywords_{label}.parquet'
        audio_path = PROC_DIR / f'features_audio_{label}.parquet'
        
        skip_hashtags = hashtag_path.exists()
        skip_keywords = keyword_path.exists()
        skip_audio = audio_path.exists()
        
        if skip_hashtags and skip_keywords and skip_audio:
            logger.info(f"✅ All {label} files already exist, skipping...")
            continue
        
        def process_hashtags_tf(name: str, df: pd.DataFrame) -> pd.DataFrame:
            return process_dataset_with_chunking(name, df, 'hashtags', label)
        
        def process_keywords_tf(name: str, df: pd.DataFrame) -> pd.DataFrame:
            return process_dataset_with_chunking(name, df, 'keywords', label)
        
        # Process hashtags for this timeframe
        if not skip_hashtags:
            logger.info(f"🔄 Processing hashtags for {label}...")
            hashtag_frames = []
            for name, df in tqdm(all_sources, desc=f"Hashtags {label}", unit="source", leave=False):
                result = process_hashtags_tf(name, df)
                if result is not None:
                    hashtag_frames.append(result)
            
            if hashtag_frames:
                combined_hashtags = pd.concat(hashtag_frames, ignore_index=True)
                combined_hashtags['timeframe'] = label
                combined_hashtags.to_parquet(hashtag_path, index=False)
                logger.info(f"💾 Saved {label} hashtags: {len(combined_hashtags):,} records")
        else:
            logger.info(f"⏭️  Skipping {label} hashtags (already exists)")
        
        # Process keywords for this timeframe
        if not skip_keywords:
            logger.info(f"🔄 Processing keywords for {label}...")
            keyword_frames = []
            for name, df in tqdm(all_sources, desc=f"Keywords {label}", unit="source", leave=False):
                result = process_keywords_tf(name, df)
                if result is not None:
                    keyword_frames.append(result)
            
            if keyword_frames:
                combined_keywords = pd.concat(keyword_frames, ignore_index=True)
                combined_keywords['timeframe'] = label
                combined_keywords.to_parquet(keyword_path, index=False)
                logger.info(f"💾 Saved {label} keywords: {len(combined_keywords):,} records")
        else:
            logger.info(f"⏭️  Skipping {label} keywords (already exists)")
        
        # Create empty audio file for consistency
        if not skip_audio:
            empty_audio = pd.DataFrame(columns=['bin', 'feature', 'count', 'category', 'timeframe'])
            empty_audio['timeframe'] = label
            empty_audio.to_parquet(audio_path, index=False)
            logger.info(f"📁 Created empty {label} audio file")
        else:
            logger.info(f"⏭️  Skipping {label} audio (already exists)")

def load_all_timeframes(feature_type: str) -> pd.DataFrame:
    """Load and combine all timeframe data for a feature type."""
    frames = []
    for label in TIMEFRAME_LABELS:
        file_path = PROC_DIR / f'features_{feature_type}_{label}.parquet'
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    df['timeframe'] = label
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
    
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

def process_emerging_trends(comment_sources, video_sources):
    """Process emerging trends for all timeframes."""
    logger.info("Phase 3: Running multi-timeframe dataset-aware emerging trends detection with chunked processing")
    
    all_sources = comment_sources + video_sources
    emerging_frames = []
    
    # Check if combined file already exists
    combined_path = PROC_DIR / 'features_emerging_terms_all.parquet'
    if combined_path.exists():
        logger.info("✅ Combined emerging trends file already exists, loading from cache...")
        try:
            all_emerging = pd.read_parquet(combined_path)
            logger.info(f"Loaded cached emerging trends: {len(all_emerging):,} records")
            return all_emerging
        except Exception as e:
            logger.warning(f"Failed to load cached file: {e}. Regenerating...")
    
    for label in tqdm(TIMEFRAME_LABELS, desc="🔍 Emerging trends", unit="timeframe"):
        logger.info(f"🔍 Processing emerging trends for timeframe: {label}")
        
        # Check if individual file exists
        output_path = PROC_DIR / f'features_emerging_terms_{label}.parquet'
        if output_path.exists():
            logger.info(f"✅ {label} emerging trends already exist, loading...")
            try:
                emerging_df = pd.read_parquet(output_path)
                emerging_df['timeframe'] = label
                emerging_frames.append(emerging_df)
                logger.info(f"📂 Loaded {label} emerging trends: {len(emerging_df):,} records")
                continue
            except Exception as e:
                logger.warning(f"Failed to load {output_path}: {e}. Regenerating...")
        
        # Use the aggregate_emerging_terms_optimized function
        logger.info(f"🔄 Computing new emerging trends for {label}...")
        emerging_df = aggregate_emerging_terms_optimized(all_sources, label)
        
        if not emerging_df.empty:
            emerging_df['timeframe'] = label
            emerging_frames.append(emerging_df)
            
            # Save individual timeframe results
            emerging_df.to_parquet(output_path, index=False)
            logger.info(f"💾 Saved {label} emerging trends: {len(emerging_df):,} records")
        else:
            logger.warning(f"⚠️  No emerging trends found for {label}")
    
    # Combine all emerging trends
    if emerging_frames:
        all_emerging = pd.concat(emerging_frames, ignore_index=True)
        all_emerging.to_parquet(combined_path, index=False)
        logger.info(f"💾 Saved combined emerging trends: {len(all_emerging):,} records")
        return all_emerging
    else:
        logger.warning("⚠️  No emerging trends detected across any timeframe")
        return pd.DataFrame()

def main():
    """Main optimized data processing pipeline with modular design."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"🚀 STARTING MODULAR DATA PROCESSING PIPELINE")
    logger.info(f"📊 Performance Mode: {PERFORMANCE_MODE}")
    logger.info(f"⚙️  Configuration: {CONFIG}")
    logger.info("=" * 80)
    
    # Load and preprocess data
    logger.info("📂 Phase 1: Loading and preprocessing datasets...")
    samples = load_samples_optimized()
    if not samples:
        logger.error("❌ No datasets loaded. Exiting.")
        return
    
    # Separate comment and video sources
    comment_sources = []
    video_sources = []
    
    for name, df in samples.items():
        if df.empty:
            continue
        
        dataset_type = df['dataset_type'].iloc[0] if 'dataset_type' in df.columns else 'unknown'
        if dataset_type == 'comments':
            comment_sources.append((name, df))
        elif dataset_type == 'videos':
            video_sources.append((name, df))
        else:
            # Default to comments if unknown
            comment_sources.append((name, df))
    
    total_comment_rows = sum(len(df) for _, df in comment_sources)
    total_video_rows = sum(len(df) for _, df in video_sources)
    
    logger.info(f"✅ Data loaded successfully!")
    logger.info(f"   📝 Comment sources: {len(comment_sources)} files, {total_comment_rows:,} total rows")
    logger.info(f"   🎥 Video sources: {len(video_sources)} files, {total_video_rows:,} total rows")
    logger.info("")
    
    # Phase 2: Process baseline 6h aggregations
    logger.info("📊 Phase 2a: Processing baseline 6h aggregations...")
    ts_hashtags, ts_keywords, ts_audio = process_baseline_6h_aggregations(comment_sources, video_sources)
    
    logger.info(f"✅ Baseline 6h processing completed!")
    logger.info(f"   #️⃣ Hashtags: {len(ts_hashtags):,} records")
    logger.info(f"   🔑 Keywords: {len(ts_keywords):,} records")
    logger.info("")
    
    # Phase 2: Process multi-timeframe aggregations
    logger.info("📊 Phase 2b: Processing multi-timeframe aggregations...")
    process_multi_timeframe_aggregations(comment_sources, video_sources)
    
    logger.info("✅ Multi-timeframe processing completed!")
    logger.info("")
    
    # Generate Phase 2 report
    logger.info("📝 Phase 2c: Generating enhanced Phase 2 report...")
    all_timeframe_data = {
        'hashtags': load_all_timeframes('hashtags'),
        'keywords': load_all_timeframes('keywords'),
        'audio': load_all_timeframes('audio')
    }
    
    total_hashtag_features = all_timeframe_data['hashtags']['feature'].nunique() if not all_timeframe_data['hashtags'].empty else 0
    total_keyword_features = all_timeframe_data['keywords']['feature'].nunique() if not all_timeframe_data['keywords'].empty else 0
    
    write_enhanced_phase2_report(all_timeframe_data)
    
    logger.info(f"✅ Phase 2 report generated!")
    logger.info(f"   #️⃣ Total hashtag features: {total_hashtag_features:,}")
    logger.info(f"   🔑 Total keyword features: {total_keyword_features:,}")
    logger.info("")
    
    # Phase 3: Process emerging trends
    logger.info("🔍 Phase 3a: Processing emerging trends detection...")
    all_emerging = process_emerging_trends(comment_sources, video_sources)
    
    emerging_count = len(all_emerging)
    unique_emerging = all_emerging['feature'].nunique() if not all_emerging.empty else 0
    
    logger.info(f"✅ Emerging trends processing completed!")
    logger.info(f"   🚀 Total emerging records: {emerging_count:,}")
    logger.info(f"   🎯 Unique emerging features: {unique_emerging:,}")
    logger.info("")
    
    # Generate Phase 3 report
    logger.info("📝 Phase 3b: Generating emerging trends report...")
    try:
        write_phase3_emerging_trends_report(all_emerging, ts_hashtags, ts_keywords)
        logger.info("✅ Phase 3 report generated successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to write Phase 3 report: {e}")
    
    logger.info("")
    
    # Performance summary
    end_time = time.time()
    total_files_generated = len(list(PROC_DIR.glob('features_*.parquet')))
    
    logger.info("📈 Generating performance summary...")
    generate_performance_report(
        start_time, end_time,
        len(comment_sources), len(video_sources),
        total_files_generated
    )
    
    duration = end_time - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"⏱️  Total execution time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"📊 Performance mode: {PERFORMANCE_MODE}")
    logger.info(f"📁 Files generated: {total_files_generated}")
    logger.info(f"🔧 Chunked processing: {'Enabled' if CONFIG.get('enable_chunking', True) else 'Disabled'}")
    logger.info(f"⚡ Parallel processing: {'Enabled' if CONFIG['enable_parallel'] else 'Disabled'}")
    logger.info("")
    logger.info("📂 Output locations:")
    logger.info(f"   📄 Reports: {INTERIM_DIR}")
    logger.info(f"   📊 Features: {PROC_DIR}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
