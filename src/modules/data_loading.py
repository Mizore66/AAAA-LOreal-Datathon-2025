#!/usr/bin/env python3
"""
Data loading and preprocessing module for L'Oréal Datathon 2025
Handles dataset loading, preprocessing, and standardization.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from .config import PROC_DIR, CONFIG
from .text_processing import clean_text_optimized, extract_hashtags_optimized

logger = logging.getLogger(__name__)

def get_dataset_size_mb(df: pd.DataFrame) -> float:
    """Calculate dataset size in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def should_use_streaming(df: pd.DataFrame) -> bool:
    """Determine if streaming should be used based on dataset size."""
    size_mb = get_dataset_size_mb(df)
    return size_mb > CONFIG["streaming_threshold_mb"]

def sample_large_dataset(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Sample large datasets intelligently."""
    if max_rows is None or len(df) <= max_rows:
        return df
    
    logger.info(f"Sampling {max_rows:,} rows from {len(df):,} total rows")
    # Stratified sampling by time if possible
    if 'ts' in df.columns or any('time' in col.lower() for col in df.columns):
        return df.sample(n=max_rows, random_state=42).sort_index()
    else:
        return df.sample(n=max_rows, random_state=42)

def preprocess_comments_dataset(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Preprocess YouTube comments dataset based on schema.
    
    Expected columns: commentId, parentCommentId, channelId, videoId, authorId,
                     textOriginal, likeCount, publishedAt, updatedAt
    """
    if df.empty:
        return None
    
    logger.info(f"Preprocessing comments dataset: {len(df):,} rows")
    
    # Check for required columns
    required_cols = ['textOriginal', 'publishedAt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns for comments: {missing_cols}")
        return None
    
    # Create standardized columns for text processing
    processed_df = df.copy()
    
    # Standardize timestamp column
    processed_df['timestamp'] = pd.to_datetime(processed_df['publishedAt'], errors='coerce')
    
    # Standardize text column
    processed_df['text'] = processed_df['textOriginal'].astype(str)
    
    # Add engagement metrics if available
    if 'likeCount' in processed_df.columns:
        processed_df['engagement_score'] = pd.to_numeric(processed_df['likeCount'], errors='coerce').fillna(0)
    else:
        processed_df['engagement_score'] = 0
    
    # Add reply indicator for threading analysis
    if 'parentCommentId' in processed_df.columns:
        processed_df['is_reply'] = processed_df['parentCommentId'].notna()
    else:
        processed_df['is_reply'] = False
    
    # Add video context if available
    if 'videoId' in processed_df.columns:
        processed_df['video_id'] = processed_df['videoId']
    
    # Filter out invalid entries
    processed_df = processed_df.dropna(subset=['timestamp', 'text'])
    processed_df = processed_df[processed_df['text'].str.len() > 0]
    
    # Add dataset type identifier
    processed_df['dataset_type'] = 'comments'
    
    logger.info(f"Comments preprocessing complete: {len(processed_df):,} valid rows")
    return processed_df

def preprocess_videos_dataset(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Preprocess YouTube videos dataset based on schema.
    
    Expected columns: channelId, videoId, title, description, tags, defaultLanguage,
                     defaultAudioLanguage, contentDuration, viewCount, likeCount,
                     favouriteCount, commentCount, topicCategories, publishedAt
    """
    if df.empty:
        return None
    
    logger.info(f"Preprocessing videos dataset: {len(df):,} rows")
    
    # Check for required columns
    required_cols = ['title', 'publishedAt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns for videos: {missing_cols}")
        return None
    
    # Create standardized columns for text processing
    processed_df = df.copy()
    
    # Standardize timestamp column
    processed_df['timestamp'] = pd.to_datetime(processed_df['publishedAt'], errors='coerce')
    
    # Combine title, description, and tags for comprehensive text analysis
    text_parts = []
    if 'title' in processed_df.columns:
        text_parts.append(processed_df['title'].fillna('').astype(str))
    if 'description' in processed_df.columns:
        text_parts.append(processed_df['description'].fillna('').astype(str))
    if 'tags' in processed_df.columns:
        # Handle tags - they might be stored as strings or lists
        tags_text = processed_df['tags'].fillna('').astype(str)
        text_parts.append(tags_text)
    
    if text_parts:
        processed_df['text'] = text_parts[0]
        for part in text_parts[1:]:
            processed_df['text'] += ' ' + part
    else:
        processed_df['text'] = ''
    
    # Calculate comprehensive engagement score
    engagement_components = []
    if 'viewCount' in processed_df.columns:
        views = pd.to_numeric(processed_df['viewCount'], errors='coerce').fillna(0)
        engagement_components.append(views * 0.1)  # Weight views lower
    if 'likeCount' in processed_df.columns:
        likes = pd.to_numeric(processed_df['likeCount'], errors='coerce').fillna(0)
        engagement_components.append(likes * 5)  # Weight likes higher
    if 'commentCount' in processed_df.columns:
        comments = pd.to_numeric(processed_df['commentCount'], errors='coerce').fillna(0)
        engagement_components.append(comments * 10)  # Weight comments highest
    if 'favouriteCount' in processed_df.columns:
        favorites = pd.to_numeric(processed_df['favouriteCount'], errors='coerce').fillna(0)
        engagement_components.append(favorites * 15)  # Weight favorites very high
    
    if engagement_components:
        processed_df['engagement_score'] = sum(engagement_components)
    else:
        processed_df['engagement_score'] = 0
    
    # Add content metadata
    if 'contentDuration' in processed_df.columns:
        processed_df['content_duration'] = processed_df['contentDuration']
    
    if 'topicCategories' in processed_df.columns:
        processed_df['topic_categories'] = processed_df['topicCategories']
    
    if 'videoId' in processed_df.columns:
        processed_df['video_id'] = processed_df['videoId']
    
    # Filter out invalid entries
    processed_df = processed_df.dropna(subset=['timestamp', 'text'])
    processed_df = processed_df[processed_df['text'].str.len() > 0]
    
    # Add dataset type identifier
    processed_df['dataset_type'] = 'videos'
    
    logger.info(f"Videos preprocessing complete: {len(processed_df):,} valid rows")
    return processed_df

def prepare_text_df_optimized(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Optimized text DataFrame preparation for standardized comment/video datasets."""
    if df.empty:
        return None
        
    # Use standardized columns from preprocessing
    if 'timestamp' not in df.columns or 'text' not in df.columns:
        logger.error("Missing required columns: timestamp, text")
        return None
    
    # Optimize column selection with standardized schema
    use = df[['timestamp', 'text']].copy()
    
    # Add engagement and dataset type info if available
    if 'engagement_score' in df.columns:
        use['engagement_score'] = df['engagement_score']
    if 'dataset_type' in df.columns:
        use['dataset_type'] = df['dataset_type']
    if 'is_reply' in df.columns:
        use['is_reply'] = df['is_reply']
    if 'video_id' in df.columns:
        use['video_id'] = df['video_id']
    
    # Use the already processed timestamp
    use["ts"] = use["timestamp"]
    use = use.dropna(subset=["ts"])
    
    if use.empty:
        return None
    
    # Optimized text processing
    use["text_raw"] = use["text"].astype(str)
    
    # Batch text cleaning for better performance
    logger.info(f"Processing {len(use):,} text records")
    use["text_clean"] = use["text_raw"].apply(clean_text_optimized)
    use["hashtags"] = use["text_raw"].apply(extract_hashtags_optimized)
    
    # Select final columns
    final_cols = ["ts", "text_clean", "hashtags"]
    if 'engagement_score' in use.columns:
        final_cols.append('engagement_score')
    if 'dataset_type' in use.columns:
        final_cols.append('dataset_type')
    if 'is_reply' in use.columns:
        final_cols.append('is_reply')
    if 'video_id' in use.columns:
        final_cols.append('video_id')
    
    return use[final_cols]

def load_samples_optimized() -> Dict[str, pd.DataFrame]:
    """Optimized data loading with memory management and dataset type separation."""
    logger.info("Loading processed datasets with memory optimization and dataset type separation")
    out: Dict[str, pd.DataFrame] = {}
    
    # Separate comment and video files for optimized processing
    comment_files = [p for p in PROC_DIR.glob('comments*.parquet') if not p.name.endswith('.sample.parquet')]
    video_files = [p for p in PROC_DIR.glob('videos*.parquet') if not p.name.endswith('.sample.parquet')]
    
    # If no full files, fall back to sample files
    if not comment_files:
        comment_files = [p for p in PROC_DIR.glob('comments*.sample.parquet')]
    if not video_files:
        video_files = [p for p in PROC_DIR.glob('videos*.sample.parquet')]
    
    all_files = comment_files + video_files
    logger.info(f"Found {len(comment_files)} comment files and {len(video_files)} video files")
    
    total_memory_mb = 0
    for p in all_files:
        name = p.stem
        dataset_type = "comments" if "comment" in p.name.lower() else "videos"
        
        try:
            df = pd.read_parquet(p)
            size_mb = get_dataset_size_mb(df)
            total_memory_mb += size_mb
            
            logger.info(f"Loaded {name}: {len(df):,} rows, {size_mb:.1f}MB")
            
            # Preprocess based on dataset type
            if dataset_type == "comments":
                processed_df = preprocess_comments_dataset(df)
            else:  # videos
                processed_df = preprocess_videos_dataset(df)
            
            if processed_df is not None:
                # Apply sampling if configured
                if CONFIG["sample_rows_per_source"]:
                    processed_df = sample_large_dataset(processed_df, CONFIG["sample_rows_per_source"])
                
                out[name] = processed_df
                logger.info(f"Processed {name}: {len(processed_df):,} rows after preprocessing")
            else:
                logger.warning(f"Failed to preprocess {name}")
                
        except Exception as e:
            logger.error(f"Failed to load {p}: {e}")
    
    logger.info(f"Total memory usage: {total_memory_mb:.1f}MB across {len(out)} datasets")
    return out
