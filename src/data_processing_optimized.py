#!/usr/bin/env python3
"""
Data Processing Pipeline for L'Oréal Datathon 2025
Optimized implementation focusing on text cleaning, audio processing, and trend detection.

Objective: Clean the data and transform it into features suitable for modeling.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Generator, Any
import logging
from datetime import datetime, timedelta
import emoji
from tqdm import tqdm
import json
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import gc

# Required for pyarrow file information
import pyarrow.parquet as pq

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import librosa for audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - audio processing will be disabled")

# Define paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed" / "dataset"
INTERIM_DIR = ROOT / "data" / "interim"

# Create directories if they don't exist
for dir_path in [RAW_DIR, PROC_DIR, INTERIM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration for large-scale processing
CHUNK_SIZE = 50000  # Process data in chunks of 50K rows
MAX_WORKERS = min(cpu_count(), 8)  # Limit CPU usage
MEMORY_LIMIT_MB = 2000  # Memory limit per chunk in MB

# Text cleaning patterns (pre-compiled for performance)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
MENTION_PATTERN = re.compile(r'@\w+', re.IGNORECASE)
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s#]', re.UNICODE)
HASHTAG_PATTERN = re.compile(r'#(\w+)')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Fast keyword lookup sets for early filtering
BEAUTY_KEYWORDS = {
    'skincare': ['skincare', 'skin care', 'serum', 'moisturizer', 'cleanser', 'toner', 
                 'sunscreen', 'spf', 'retinol', 'niacinamide', 'hyaluronic acid', 
                 'vitamin c', 'glycolic acid', 'salicylic acid', 'barrier repair',
                 'double cleanse', 'glass skin', 'dewy skin', 'skin cycling'],
    'makeup': ['makeup', 'foundation', 'concealer', 'mascara', 'eyeliner', 'lipstick',
               'eyeshadow', 'blush', 'highlighter', 'contour', 'bronzer', 'primer',
               'setting spray', 'lip gloss', 'lip tint', 'brow gel'],
    'beauty': ['beauty', 'glow', 'radiant', 'aesthetic', 'selfcare', 'self care',
               'routine', 'transformation', 'grwm', 'get ready', 'before after'],
    'fashion': ['fashion', 'style', 'outfit', 'ootd', 'fashion week', 'trends',
                'chic', 'elegant', 'stylish', 'wardrobe', 'accessories'],
    'hair': ['hair', 'shampoo', 'conditioner', 'hair mask', 'hair oil',
             'scalp care', 'hair growth', 'keratin', 'heat protectant']
}

# Fast keyword lookup set for early filtering
BEAUTY_FAST_KEYWORDS = set()
for keywords in BEAUTY_KEYWORDS.values():
    BEAUTY_FAST_KEYWORDS.update(keywords)

# Define stop words for text processing
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'among', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
}

class VectorizedTextCleaner:
    """Optimized text data cleaning using vectorized operations for large datasets."""
    
    def __init__(self, handle_emojis: str = 'remove'):
        """
        Initialize text cleaner.
        
        Args:
            handle_emojis: How to handle emojis ('remove' or 'convert')
        """
        self.handle_emojis = handle_emojis
        
    def fast_relevance_filter(self, text_series: pd.Series) -> pd.Series:
        """
        Fast pre-filter to identify potentially relevant texts using simple string contains.
        This avoids expensive processing on clearly irrelevant content.
        
        Args:
            text_series: Pandas Series of text data
            
        Returns:
            Boolean series indicating relevance
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text_series.str.lower()
        
        # Check if any beauty keyword appears in the text
        relevant_mask = pd.Series(False, index=text_series.index)
        
        # Use vectorized string contains for fast filtering
        for keyword in BEAUTY_FAST_KEYWORDS:
            relevant_mask |= text_lower.str.contains(keyword, na=False, regex=False)
        
        return relevant_mask
    
    def clean_text_vectorized(self, text_series: pd.Series) -> pd.Series:
        """
        Vectorized text cleaning for better performance on large datasets.
        
        Args:
            text_series: Pandas Series of text data
            
        Returns:
            Series of cleaned text
        """
        if text_series.empty:
            return text_series
        
        # Handle null values
        cleaned = text_series.fillna('').astype(str)
        
        # Convert to lowercase
        cleaned = cleaned.str.lower()
        
        # Handle emojis (if needed)
        if self.handle_emojis == 'remove':
            # Vectorized emoji removal using regex
            emoji_pattern = re.compile(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
                r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+',
                flags=re.UNICODE
            )
            cleaned = cleaned.str.replace(emoji_pattern, ' ', regex=True)
        
        # Remove URLs
        cleaned = cleaned.str.replace(URL_PATTERN, ' ', regex=True)
        
        # Remove user mentions
        cleaned = cleaned.str.replace(MENTION_PATTERN, ' ', regex=True)
        
        # Remove special characters (except hashtags)
        cleaned = cleaned.str.replace(SPECIAL_CHARS_PATTERN, ' ', regex=True)
        
        # Normalize whitespace
        cleaned = cleaned.str.replace(WHITESPACE_PATTERN, ' ', regex=True).str.strip()
        
        return cleaned
    
    def extract_hashtags_vectorized(self, text_series: pd.Series) -> pd.Series:
        """Vectorized hashtag extraction."""
        if text_series.empty:
            return pd.Series(dtype=object)
        
        # Use str.findall to extract hashtags
        hashtags = text_series.str.findall(r'#(\w+)')
        
        # Convert to lowercase
        return hashtags.apply(lambda x: [tag.lower() for tag in x] if x else [])
    
    def tokenize_and_filter_vectorized(self, text_series: pd.Series) -> pd.Series:
        """
        Vectorized tokenization and stop word removal.
        
        Args:
            text_series: Series of cleaned text
            
        Returns:
            Series of token lists
        """
        if text_series.empty:
            return pd.Series(dtype=object)
        
        # Split into tokens
        tokens = text_series.str.split()
        
        # Filter out stop words and short tokens
        filtered = tokens.apply(
            lambda x: [token for token in (x or []) 
                      if len(token) > 2 and token not in STOP_WORDS]
        )
        
        return filtered

class OptimizedCategoryClassifier:
    """Optimized category classification with vectorized operations."""
    
    def __init__(self):
        # Pre-compile patterns for faster matching
        self.category_patterns = {}
        for category, keywords in BEAUTY_KEYWORDS.items():
            pattern = '|'.join(re.escape(keyword) for keyword in keywords)
            self.category_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def classify_text_vectorized(self, text_series: pd.Series) -> pd.Series:
        """
        Vectorized text classification.
        
        Args:
            text_series: Series of text to classify
            
        Returns:
            Series of category labels
        """
        if text_series.empty:
            return pd.Series(dtype=object)
        
        # Initialize with None
        categories = pd.Series(None, index=text_series.index)
        
        # Check each category pattern (patterns are already compiled with case insensitive flag)
        for category, pattern in self.category_patterns.items():
            mask = text_series.str.contains(pattern, na=False, regex=True)
            # Only assign category if not already assigned (first match wins)
            categories = categories.where(categories.notna(), 
                                        categories.where(~mask, category))
        
        return categories

class AudioProcessor:
    """Audio data processing using librosa."""
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_audio_features(self, audio_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Extract MFCC, chroma, and spectral contrast features from audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing extracted features
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available - skipping audio processing")
            return {}
            
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Extract chroma features
            chroma = librosa.feature.chroma(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Extract spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(spectral_contrast, axis=1)
            contrast_std = np.std(spectral_contrast, axis=1)
            
            # Extract additional features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            features = {
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'chroma_mean': chroma_mean,
                'chroma_std': chroma_std,
                'spectral_contrast_mean': contrast_mean,
                'spectral_contrast_std': contrast_std,
                'tempo': tempo,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return {}
    
    def create_audio_fingerprint(self, audio_path: Union[str, Path]) -> Optional[str]:
        """
        Create audio fingerprint for similarity matching.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio fingerprint as hex string
        """
        try:
            features = self.extract_audio_features(audio_path)
            if not features:
                return None
            
            # Combine key features for fingerprinting
            fingerprint_features = np.concatenate([
                features.get('mfcc_mean', []),
                features.get('chroma_mean', []),
                features.get('spectral_contrast_mean', [])
            ])
            
            # Create hash from features
            import hashlib
            fingerprint = hashlib.md5(fingerprint_features.tobytes()).hexdigest()
            return fingerprint
            
        except Exception as e:
            logger.error(f"Error creating fingerprint for {audio_path}: {e}")
            return None

class TimeSeriesEngineer:
    """Time-series engineering for trend detection."""
    
    def __init__(self):
        self.time_intervals = {
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }
    
    def create_time_bins(self, timestamps: pd.Series, interval: str = '6h') -> pd.Series:
        """
        Aggregate timestamps into time bins.
        
        Args:
            timestamps: Series of timestamps
            interval: Time interval for binning ('1h', '6h', '1d', '1w')
            
        Returns:
            Series of binned timestamps
        """
        if interval not in self.time_intervals:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)
        
        # Create bins based on interval
        if interval == '1h':
            return timestamps.dt.floor('H')
        elif interval == '6h':
            # Floor to 6-hour intervals starting from midnight
            return timestamps.dt.floor('6H')
        elif interval == '1d':
            return timestamps.dt.floor('D')
        elif interval == '1w':
            return timestamps.dt.floor('W')
    
    def aggregate_features(self, df: pd.DataFrame, feature_col: str, 
                         timestamp_col: str, interval: str = '6h') -> pd.DataFrame:
        """
        Aggregate features into time bins and calculate counts.
        
        Args:
            df: Input dataframe
            feature_col: Column containing features (hashtags, keywords, etc.)
            timestamp_col: Column containing timestamps
            interval: Time interval for aggregation
            
        Returns:
            Aggregated dataframe with counts per time bin
        """
        # Create time bins
        df = df.copy()
        df['time_bin'] = self.create_time_bins(df[timestamp_col], interval)
        
        # Aggregate by time bin and feature
        aggregated = df.groupby(['time_bin', feature_col]).size().reset_index(name='count')
        
        # Calculate additional metrics
        aggregated['unique_audio_ids'] = df.groupby(['time_bin', feature_col])['audio_id'].nunique().values if 'audio_id' in df.columns else 0
        aggregated['unique_hashtags'] = df.groupby(['time_bin', feature_col])['hashtags'].apply(lambda x: len(set([h for sublist in x for h in sublist]))).values if 'hashtags' in df.columns else 0
        
        # Calculate rolling statistics
        aggregated = aggregated.sort_values(['time_bin', feature_col])
        aggregated['rolling_mean'] = aggregated.groupby(feature_col)['count'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        aggregated['rate_of_change'] = aggregated.groupby(feature_col)['count'].pct_change().fillna(0)
        
        return aggregated

class TrendCandidateTable:
    """Create and manage the trend candidate master table."""
    
    def __init__(self):
        self.trend_data = []
    
    def add_trend_candidate(self, timestamp: datetime, feature: str, count: int,
                          rolling_mean: float, rate_of_change: float,
                          category: str = None, audio_ids: int = 0,
                          hashtag_count: int = 0) -> None:
        """
        Add a trend candidate to the master table.
        
        Args:
            timestamp: Timestamp for the trend
            feature: Feature name (hashtag, keyword, etc.)
            count: Count for this time period
            rolling_mean: Rolling mean of counts
            rate_of_change: Rate of change from previous period
            category: Category classification
            audio_ids: Number of unique audio IDs
            hashtag_count: Number of unique hashtags
        """
        candidate = {
            'timestamp': timestamp,
            'feature': feature,
            'count': count,
            'rolling_mean': rolling_mean,
            'rate_of_change': rate_of_change,
            'category': category,
            'unique_audio_ids': audio_ids,
            'unique_hashtags': hashtag_count,
            'created_at': datetime.now()
        }
        self.trend_data.append(candidate)
    
    def get_trend_table(self) -> pd.DataFrame:
        """Get the trend candidate table as a DataFrame."""
        if not self.trend_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trend_data)
        
        # Sort by timestamp and rate of change
        df = df.sort_values(['timestamp', 'rate_of_change'], ascending=[True, False])
        
        return df
    
    def save_trend_table(self, filepath: Union[str, Path]) -> None:
        """Save the trend candidate table to file."""
        df = self.get_trend_table()
        if not df.empty:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved trend candidate table with {len(df)} entries to {filepath}")

class DatasetPreprocessor:
    """Dataset-specific preprocessing for comments and videos."""
    
    def __init__(self):
        self.text_cleaner = VectorizedTextCleaner(handle_emojis='remove')
        
    def detect_dataset_type(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect whether dataframe contains comments or videos data.
        
        Args:
            df: Input dataframe
            
        Returns:
            'comments', 'videos', or None if unable to determine
        """
        if df.empty:
            return None
            
        columns = set(df.columns)
        
        # Check for comments schema
        comment_indicators = {'textOriginal', 'commentId', 'parentCommentId'}
        if comment_indicators.issubset(columns):
            return 'comments'
            
        # Check for videos schema  
        video_indicators = {'title', 'description', 'videoId'}
        if video_indicators.issubset(columns):
            return 'videos'
            
        # Fallback check - if we have textOriginal, likely comments
        if 'textOriginal' in columns:
            return 'comments'
        elif 'title' in columns:
            return 'videos'
            
        return None
    
    def preprocess_comments_dataset(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
        
        # Standardize text column - use textOriginal for comments
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
        
        # Add author context if available
        if 'authorId' in processed_df.columns:
            processed_df['author_id'] = processed_df['authorId']
            
        # Filter out invalid entries
        processed_df = processed_df.dropna(subset=['timestamp', 'text'])
        processed_df = processed_df[processed_df['text'].str.len() > 0]
        
        # Add dataset type identifier
        processed_df['dataset_type'] = 'comments'
        
        logger.info(f"Comments preprocessing complete: {len(processed_df):,} valid rows")
        return processed_df
    
    def preprocess_videos_dataset(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
                processed_df['text'] = processed_df['text'] + ' ' + part
        else:
            processed_df['text'] = ''
        
        # Calculate comprehensive engagement score for videos
        engagement_components = []
        if 'viewCount' in processed_df.columns:
            views = pd.to_numeric(processed_df['viewCount'], errors='coerce').fillna(0)
            engagement_components.append(views * 0.1)  # Weight views lower
        if 'likeCount' in processed_df.columns:
            likes = pd.to_numeric(processed_df['likeCount'], errors='coerce').fillna(0)
            engagement_components.append(likes * 2.0)  # Weight likes higher
        if 'commentCount' in processed_df.columns:
            comments = pd.to_numeric(processed_df['commentCount'], errors='coerce').fillna(0)
            engagement_components.append(comments * 1.5)  # Weight comments high
        if 'favouriteCount' in processed_df.columns:
            favorites = pd.to_numeric(processed_df['favouriteCount'], errors='coerce').fillna(0)
            engagement_components.append(favorites * 3.0)  # Weight favorites highest
        
        if engagement_components:
            processed_df['engagement_score'] = sum(engagement_components)
        else:
            processed_df['engagement_score'] = 0
        
        # Add video-specific metadata
        if 'contentDuration' in processed_df.columns:
            processed_df['content_duration'] = processed_df['contentDuration']
        if 'videoId' in processed_df.columns:
            processed_df['video_id'] = processed_df['videoId']
        if 'channelId' in processed_df.columns:
            processed_df['channel_id'] = processed_df['channelId']
        if 'topicCategories' in processed_df.columns:
            processed_df['topic_categories'] = processed_df['topicCategories']
            
        # Filter out invalid entries
        processed_df = processed_df.dropna(subset=['timestamp', 'text'])
        processed_df = processed_df[processed_df['text'].str.len() > 0]
        
        # Add dataset type identifier
        processed_df['dataset_type'] = 'videos'
        
        logger.info(f"Videos preprocessing complete: {len(processed_df):,} valid rows")
        return processed_df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Automatically detect dataset type and apply appropriate preprocessing.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe with standardized columns
        """
        dataset_type = self.detect_dataset_type(df)
        
        if dataset_type == 'comments':
            return self.preprocess_comments_dataset(df)
        elif dataset_type == 'videos':
            return self.preprocess_videos_dataset(df)
        else:
            logger.warning(f"Unable to determine dataset type for dataframe with columns: {list(df.columns)}")
            return None

class OptimizedDataProcessor:
    """Optimized main data processing pipeline for large-scale data."""
    
    def __init__(self):
        self.text_cleaner = VectorizedTextCleaner(handle_emojis='remove')
        self.category_classifier = OptimizedCategoryClassifier()
        self.audio_processor = AudioProcessor()
        self.time_engineer = TimeSeriesEngineer()
        self.trend_table = TrendCandidateTable()
        self.preprocessor = DatasetPreprocessor()
        
    def chunked_reader(self, filepath: Union[str, Path], chunk_size: int = CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
        """
        Read large parquet files in chunks to manage memory usage.
        
        Args:
            filepath: Path to parquet file
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        try:
            # For parquet files, we need to use pyarrow for chunked reading
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(filepath)
            
            # Read in batches
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df_chunk = batch.to_pandas()
                yield df_chunk
                
        except Exception as e:
            logger.warning(f"Could not read {filepath} in chunks: {e}. Loading full file.")
            # Fallback to reading entire file
            df = pd.read_parquet(filepath)
            
            # Split into chunks manually
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                yield df.iloc[start:end].copy()
    
    def process_text_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of text data with optimized operations.
        
        Args:
            chunk: DataFrame chunk to process
            
        Returns:
            Processed chunk
        """
        if chunk.empty:
            return pd.DataFrame()
        
        # Processing steps with progress tracking
        processing_steps = [
            ("Dataset preprocessing", lambda: self.preprocessor.preprocess_dataset(chunk)),
            ("Relevance filtering", None),  # Special handling
            ("Text cleaning", None),  # Special handling  
            ("Category classification", None),  # Special handling
            ("Final filtering", None)  # Special handling
        ]
        
        with tqdm(total=len(processing_steps), desc=f"Processing chunk ({len(chunk):,} rows)", leave=False) as pbar:
            # Step 1: Apply dataset-specific preprocessing to standardize the data
            pbar.set_postfix(step="preprocessing")
            chunk_preprocessed = self.preprocessor.preprocess_dataset(chunk)
            pbar.update(1)
            
            if chunk_preprocessed is None or chunk_preprocessed.empty:
                return pd.DataFrame()
            
            # Step 2: Early filtering for relevance using fast string operations
            pbar.set_postfix(step="relevance filtering")
            relevant_mask = self.text_cleaner.fast_relevance_filter(chunk_preprocessed['text'])
            pbar.update(1)
            
            if not relevant_mask.any():
                logger.debug(f"No relevant content found in chunk of {len(chunk)} rows")
                return pd.DataFrame()
            
            # Filter to relevant rows only
            chunk_relevant = chunk_preprocessed[relevant_mask].copy()
            logger.debug(f"Filtered chunk from {len(chunk)} to {len(chunk_relevant)} relevant rows")
            
            # Step 3: Vectorized text processing
            pbar.set_postfix(step="text cleaning")
            chunk_relevant['cleaned_text'] = self.text_cleaner.clean_text_vectorized(chunk_relevant['text'])
            chunk_relevant['hashtags'] = self.text_cleaner.extract_hashtags_vectorized(chunk_relevant['text'])
            chunk_relevant['tokens'] = self.text_cleaner.tokenize_and_filter_vectorized(chunk_relevant['cleaned_text'])
            pbar.update(1)
            
            # Step 4: Vectorized category classification
            pbar.set_postfix(step="categorization")
            chunk_relevant['category'] = self.category_classifier.classify_text_vectorized(chunk_relevant['cleaned_text'])
            pbar.update(1)
            
            # Step 5: Final filter - only keep rows with assigned categories
            pbar.set_postfix(step="final filtering")
            final_mask = chunk_relevant['category'].notna()
            chunk_final = chunk_relevant[final_mask].copy()
            pbar.update(1)
        
        # Memory cleanup
        del chunk_preprocessed, chunk_relevant
        gc.collect()
        
        return chunk_final
    
    def process_text_data_chunked(self, df: pd.DataFrame = None, filepath: Union[str, Path] = None) -> pd.DataFrame:
        """
        Process large text data using chunked processing for memory efficiency.
        
        Args:
            df: Input dataframe (for small datasets)
            filepath: Path to parquet file (for large datasets)
            
        Returns:
            Processed dataframe with cleaned text and categories
        """
        if filepath:
            logger.info(f"Processing large text data from file: {filepath}")
            chunks_processed = []
            total_input_rows = 0
            total_output_rows = 0
            
            # First, estimate total chunks for progress bar
            try:
                # Quick count of total rows
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(filepath)
                estimated_total_rows = parquet_file.metadata.num_rows
                estimated_chunks = (estimated_total_rows // CHUNK_SIZE) + 1
                logger.info(f"Estimated {estimated_total_rows:,} rows in {estimated_chunks} chunks")
            except:
                estimated_chunks = None
                logger.info("Could not estimate total chunks, will show incremental progress")
            
            # Process file in chunks with progress bar
            chunk_iterator = self.chunked_reader(filepath)
            if estimated_chunks:
                chunk_iterator = tqdm(chunk_iterator, total=estimated_chunks, desc="Processing chunks", unit="chunk")
            else:
                chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", unit="chunk")
            
            for i, chunk in enumerate(chunk_iterator):
                total_input_rows += len(chunk)
                
                processed_chunk = self.process_text_chunk(chunk)
                
                if not processed_chunk.empty:
                    chunks_processed.append(processed_chunk)
                    total_output_rows += len(processed_chunk)
                
                # Update progress bar description with stats
                relevance_rate = (total_output_rows / total_input_rows * 100) if total_input_rows > 0 else 0
                if hasattr(chunk_iterator, 'set_postfix'):
                    chunk_iterator.set_postfix({
                        'Input': f"{total_input_rows:,}",
                        'Output': f"{total_output_rows:,}",
                        'Relevance': f"{relevance_rate:.1f}%"
                    })
            
            if chunks_processed:
                logger.info("📊 Concatenating processed chunks...")
                with tqdm(total=1, desc="Finalizing dataset") as pbar:
                    final_df = pd.concat(chunks_processed, ignore_index=True)
                    pbar.update(1)
                
                logger.info(f"✅ Chunked processing complete: {total_input_rows:,} → {len(final_df):,} rows ({len(final_df)/total_input_rows*100:.1f}% relevant)")
                return final_df
            else:
                logger.warning("No relevant data found in file")
                return pd.DataFrame()
                
        elif df is not None:
            logger.info(f"Processing {len(df):,} text records in memory")
            
            # For smaller datasets, we can still use chunking for consistency
            if len(df) > CHUNK_SIZE:
                chunks_processed = []
                num_chunks = (len(df) // CHUNK_SIZE) + 1
                
                with tqdm(total=num_chunks, desc="Processing chunks in memory", unit="chunk") as pbar:
                    for start in range(0, len(df), CHUNK_SIZE):
                        end = min(start + CHUNK_SIZE, len(df))
                        chunk = df.iloc[start:end].copy()
                        
                        processed_chunk = self.process_text_chunk(chunk)
                        if not processed_chunk.empty:
                            chunks_processed.append(processed_chunk)
                        
                        pbar.update(1)
                        pbar.set_postfix({'Rows': f"{end:,}/{len(df):,}"})
                
                if chunks_processed:
                    logger.info("📊 Concatenating processed chunks...")
                    with tqdm(total=1, desc="Finalizing dataset") as pbar:
                        result = pd.concat(chunks_processed, ignore_index=True)
                        pbar.update(1)
                    return result
                else:
                    return pd.DataFrame()
            else:
                # Small enough to process as single chunk
                with tqdm(total=1, desc="Processing single chunk") as pbar:
                    result = self.process_text_chunk(df)
                    pbar.update(1)
                return result
        
        else:
            raise ValueError("Either df or filepath must be provided")
    
    def process_text_data_chunked_with_save(self, filepath: Union[str, Path] = None, output_dir: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Process large text data and save the results, returning file paths for pipeline tracking.
        
        Args:
            filepath: Path to parquet file to process
            output_dir: Directory to save processed files
            
        Returns:
            Dictionary with processed file paths and statistics
        """
        if not filepath:
            raise ValueError("filepath must be provided")
            
        # Set default output directory
        if not output_dir:
            output_dir = PROC_DIR / "processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the data
        processed_df = self.process_text_data_chunked(filepath=filepath)
        
        if processed_df.empty:
            logger.warning(f"No processed data to save from {filepath}")
            return {
                'processed_files': [],
                'statistics': {'input_rows': 0, 'output_rows': 0},
                'source_file': str(filepath)
            }
        
        # Generate output filename
        input_path = Path(filepath)
        output_filename = f"{input_path.stem}_processed.parquet"
        output_path = output_dir / output_filename
        
        # Save the processed data
        logger.info(f"💾 Saving processed data to: {output_path}")
        processed_df.to_parquet(output_path, compression='snappy')
        
        # Extract and save features if we have processed text
        feature_files = []
        if 'processed_text' in processed_df.columns:
            feature_files = self._extract_and_save_features(processed_df, output_dir, input_path.stem)
        
        # Return tracking information
        all_files = [str(output_path)] + feature_files
        
        return {
            'processed_files': all_files,
            'statistics': {
                'input_rows': len(processed_df),
                'output_rows': len(processed_df),
                'features_extracted': len(feature_files)
            },
            'source_file': str(filepath),
            'main_output': str(output_path)
        }
    
    def _extract_and_save_features(self, df: pd.DataFrame, output_dir: Path, base_name: str) -> List[str]:
        """Extract features from processed text and save them."""
        logger.info(f"🔍 Extracting features from {len(df):,} processed records...")
        feature_files = []
        
        if 'processed_text' not in df.columns:
            return feature_files
        
        try:
            # Extract hashtags
            hashtags = []
            keywords = []
            
            with tqdm(total=len(df), desc="Extracting features") as pbar:
                for text in df['processed_text']:
                    if pd.isna(text):
                        continue
                    
                    text_str = str(text).lower()
                    
                    # Extract hashtags
                    import re
                    hashtag_matches = re.findall(r'#\w+', text_str)
                    hashtags.extend(hashtag_matches)
                    
                    # Extract beauty/fashion keywords
                    beauty_keywords = [
                        'skincare', 'makeup', 'foundation', 'concealer', 'lipstick', 'mascara',
                        'eyeshadow', 'blush', 'bronzer', 'primer', 'serum', 'moisturizer',
                        'cleanser', 'toner', 'sunscreen', 'retinol', 'hyaluronic', 'vitamin',
                        'beauty', 'cosmetics', 'routine', 'tutorial', 'review', 'haul'
                    ]
                    
                    for keyword in beauty_keywords:
                        if keyword in text_str:
                            keywords.append(keyword)
                    
                    pbar.update(1)
            
            # Save hashtags if found
            if hashtags:
                hashtag_counts = pd.Series(hashtags).value_counts().reset_index()
                hashtag_counts.columns = ['feature', 'count']
                hashtag_counts['type'] = 'hashtag'
                
                hashtag_file = output_dir / f"{base_name}_hashtags.parquet"
                hashtag_counts.to_parquet(hashtag_file, compression='snappy')
                feature_files.append(str(hashtag_file))
                logger.info(f"💾 Saved {len(hashtag_counts)} unique hashtags to {hashtag_file}")
            
            # Save keywords if found
            if keywords:
                keyword_counts = pd.Series(keywords).value_counts().reset_index()
                keyword_counts.columns = ['feature', 'count']
                keyword_counts['type'] = 'keyword'
                
                keyword_file = output_dir / f"{base_name}_keywords.parquet"
                keyword_counts.to_parquet(keyword_file, compression='snappy')
                feature_files.append(str(keyword_file))
                logger.info(f"💾 Saved {len(keyword_counts)} unique keywords to {keyword_file}")
        
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
        
        return feature_files

class DataProcessor(OptimizedDataProcessor):
    """Backward compatibility wrapper for the optimized processor."""
    
    def process_text_data(self, df: pd.DataFrame, text_col: str = None, 
                         timestamp_col: str = None) -> pd.DataFrame:
        """
        Legacy method for backward compatibility.
        
        Args:
            df: Input dataframe (raw from parquet file)
            text_col: Column containing text data (deprecated - auto-detected)
            timestamp_col: Column containing timestamps (deprecated - auto-detected)
            
        Returns:
            Processed dataframe with cleaned text and categories
        """
        return self.process_text_data_chunked(df=df)
    
    def process_audio_data(self, audio_files: List[Union[str, Path]]) -> pd.DataFrame:
        """
        Process audio files and extract features.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            DataFrame with audio features
        """
        logger.info(f"Processing {len(audio_files)} audio files")
        
        audio_data = []
        for audio_file in tqdm(audio_files, desc="Processing audio"):
            features = self.audio_processor.extract_audio_features(audio_file)
            if features:
                fingerprint = self.audio_processor.create_audio_fingerprint(audio_file)
                
                # Flatten features for dataframe
                feature_row = {
                    'audio_file': str(audio_file),
                    'fingerprint': fingerprint,
                    'filename': Path(audio_file).stem
                }
                
                # Add individual features
                for key, value in features.items():
                    if isinstance(value, np.ndarray):
                        for i, v in enumerate(value):
                            feature_row[f"{key}_{i}"] = v
                    else:
                        feature_row[key] = value
                
                audio_data.append(feature_row)
        
        if audio_data:
            return pd.DataFrame(audio_data)
        else:
            return pd.DataFrame()
    
    def create_time_series_features(self, df: pd.DataFrame, feature_col: str,
                                  timestamp_col: str, intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create time-series features for different intervals.
        
        Args:
            df: Input dataframe
            feature_col: Column containing features
            timestamp_col: Column containing timestamps
            intervals: List of time intervals to process
            
        Returns:
            Dictionary of aggregated dataframes for each interval
        """
        if intervals is None:
            intervals = ['1h', '6h']
        
        results = {}
        
        for interval in intervals:
            logger.info(f"Creating time series features for {interval} intervals")
            
            aggregated = self.time_engineer.aggregate_features(
                df, feature_col, timestamp_col, interval
            )
            
            results[interval] = aggregated
            
            # Add to trend candidate table
            for _, row in aggregated.iterrows():
                self.trend_table.add_trend_candidate(
                    timestamp=row['time_bin'],
                    feature=row[feature_col],
                    count=row['count'],
                    rolling_mean=row['rolling_mean'],
                    rate_of_change=row['rate_of_change'],
                    category=df[df[feature_col] == row[feature_col]]['category'].iloc[0] if 'category' in df.columns else None,
                    audio_ids=row.get('unique_audio_ids', 0),
                    hashtag_count=row.get('unique_hashtags', 0)
                )
        
        return results
    
    def run_full_pipeline(self, data_sources: Union[Dict[str, pd.DataFrame], Dict[str, str]]) -> Dict[str, any]:
        """
        Run the complete data processing pipeline with optimization for large datasets.
        
        Args:
            data_sources: Dictionary of data sources (DataFrames or file paths)
            
        Returns:
            Dictionary containing all processed results
        """
        logger.info("Starting optimized data processing pipeline for large-scale data")
        
        results = {
            'processed_text': {},
            'audio_features': None,
            'time_series': {},
            'trend_candidates': None
        }
        
        # Process text data sources
        for source_name, source in data_sources.items():
            if source_name == 'audio_files':
                continue  # Handle audio separately
                
            logger.info(f"Processing data from {source_name}")
            
            # Determine if source is DataFrame or file path
            if isinstance(source, (str, Path)):
                # Process large file using chunked reading
                processed_text = self.process_text_data_chunked(filepath=source)
            elif isinstance(source, pd.DataFrame) and not source.empty:
                # Process DataFrame in memory (with chunking if large)
                processed_text = self.process_text_data_chunked(df=source)
            else:
                logger.warning(f"Skipping invalid source: {source_name}")
                continue
            
            if not processed_text.empty:
                results['processed_text'][source_name] = processed_text
                
                # Optimized time series processing
                logger.info(f"Creating time series features for {source_name}")
                
                # Process hashtags in batches to avoid memory issues
                if 'hashtags' in processed_text.columns:
                    hashtag_ts = self._create_optimized_time_series(
                        processed_text, 'hashtags', source_name + '_hashtags'
                    )
                    if hashtag_ts:
                        results['time_series'][f'{source_name}_hashtags'] = hashtag_ts
                
                # Process tokens in batches
                token_ts = self._create_optimized_time_series(
                    processed_text, 'tokens', source_name + '_keywords'
                )
                if token_ts:
                    results['time_series'][f'{source_name}_keywords'] = token_ts
                    
            else:
                logger.warning(f"No relevant data found in {source_name}")
        
        # Process audio data if available
        if 'audio_files' in data_sources:
            audio_files = data_sources['audio_files']
            if isinstance(audio_files, list) and audio_files:
                results['audio_features'] = self.process_audio_data(audio_files)
        
        # Create final trend candidate table
        results['trend_candidates'] = self.trend_table.get_trend_table()
        
        # Save results
        self.save_results(results)
        
        logger.info("Optimized data processing pipeline completed successfully")
        return results
    
    def _create_optimized_time_series(self, df: pd.DataFrame, feature_col: str, name: str) -> Dict[str, pd.DataFrame]:
        """
        Create time series features in an optimized way for large datasets.
        
        Args:
            df: Input dataframe
            feature_col: Column containing features (lists)
            name: Name for logging
            
        Returns:
            Dictionary of time series DataFrames
        """
        try:
            # Check if we have the required columns
            if feature_col not in df.columns or 'timestamp' not in df.columns:
                logger.warning(f"Missing columns for time series: {feature_col} or timestamp")
                return {}
            
            # Process in chunks to avoid memory issues with explode
            chunk_size = 10000  # Smaller chunks for explode operations
            all_chunks = []
            
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                chunk = df.iloc[start:end].copy()
                
                # Explode the feature column
                exploded_chunk = chunk.explode(feature_col).dropna(subset=[feature_col])
                if not exploded_chunk.empty:
                    all_chunks.append(exploded_chunk)
            
            if not all_chunks:
                return {}
            
            # Combine chunks
            exploded_df = pd.concat(all_chunks, ignore_index=True)
            logger.info(f"Exploded {name}: {len(df):,} → {len(exploded_df):,} rows")
            
            # Create time series for different intervals
            intervals = ['1h', '6h']  # Reduced intervals for performance
            results = {}
            
            for interval in intervals:
                logger.debug(f"Creating {interval} time series for {name}")
                aggregated = self.time_engineer.aggregate_features(
                    exploded_df, feature_col, 'timestamp', interval
                )
                results[interval] = aggregated
            
            return results
            
        except Exception as e:
            logger.error(f"Error creating time series for {name}: {e}")
            return {}

    def save_results(self, results: Dict) -> None:
        """Save processing results to files with optimization for large datasets."""
        
        # Save processed text data
        for source_name, df in results['processed_text'].items():
            filepath = PROC_DIR / f"processed_text_{source_name}.parquet"
            # Use compression for large files
            df.to_parquet(filepath, index=False, compression='snappy')
            logger.info(f"Saved processed text data to {filepath} ({len(df):,} rows)")
        
        # Save audio features
        if results['audio_features'] is not None and not results['audio_features'].empty:
            filepath = PROC_DIR / "audio_features.parquet"
            results['audio_features'].to_parquet(filepath, index=False, compression='snappy')
            logger.info(f"Saved audio features to {filepath}")
        
        # Save time series data
        for source_name, intervals in results['time_series'].items():
            for interval, df in intervals.items():
                filepath = PROC_DIR / f"timeseries_{source_name}_{interval}.parquet"
                df.to_parquet(filepath, index=False, compression='snappy')
                logger.info(f"Saved time series data to {filepath}")
        
        # Save trend candidate table
        if results['trend_candidates'] is not None and not results['trend_candidates'].empty:
            self.trend_table.save_trend_table(PROC_DIR / "trend_candidates.parquet")

def create_large_test_dataset(num_rows: int = 100000) -> pd.DataFrame:
    """
    Create a large test dataset for performance testing.
    
    Args:
        num_rows: Number of rows to generate
        
    Returns:
        Large test DataFrame
    """
    logger.info(f"Creating test dataset with {num_rows:,} rows")
    
    import random
    
    # Beauty/fashion content templates
    content_templates = [
        "Love this #{} routine! #{} #beauty",
        "Best #{} tutorial ever! #{} #{}",
        "This #{} {} is amazing! #{}",
        "Get ready with me! #{} #{} #grwm",
        "#{} of the day! #{} #fashion",
        "My #{} essentials! #{} #selfcare",
        "#{} transformation! #{} #beforeafter",
        "New #{} haul! #{} #shopping",
        "#{} tips and tricks! #{} #beauty",
        "Daily #{} routine! #{} #skincare"
    ]
    
    # Keywords for filling templates
    beauty_words = list(BEAUTY_FAST_KEYWORDS)
    hashtags = ['glowup', 'selfcare', 'beauty', 'makeup', 'skincare', 'fashion', 'style', 'routine']
    
    # Generate data
    data = []
    for i in range(num_rows):
        template = random.choice(content_templates)
        words = random.choices(beauty_words, k=3)
        tags = random.choices(hashtags, k=2)
        
        text = template.format(*words, *tags)
        
        # Add some irrelevant content (should be filtered out)
        if i % 10 == 0:
            text = f"Random content about cooking and sports {i}"
        
        data.append({
            'commentId': f'c{i}',
            'textOriginal': text,
            'likeCount': random.randint(0, 1000),
            'publishedAt': pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i % 24),
            'videoId': f'v{i % 1000}',
            'authorId': f'a{i % 10000}'
        })
    
    return pd.DataFrame(data)

def performance_test(num_rows: int = 100000):
    """
    Test performance of optimized vs standard processing.
    
    Args:
        num_rows: Number of rows to test with
    """
    import time
    
    logger.info(f"=== PERFORMANCE TEST: {num_rows:,} ROWS ===")
    
    # Create test data
    test_df = create_large_test_dataset(num_rows)
    logger.info(f"Test dataset created: {len(test_df):,} rows, {test_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Test optimized processor
    logger.info("Testing OptimizedDataProcessor...")
    start_time = time.time()
    
    optimized_processor = OptimizedDataProcessor()
    optimized_results = optimized_processor.process_text_data_chunked(df=test_df)
    
    optimized_time = time.time() - start_time
    
    logger.info(f"Optimized processing completed in {optimized_time:.2f} seconds")
    logger.info(f"Processed {len(test_df):,} → {len(optimized_results):,} rows ({len(optimized_results)/len(test_df)*100:.1f}% relevant)")
    logger.info(f"Processing rate: {len(test_df)/optimized_time:,.0f} rows/second")
    
    # Estimate time for 7 million rows
    estimated_time_7m = (7_000_000 / len(test_df)) * optimized_time
    logger.info(f"Estimated time for 7M rows: {estimated_time_7m/60:.1f} minutes")
    
    if estimated_time_7m <= 3600:  # 1 hour
        logger.info("✅ Target of <1 hour for 7M rows should be achievable!")
    else:
        logger.warning(f"⚠️  May exceed 1 hour target. Consider further optimization.")
    
    return optimized_results, optimized_time

def main():
    """Main function to run the data processing pipeline."""
    
    logger.info("L'Oréal Datathon 2025 - Optimized Data Processing Pipeline")
    
    # Performance test with different sizes
    performance_test(10000)   # Small test
    performance_test(50000)   # Medium test
    
    # Sample data demo
    logger.info("\n=== SAMPLE DATA DEMO ===")
    
    # Create sample data that matches the actual YouTube dataset schemas
    sample_data = {
        'comments': pd.DataFrame({
            'commentId': ['c1', 'c2', 'c3', 'c4', 'c5'],
            'parentCommentId': [None, 'c1', None, None, None],
            'channelId': ['ch1', 'ch2', 'ch1', 'ch3', 'ch2'],
            'videoId': ['v1', 'v1', 'v2', 'v3', 'v2'],
            'authorId': ['a1', 'a2', 'a3', 'a4', 'a5'],
            'textOriginal': [
                'Love this #skincare routine! #glowup #beauty',
                'Best #makeup tutorial ever! #contour #highlight',
                'This #niacinamide serum is amazing! #skincare',
                'Politics and news today',  # This should be filtered out
                '#OOTD looking fresh! #fashion #style'
            ],
            'likeCount': [15, 23, 8, 2, 12],
            'publishedAt': pd.date_range('2025-01-01', periods=5, freq='6H'),
            'updatedAt': pd.date_range('2025-01-01', periods=5, freq='6H')
        }),
        'videos': pd.DataFrame({
            'channelId': ['ch1', 'ch2', 'ch3', 'ch4'],
            'videoId': ['v1', 'v2', 'v3', 'v4'],
            'title': [
                'Ultimate Skincare Routine for Glowing Skin',
                'Get Ready With Me: Makeup Tutorial',
                'Random cooking video',  # This should be filtered out
                'Fashion Haul: Spring 2025 Trends'
            ],
            'description': [
                'In this video I share my complete skincare routine with #niacinamide and #retinol',
                'Full makeup tutorial with #contour and #highlight techniques #beauty',
                'Cooking pasta with tomatoes',  # This should be filtered out
                'Showing you the latest #fashion trends for spring #style #ootd'
            ],
            'tags': [
                'skincare,beauty,niacinamide,routine',
                'makeup,tutorial,beauty,contour',
                'cooking,food,pasta',
                'fashion,style,haul,trends'
            ],
            'defaultLanguage': ['en', 'en', 'en', 'en'],
            'contentDuration': ['PT10M30S', 'PT15M45S', 'PT8M20S', 'PT12M10S'],
            'viewCount': [150000, 89000, 25000, 67000],
            'likeCount': [8500, 4200, 850, 3200],
            'commentCount': [320, 180, 45, 150],
            'favouriteCount': [450, 220, 30, 180],
            'publishedAt': pd.date_range('2025-01-01', periods=4, freq='12H')
        })
    }
    
    # Initialize processor
    processor = DataProcessor()
    
    # Run pipeline
    results = processor.run_full_pipeline(sample_data)
    
    # Print summary showing different handling of comments vs videos
    print("\n" + "="*60)
    print("DATA PROCESSING SUMMARY")
    print("="*60)
    
    for source_name, df in results['processed_text'].items():
        print(f"\n{source_name.upper()}: {len(df)} relevant records")
        if not df.empty:
            print(f"  Dataset type: {df['dataset_type'].iloc[0]}")
            categories = df['category'].value_counts()
            print(f"  Categories: {dict(categories)}")
            
            # Show dataset-specific fields
            if df['dataset_type'].iloc[0] == 'comments':
                if 'is_reply' in df.columns:
                    replies = df['is_reply'].sum()
                    print(f"  Replies: {replies}/{len(df)}")
                if 'video_id' in df.columns:
                    unique_videos = df['video_id'].nunique()
                    print(f"  Videos commented on: {unique_videos}")
            elif df['dataset_type'].iloc[0] == 'videos':
                if 'content_duration' in df.columns:
                    print(f"  Has duration info: {df['content_duration'].notna().sum()}")
                if 'channel_id' in df.columns:
                    unique_channels = df['channel_id'].nunique()
                    print(f"  Unique channels: {unique_channels}")
            
            # Show engagement score range
            if 'engagement_score' in df.columns:
                min_eng = df['engagement_score'].min()
                max_eng = df['engagement_score'].max()
                print(f"  Engagement score range: {min_eng:.0f} - {max_eng:.0f}")
    
    if results['trend_candidates'] is not None:
        print(f"\nTREND CANDIDATES: {len(results['trend_candidates'])} entries")
        if not results['trend_candidates'].empty:
            top_trends = results['trend_candidates'].nlargest(5, 'rate_of_change')
            print("Top 5 trending features:")
            for _, row in top_trends.iterrows():
                print(f"  {row['feature']}: {row['rate_of_change']:.2%} growth")
    
    print("\n" + "="*60)
    print("SCHEMA-AWARE PROCESSING DEMONSTRATION:")
    print("- Comments: Used 'textOriginal' field, added reply indicators")
    print("- Videos: Combined title+description+tags, richer engagement scores") 
    print("="*60)

if __name__ == "__main__":
    main()