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
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import emoji
from tqdm import tqdm
import librosa
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed" / "dataset"
INTERIM_DIR = ROOT / "data" / "interim"

# Create directories if they don't exist
for dir_path in [RAW_DIR, PROC_DIR, INTERIM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Text cleaning patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
MENTION_PATTERN = re.compile(r'@\w+', re.IGNORECASE)
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s#]', re.UNICODE)
HASHTAG_PATTERN = re.compile(r'#(\w+)')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Define relevant categories for beauty/fashion/skincare
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

class TextCleaner:
    """Text data cleaning for hashtags, comments, and captions."""
    
    def __init__(self, handle_emojis: str = 'remove'):
        """
        Initialize text cleaner.
        
        Args:
            handle_emojis: How to handle emojis ('remove' or 'convert')
        """
        self.handle_emojis = handle_emojis
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, special characters and handling emojis.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis
        if self.handle_emojis == 'convert':
            # Convert emojis to text description
            text = emoji.demojize(text, delimiters=(" ", " "))
        else:
            # Remove emojis
            text = emoji.replace_emoji(text, replace='')
        
        # Remove URLs
        text = URL_PATTERN.sub(' ', text)
        
        # Remove user mentions
        text = MENTION_PATTERN.sub(' ', text)
        
        # Remove special characters (except hashtags)
        text = SPECIAL_CHARS_PATTERN.sub(' ', text)
        
        # Normalize whitespace
        text = WHITESPACE_PATTERN.sub(' ', text).strip()
        
        return text
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        if not isinstance(text, str):
            return []
        return [match.group(1).lower() for match in HASHTAG_PATTERN.finditer(text)]
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and remove stop words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens without stop words
        """
        if not isinstance(text, str):
            return []
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out stop words and short tokens
        filtered_tokens = [
            token for token in tokens 
            if len(token) > 2 and token not in STOP_WORDS
        ]
        
        return filtered_tokens

class CategoryClassifier:
    """Categorize data into Beauty/Fashion/Skincare sections."""
    
    def __init__(self):
        # Compile patterns for faster matching
        self.category_patterns = {}
        for category, keywords in BEAUTY_KEYWORDS.items():
            pattern = '|'.join(re.escape(keyword) for keyword in keywords)
            self.category_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def classify_text(self, text: str) -> Optional[str]:
        """
        Classify text into one of the beauty/fashion categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Category name or None if not relevant
        """
        if not isinstance(text, str):
            return None
        
        # Count matches for each category
        category_scores = {}
        for category, pattern in self.category_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                category_scores[category] = matches
        
        if not category_scores:
            return None
        
        # Return category with highest score
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def is_relevant(self, text: str) -> bool:
        """Check if text is relevant to beauty/fashion/skincare."""
        return self.classify_text(text) is not None

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

class DataProcessor:
    """Main data processing pipeline."""
    
    def __init__(self):
        self.text_cleaner = TextCleaner(handle_emojis='remove')
        self.category_classifier = CategoryClassifier()
        self.audio_processor = AudioProcessor()
        self.time_engineer = TimeSeriesEngineer()
        self.trend_table = TrendCandidateTable()
    
    def process_text_data(self, df: pd.DataFrame, text_col: str, 
                         timestamp_col: str) -> pd.DataFrame:
        """
        Process text data (comments, captions, etc.) through the full pipeline.
        
        Args:
            df: Input dataframe
            text_col: Column containing text data
            timestamp_col: Column containing timestamps
            
        Returns:
            Processed dataframe with cleaned text and categories
        """
        logger.info(f"Processing {len(df)} text records")
        
        # Clean text
        df['cleaned_text'] = df[text_col].apply(self.text_cleaner.clean_text)
        
        # Extract hashtags
        df['hashtags'] = df[text_col].apply(self.text_cleaner.extract_hashtags)
        
        # Tokenize and filter
        df['tokens'] = df['cleaned_text'].apply(self.text_cleaner.tokenize_and_filter)
        
        # Classify categories
        df['category'] = df['cleaned_text'].apply(self.category_classifier.classify_text)
        
        # Filter out irrelevant data
        relevant_mask = df['category'].notna()
        df_filtered = df[relevant_mask].copy()
        
        logger.info(f"Filtered to {len(df_filtered)} relevant records ({len(df_filtered)/len(df)*100:.1f}%)")
        
        return df_filtered
    
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
    
    def run_full_pipeline(self, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Run the complete data processing pipeline.
        
        Args:
            data_sources: Dictionary of data sources (text data, audio data, etc.)
            
        Returns:
            Dictionary containing all processed results
        """
        logger.info("Starting full data processing pipeline")
        
        results = {
            'processed_text': {},
            'audio_features': None,
            'time_series': {},
            'trend_candidates': None
        }
        
        # Process text data sources
        for source_name, df in data_sources.items():
            if 'text' in df.columns and 'timestamp' in df.columns:
                logger.info(f"Processing text data from {source_name}")
                
                processed_text = self.process_text_data(df, 'text', 'timestamp')
                results['processed_text'][source_name] = processed_text
                
                # Create time series for hashtags
                if 'hashtags' in processed_text.columns:
                    hashtag_df = processed_text.explode('hashtags').dropna(subset=['hashtags'])
                    if not hashtag_df.empty:
                        hashtag_ts = self.create_time_series_features(
                            hashtag_df, 'hashtags', 'timestamp', ['1h', '6h']
                        )
                        results['time_series'][f'{source_name}_hashtags'] = hashtag_ts
                
                # Create time series for keywords/tokens
                token_df = processed_text.explode('tokens').dropna(subset=['tokens'])
                if not token_df.empty:
                    keyword_ts = self.create_time_series_features(
                        token_df, 'tokens', 'timestamp', ['1h', '6h']
                    )
                    results['time_series'][f'{source_name}_keywords'] = keyword_ts
        
        # Process audio data if available
        if 'audio_files' in data_sources:
            audio_files = data_sources['audio_files']
            if isinstance(audio_files, list) and audio_files:
                results['audio_features'] = self.process_audio_data(audio_files)
        
        # Create final trend candidate table
        results['trend_candidates'] = self.trend_table.get_trend_table()
        
        # Save results
        self.save_results(results)
        
        logger.info("Data processing pipeline completed successfully")
        return results
    
    def save_results(self, results: Dict) -> None:
        """Save processing results to files."""
        
        # Save processed text data
        for source_name, df in results['processed_text'].items():
            filepath = PROC_DIR / f"processed_text_{source_name}.parquet"
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved processed text data to {filepath}")
        
        # Save audio features
        if results['audio_features'] is not None and not results['audio_features'].empty:
            filepath = PROC_DIR / "audio_features.parquet"
            results['audio_features'].to_parquet(filepath, index=False)
            logger.info(f"Saved audio features to {filepath}")
        
        # Save time series data
        for source_name, intervals in results['time_series'].items():
            for interval, df in intervals.items():
                filepath = PROC_DIR / f"timeseries_{source_name}_{interval}.parquet"
                df.to_parquet(filepath, index=False)
                logger.info(f"Saved time series data to {filepath}")
        
        # Save trend candidate table
        if results['trend_candidates'] is not None and not results['trend_candidates'].empty:
            self.trend_table.save_trend_table(PROC_DIR / "trend_candidates.parquet")

def main():
    """Main function to run the data processing pipeline."""
    
    # Example usage with sample data
    logger.info("L'Oréal Datathon 2025 - Data Processing Pipeline")
    
    # Create sample data for demonstration
    sample_data = {
        'comments': pd.DataFrame({
            'text': [
                'Love this #skincare routine! #glowup #beauty',
                'Best #makeup tutorial ever! #contour #highlight',
                'This #niacinamide serum is amazing! #skincare',
                'Politics and news today', # This should be filtered out
                '#OOTD looking fresh! #fashion #style'
            ],
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='6H')
        }),
        'captions': pd.DataFrame({
            'text': [
                'Get ready with me! #grwm #makeup #beauty',
                'Skincare routine for glowing skin #skincare #glassskin',
                'Random video about cooking', # This should be filtered out
                'Fashion haul from this weekend #fashion #haul'
            ],
            'timestamp': pd.date_range('2025-01-01', periods=4, freq='12H')
        })
    }
    
    # Initialize processor
    processor = DataProcessor()
    
    # Run pipeline
    results = processor.run_full_pipeline(sample_data)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA PROCESSING SUMMARY")
    print("="*60)
    
    for source_name, df in results['processed_text'].items():
        print(f"{source_name.upper()}: {len(df)} relevant records")
        if not df.empty:
            categories = df['category'].value_counts()
            print(f"  Categories: {dict(categories)}")
    
    if results['trend_candidates'] is not None:
        print(f"\nTREND CANDIDATES: {len(results['trend_candidates'])} entries")
        if not results['trend_candidates'].empty:
            top_trends = results['trend_candidates'].nlargest(5, 'rate_of_change')
            print("Top 5 trending features:")
            for _, row in top_trends.iterrows():
                print(f"  {row['feature']}: {row['rate_of_change']:.2%} growth")
    
    print("="*60)

if __name__ == "__main__":
    main()