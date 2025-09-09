#!/usr/bin/env python3
"""
Optimized Data Processing Pipeline for L'Oréal Datathon 2025
Performance improvements for Phase 3 and enhanced Phase 2 reporting.

Key optimizations:
1. Enhanced caching system with timestamp-based invalidation
2. Parallel processing where beneficial
3. Memory-efficient streaming for large datasets
4. Incremental processing to avoid recomputation
5. Early termination conditions for expensive operations
6. Optimized emerging terms detection algorithm
7. Enhanced Phase 2 reporting with ALL timeframe data
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, Any
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from collections import Counter, defaultdict
from scipy import stats
import warnings
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import combinations
import gc
import psutil
import logging

warnings.filterwarnings('ignore')

# Setup logging for performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Enhanced Performance Controls
# -----------------------------
PERFORMANCE_MODE = "BALANCED"  # Options: "OPTIMIZED", "BALANCED", "THOROUGH"
MAX_WORKERS = min(4, (psutil.cpu_count() or 1))  # Adaptive CPU usage
MEMORY_THRESHOLD_GB = 8  # Switch to streaming mode if dataset > this size
CACHE_VERSION = "v2.0"  # Increment to invalidate old caches

# Performance thresholds based on mode
PERF_CONFIGS = {
    "OPTIMIZED": {
        "max_features_emerging": 200,
        "max_features_anomaly": 200,
        "max_cluster_features": 100,
        "sample_rows_per_source": 50_000,
        "compute_velocity": False,
        "enable_parallel": True,
        "enable_caching": True,
        "streaming_threshold_mb": 100,
        "enable_chunking": True,
        "chunk_size": 25_000,
        "chunk_memory_limit_mb": 200,
        "chunking_threshold_rows": 75_000
    },
    "BALANCED": {
        "max_features_emerging": 400,
        "max_features_anomaly": 400,
        "max_cluster_features": 250,
        "sample_rows_per_source": 100_000,
        "compute_velocity": True,
        "enable_parallel": True,
        "enable_caching": False,
        "streaming_threshold_mb": 200,
        "enable_chunking": True,
        "chunk_size": 50_000,
        "chunk_memory_limit_mb": 300,
        "chunking_threshold_rows": 100_000
    },
    "THOROUGH": {
        "max_features_emerging": None,
        "max_features_anomaly": None,
        "max_cluster_features": None,
        "sample_rows_per_source": None,
        "compute_velocity": True,
        "enable_parallel": False,
        "enable_caching": True,
        "streaming_threshold_mb": 500,
        "enable_chunking": True,
        "chunk_size": 75_000,
        "chunk_memory_limit_mb": 400,
        "chunking_threshold_rows": 150_000
    }
}

CONFIG = PERF_CONFIGS[PERFORMANCE_MODE]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed" / "dataset"
INTERIM_DIR = ROOT / "data" / "interim"
CACHE_DIR = ROOT / "data" / "cache"

# Create directories
for dir_path in [INTERIM_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Enhanced regex patterns for better performance
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w_]+", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^\w\s#]", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F]"  # emoticons
    "|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
    "|[\U0001F680-\U0001F6FF]"  # transport & map symbols
    "|[\U0001F1E0-\U0001F1FF]"  # flags (iOS)
    , flags=re.UNICODE
)
HASHTAG_RE = re.compile(r"#(\w+)")

# Optimized stopwords set
STOPWORDS = frozenset([
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with", "on", "at", 
    "from", "by", "is", "are", "was", "were", "be", "been", "being", "this", 
    "that", "these", "those", "it", "its", "as", "if", "then", "than", "also", 
    "not", "no", "but", "so", "very", "can", "will", "just", "into", "over", 
    "under", "above", "below", "you", "your", "we", "our", "their", "they", 
    "i", "me", "my", "mine", "out", "up", "down", "across", "about", "after", 
    "before", "during", "between", "within", "without", "more", "less"
])

# Enhanced keyword list with categories
KEYWORDS = [
    # Skincare ingredients & actives
    "hyaluronic acid", "niacinamide", "salicylic acid", "glycolic acid", "lactic acid",
    "azelaic acid", "tranexamic acid", "benzoyl peroxide", "retinol", "retinoid", "bakuchiol",
    "vitamin c", "vitamin e", "panthenol", "ceramide", "ceramides", "peptides", "copper peptides",
    "squalane", "allantoin", "urea", "snail mucin", "centella", "cica", "tea tree",
    # Skincare concepts & routines
    "skin barrier", "barrier repair", "double cleanse", "oil cleanser", "exfoliation",
    "chemical exfoliant", "physical exfoliant", "moisturizer", "sunscreen", "spf", "spf 50",
    "reapply spf", "skin cycling", "skin flooding", "slugging", "glass skin", "dewy skin",
    "matte skin", "non comedogenic", "fragrance free", "cruelty free", "vegan", "clean beauty",
    "k beauty", "j beauty",
    # Makeup
    "foundation", "concealer", "blush", "cream blush", "mascara", "eyeliner", "lipstick",
    "lip oil", "lip gloss", "lip tint", "setting spray", "contour", "highlighter", "bronzer",
    "eyeshadow", "brow gel", "no makeup makeup", "soft glam", "latte makeup", "underpainting",
    # Hair
    "hair mask", "heat protectant", "leave in", "bond builder", "scalp serum", "shampoo",
    "conditioner", "sulfate free", "keratin", "argan oil", "rosemary oil", "hair growth",
    # General
    "glow", "makeup", "skincare"
]

# Category mapping for faster lookups
KEYWORD_CATEGORY: Dict[str, str] = {}
skincare_keywords = [
    "hyaluronic acid", "niacinamide", "salicylic acid", "glycolic acid", "lactic acid", "azelaic acid",
    "tranexamic acid", "benzoyl peroxide", "retinol", "retinoid", "bakuchiol", "vitamin c", "vitamin e",
    "panthenol", "ceramide", "ceramides", "peptides", "copper peptides", "squalane", "allantoin", "urea",
    "snail mucin", "centella", "cica", "tea tree", "skin barrier", "barrier repair", "double cleanse",
    "oil cleanser", "exfoliation", "chemical exfoliant", "physical exfoliant", "moisturizer", "sunscreen",
    "spf", "spf 50", "reapply spf", "skin cycling", "skin flooding", "slugging", "glass skin", "dewy skin",
    "matte skin", "non comedogenic", "fragrance free", "cruelty free", "vegan", "clean beauty", "k beauty",
    "j beauty", "skincare"
]
makeup_keywords = [
    "foundation", "concealer", "blush", "cream blush", "mascara", "eyeliner", "lipstick", "lip oil",
    "lip gloss", "lip tint", "setting spray", "contour", "highlighter", "bronzer", "eyeshadow", "brow gel",
    "no makeup makeup", "soft glam", "latte makeup", "underpainting", "makeup"
]
hair_keywords = [
    "hair mask", "heat protectant", "leave in", "bond builder", "scalp serum", "shampoo",
    "conditioner", "sulfate free", "keratin", "argan oil", "rosemary oil", "hair growth"
]
# Enhanced with Beauty and Fashion categories
beauty_keywords = [
    "beauty", "beautiful", "glow", "glowing", "radiant", "aesthetic", "skincare routine", 
    "beauty routine", "self care", "selfcare", "beauty tips", "beauty hack", "beauty trends",
    "natural beauty", "fresh faced", "glam", "glamorous", "stunning", "gorgeous"
]
fashion_keywords = [
    "fashion", "style", "outfit", "ootd", "fashion week", "trends", "runway", "designer",
    "chic", "elegant", "fashionable", "stylish", "wardrobe", "fashion blogger", "style tips",
    "fashion trends", "street style", "haute couture", "accessories", "jewelry"
]

for kw in skincare_keywords:
    KEYWORD_CATEGORY[kw] = "Skincare"
for kw in makeup_keywords:
    KEYWORD_CATEGORY[kw] = "Makeup"
for kw in hair_keywords:
    KEYWORD_CATEGORY[kw] = "Hair"
for kw in beauty_keywords:
    KEYWORD_CATEGORY[kw] = "Beauty"
for kw in fashion_keywords:
    KEYWORD_CATEGORY[kw] = "Fashion"

# Timeframe configurations
TIMEFRAME_LABELS = ["1h", "3h", "6h", "1d", "3d", "7d", "14d", "1m", "3m", "6m"]

# Target categories for relevance filtering
RELEVANT_CATEGORIES = {"Beauty", "Fashion", "Skincare", "Makeup", "Hair"}

def is_trend_relevant(feature: str, category: str = None) -> bool:
    """
    Check if a trend word is similar/relevant to Beauty/Fashion/Skincare/Makeup/Hair categories.
    
    Args:
        feature: The trend word/feature to check
        category: Optional pre-computed category to avoid recomputation
    
    Returns:
        bool: True if the trend is relevant to target categories
    """
    if not isinstance(feature, str):
        return False
    
    # Normalize the feature
    f = feature.lstrip('#').lower()
    
    # Quick exclusion filter for clearly irrelevant terms
    irrelevant_patterns = [
        "technology", "tech", "gaming", "game", "sport", "politics", "political",
        "news", "food", "recipe", "cooking", "travel", "vacation", "study", "work",
        "job", "career", "finance", "money", "crypto", "stock", "investment",
        "education", "school", "university", "medical", "health", "fitness", "gym",
        "workout", "exercise", "diet", "nutrition", "weight loss"
    ]
    
    # If any irrelevant patterns are found, exclude immediately
    for pattern in irrelevant_patterns:
        if pattern in f:
            return False
    
    # Use provided category or compute it
    if category is None:
        category = categorize_feature(feature)
    
    # Primary filter: check if category is in our target categories
    if category in RELEVANT_CATEGORIES:
        return True
    
    # Secondary filter: additional semantic similarity checks for edge cases
    # Check for common beauty/fashion related terms that might not be caught by categorization
    beauty_related_patterns = [
        # General beauty terms
        "glow", "radiant", "stunning", "gorgeous", "beautiful", "pretty", "cute", "hot",
        # Self-care and wellness
        "selfcare", "self care", "wellness", "pamper", "relax", "spa",
        # Social media beauty trends
        "getready", "grwm", "transformation", "makeover", "reveal", "before", "after",
        # Aesthetic terms
        "aesthetic", "vibe", "mood", "energy", "aura", "soft", "natural", "fresh",
        # General style terms
        "style", "stylish", "chic", "trendy", "fashionable", "classy", "elegant",
        # Product-related terms that might be missed
        "product", "brand", "review", "haul", "favorite", "recommend", "obsess",
        # Body/appearance related
        "face", "eyes", "lips", "lashes", "eyebrows", "cheeks", "complexion"
    ]
    
    # Check if any beauty-related patterns are present
    for pattern in beauty_related_patterns:
        if pattern in f:
            return True
    
    # Tertiary filter: check for brand names or product mentions (more specific)
    # Only consider as relevant if it's clearly beauty/fashion product related
    potential_beauty_product_indicators = [
        "serum", "cream", "moisturizer", "cleanser", "mask", "treatment", "oil", 
        "product review", "brand review", "beauty product", "skincare product",
        "makeup product", "hair product", "collection launch", "beauty brand",
        "skincare brand", "makeup brand", "hair brand", "beauty line",
        "collab", "collaboration", "partnership"
    ]
    
    for indicator in potential_beauty_product_indicators:
        if indicator in f and len(f) > 5:  # More stringent length requirement
            return True
    
    return False

def filter_relevant_emerging_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter emerging terms DataFrame to only include relevant Beauty/Fashion/Skincare/Makeup/Hair terms.
    
    Args:
        df: DataFrame with emerging terms data
    
    Returns:
        pd.DataFrame: Filtered DataFrame with only relevant terms
    """
    if df.empty:
        return df
    
    # Apply relevance filter
    if 'category' in df.columns:
        # Use pre-computed categories for efficiency
        df['is_relevant'] = df.apply(lambda row: is_trend_relevant(row['feature'], row['category']), axis=1)
    else:
        # Compute relevance without category
        df['is_relevant'] = df['feature'].apply(lambda x: is_trend_relevant(x))
    
    # Filter to only relevant terms
    relevant_df = df[df['is_relevant']].copy()
    
    # Drop the temporary column
    relevant_df = relevant_df.drop(columns=['is_relevant'])
    
    # Log filtering results
    original_count = len(df)
    filtered_count = len(relevant_df)
    logger.info(f"Relevance filtering: {original_count} -> {filtered_count} terms "
                f"({filtered_count/max(1,original_count)*100:.1f}% relevant)")
    
    return relevant_df

_FREQ_MAP = {
    '1h': '1H', '3h': '3H', '6h': '6H', '1d': '1D', '3d': '3D', '7d': '7D', '14d': '14D'
}

_ROLLING_WINDOW = {
    '1h': 24, '3h': 8, '6h': 4, '1d': 7, '3d': 7, '7d': 4, '14d': 4, '1m': 6, '3m': 4, '6m': 4
}

_MIN_FREQ_BY_LABEL = {
    '1h': 2, '3h': 2, '6h': 3, '1d': 5, '3d': 5, '7d': 8, '14d': 10, '1m': 15, '3m': 20, '6m': 30
}

# Growth rate thresholds tuned per timeframe (lower thresholds for longer periods)
MIN_GROWTH_RATE_BY_LABEL = {
    '1h': 2.5,
    '3h': 2.5,
    '6h': 2.5,
    '1d': 2.0,
    '3d': 1.8,
    '7d': 1.6,
    '14d': 1.5,
    '1m': 1.4,
    '3m': 1.3,
    '6m': 1.2
}

# -----------------------------
# Enhanced Caching System
# -----------------------------

class PerformanceCache:
    """Enhanced caching system with timestamp-based invalidation and compression."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path with version and hash."""
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{CACHE_VERSION}_{key_hash}.pkl"
    
    def _get_data_fingerprint(self, data: Any) -> str:
        """Generate fingerprint for data to detect changes."""
        if isinstance(data, pd.DataFrame):
            return f"{len(data)}_{data.columns.tolist()}_{hash(tuple(data.dtypes))}"
        elif isinstance(data, list) and data and isinstance(data[0], tuple):
            return f"{len(data)}_{[name for name, _ in data]}"
        return str(hash(str(data)))
    
    def get(self, cache_key: str, data_fingerprint: str = None) -> Optional[Any]:
        """Get cached data if valid and recent."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check version compatibility
            if cache_data.get('version') != CACHE_VERSION:
                logger.info(f"Cache version mismatch for {cache_key}, invalidating")
                cache_path.unlink()
                return None
            
            # Check data fingerprint if provided
            if data_fingerprint and cache_data.get('data_fingerprint') != data_fingerprint:
                logger.info(f"Data fingerprint mismatch for {cache_key}, invalidating")
                cache_path.unlink()
                return None
            
            # Check age (invalidate if older than 1 hour for optimization)
            cache_time = cache_data.get('timestamp', 0)
            if time.time() - cache_time > 3600:  # 1 hour
                logger.info(f"Cache expired for {cache_key}")
                cache_path.unlink()
                return None
                
            logger.info(f"Cache hit for {cache_key}")
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def set(self, cache_key: str, data: Any, data_fingerprint: str = None) -> None:
        """Save data to cache with metadata."""
        cache_path = self._get_cache_path(cache_key)
        try:
            cache_data = {
                'version': CACHE_VERSION,
                'timestamp': time.time(),
                'data_fingerprint': data_fingerprint,
                'data': data
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

# Global cache instance
cache = PerformanceCache(CACHE_DIR)

# -----------------------------
# Optimized Text Processing
# -----------------------------

def categorize_feature(feature: str) -> str:
    """Optimized feature categorization with enhanced Beauty/Fashion support."""
    if not isinstance(feature, str):
        return "Other"
    
    f = feature.lstrip('#').lower()
    
    # Direct keyword match (fastest)
    if f in KEYWORD_CATEGORY:
        return KEYWORD_CATEGORY[f]
    
    # Heuristic contains-based match (cached for common terms)
    for k, cat in KEYWORD_CATEGORY.items():
        if k in f:
            return cat
    
    # Enhanced heuristic categorization with Beauty/Fashion
    # Skincare patterns
    if any(tok in f for tok in ["skin", "spf", "sunscreen", "barrier", "niacinamide", "retinol", "serum", 
                                "cleanser", "moistur", "acne", "pore", "anti-aging", "wrinkle"]):
        return "Skincare"
    
    # Makeup patterns
    if any(tok in f for tok in ["lip", "lash", "brow", "blush", "contour", "eyeshadow", "mascara", 
                                "liner", "foundation", "concealer", "powder", "primer", "bronzer"]):
        return "Makeup"
    
    # Hair patterns
    if any(tok in f for tok in ["hair", "scalp", "shampoo", "conditioner", "keratin", "treatment",
                                "curl", "straight", "frizz", "volume", "hair care"]):
        return "Hair"
    
    # Beauty patterns
    if any(tok in f for tok in ["beauty", "beautiful", "glow", "radiant", "aesthetic", "selfcare",
                                "self care", "routine", "transformation", "before after", "reveal"]):
        return "Beauty"
    
    # Fashion patterns
    if any(tok in f for tok in ["fashion", "style", "outfit", "ootd", "look", "trend", "chic",
                                "elegant", "outfit of the day", "street style", "fashion week"]):
        return "Fashion"
    
    return "Other"

def clean_text_optimized(text: Optional[str]) -> str:
    """Optimized text cleaning with pre-compiled regex."""
    if not isinstance(text, str) or not text:
        return ""
    
    s = text.lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def extract_hashtags_optimized(text: str) -> List[str]:
    """Optimized hashtag extraction."""
    if not text:
        return []
    return [m.group(1).lower() for m in HASHTAG_RE.finditer(text)]

def extract_ngrams_optimized(text: str, n: int = 2, max_terms: int = 50) -> List[str]:
    """Optimized n-gram extraction with limits."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    words = text.split()
    # Pre-filter words for performance
    words = [w for w in words[:20] if len(w) >= 3 and w not in STOPWORDS]  # Limit word count
    
    if len(words) < n:
        return []
    
    ngrams = []
    for i in range(min(len(words) - n + 1, max_terms)):  # Limit n-grams
        ngram = " ".join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def extract_all_terms_optimized(text: str, min_length: int = 3, max_terms: int = 30) -> List[str]:
    """Optimized term extraction with performance limits."""
    if not isinstance(text, str):
        return []
    
    terms = []
    words = text.split()[:15]  # Limit words processed
    
    # Add individual words (filtered)
    word_count = 0
    for word in words:
        if len(word) >= min_length and word not in STOPWORDS:
            terms.append(word)
            word_count += 1
            if word_count >= max_terms // 2:  # Reserve space for n-grams
                break
    
    # Add bigrams (limited)
    terms.extend(extract_ngrams_optimized(text, 2, max_terms // 4))
    
    # Only add trigrams if we have space and they're valuable
    if len(terms) < max_terms * 0.75:
        terms.extend(extract_ngrams_optimized(text, 3, max_terms // 6))
    
    return terms[:max_terms]  # Hard limit

# -----------------------------
# Dataset-Specific Preprocessing Functions
# -----------------------------

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
            processed_df['text'] = processed_df['text'] + ' ' + part
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

# -----------------------------
# Memory-Efficient Data Loading
# -----------------------------

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
            # Check file size before loading
            file_size_mb = p.stat().st_size / (1024 * 1024)
            
            df = pd.read_parquet(p)
            
            # Apply dataset-specific preprocessing
            if dataset_type == "comments":
                df = preprocess_comments_dataset(df)
            else:
                df = preprocess_videos_dataset(df)
            
            if df is None or df.empty:
                logger.warning(f"Skipping {p.name}: no valid data after preprocessing")
                continue
            
            dataset_size_mb = get_dataset_size_mb(df)
            total_memory_mb += dataset_size_mb
            
            # Apply sampling if needed
            if CONFIG["sample_rows_per_source"] and len(df) > CONFIG["sample_rows_per_source"]:
                df = sample_large_dataset(df, CONFIG["sample_rows_per_source"])
                dataset_size_mb = get_dataset_size_mb(df)
            
            out[name] = df
            logger.info(f"Loaded {p.name} ({dataset_type}): {len(df):,} rows, {dataset_size_mb:.1f}MB")
            
            # Memory management
            if total_memory_mb > 500:  # 500MB threshold
                logger.warning("High memory usage detected, forcing garbage collection")
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to load {p}: {e}")
    
    logger.info(f"Total memory usage: {total_memory_mb:.1f}MB across {len(out)} datasets")
    return out

# -----------------------------
# Chunked Processing System
# -----------------------------

class ChunkedDataProcessor:
    """
    Advanced chunked processing system for handling large datasets efficiently.
    Splits datasets into manageable chunks and processes them independently.
    """
    
    def __init__(self, chunk_size: int = 50_000, memory_limit_mb: int = 500):
        """
        Initialize chunked processor.
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_mb: Memory limit for adaptive chunk sizing
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.processed_chunks = 0
        self.total_chunks = 0
        
    def calculate_optimal_chunk_size(self, df: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on memory usage."""
        if len(df) < 1000:
            return len(df)
        
        # Sample a small portion to estimate memory per row
        sample_size = min(1000, len(df))
        sample_df = df.head(sample_size)
        memory_per_row_mb = get_dataset_size_mb(sample_df) / sample_size
        
        # Calculate chunk size that stays under memory limit
        max_rows_per_chunk = int(self.memory_limit_mb / memory_per_row_mb)
        optimal_chunk_size = min(max_rows_per_chunk, self.chunk_size)
        
        logger.info(f"Estimated {memory_per_row_mb:.4f}MB per row, using chunk size: {optimal_chunk_size:,}")
        return max(1000, optimal_chunk_size)  # Minimum 1000 rows per chunk
    
    def create_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into chunks efficiently."""
        if df.empty:
            return []
        
        optimal_chunk_size = self.calculate_optimal_chunk_size(df)
        self.total_chunks = len(df) // optimal_chunk_size + (1 if len(df) % optimal_chunk_size > 0 else 0)
        
        chunks = []
        for i in range(0, len(df), optimal_chunk_size):
            chunk = df.iloc[i:i + optimal_chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks with ~{optimal_chunk_size:,} rows each")
        return chunks
    
    def process_chunk_hashtags(self, chunk: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
        """Process a single chunk for hashtag aggregation."""
        try:
            self.processed_chunks += 1
            logger.debug(f"Processing hashtag chunk {self.processed_chunks}/{self.total_chunks}")
            
            p = prepare_text_df_optimized(chunk)
            if p is None or p.empty:
                return None
            
            dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
            
            # Time binning
            if label in _FREQ_MAP:
                p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
            else:
                p['bin'] = assign_time_bin_optimized(p['ts'], label)
            
            # Explode hashtags
            pe = p[['bin', 'hashtags']].explode('hashtags').dropna(subset=['hashtags'])
            if pe.empty:
                return None
            
            pe = pe.rename(columns={'hashtags': 'feature'})
            
            # Apply engagement weighting if available
            if 'engagement_score' in p.columns:
                engagement_map = p.groupby('bin')['engagement_score'].mean().to_dict()
                pe['engagement_weight'] = pe['bin'].map(engagement_map).fillna(1.0)
                
                if dataset_type == 'comments':
                    pe['weighted_count'] = pe['engagement_weight']
                else:
                    pe['weighted_count'] = np.log1p(pe['engagement_weight'])
            else:
                pe['weighted_count'] = 1.0
            
            # Group and aggregate
            g = pe.groupby(['bin', 'feature'], as_index=False).agg({
                'weighted_count': 'sum'
            }).rename(columns={'weighted_count': 'count'})
            
            g['count'] = g['count'].round().astype(int).clip(lower=1)
            g['category'] = g['feature'].apply(categorize_feature)
            g['source_type'] = dataset_type
            
            return g
            
        except Exception as e:
            logger.error(f"Error processing hashtag chunk {self.processed_chunks}: {e}")
            return None
    
    def process_chunk_keywords(self, chunk: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
        """Process a single chunk for keyword aggregation."""
        try:
            self.processed_chunks += 1
            logger.debug(f"Processing keyword chunk {self.processed_chunks}/{self.total_chunks}")
            
            p = prepare_text_df_optimized(chunk)
            if p is None or p.empty:
                return None
            
            dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
            
            # Pre-compile patterns for better performance
            kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
            
            if label in _FREQ_MAP:
                p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
            else:
                p['bin'] = assign_time_bin_optimized(p['ts'], label)
            
            frames = []
            for kw, pat in kw_patterns:
                mask = p['text_clean'].str.contains(pat, na=False)
                if not mask.any():
                    continue
                
                matched_data = p.loc[mask].copy()
                
                # Apply engagement weighting
                if 'engagement_score' in matched_data.columns:
                    if dataset_type == 'comments':
                        matched_data['weight'] = matched_data['engagement_score']
                    else:
                        matched_data['weight'] = np.log1p(matched_data['engagement_score'])
                else:
                    matched_data['weight'] = 1.0
                
                g = matched_data.groupby('bin', as_index=False)['weight'].sum().rename(columns={'weight': 'count'})
                g['count'] = g['count'].round().astype(int).clip(lower=1)
                g['feature'] = kw
                g['category'] = categorize_feature(kw)
                g['source_type'] = dataset_type
                frames.append(g)
            
            if not frames:
                return None
            
            return pd.concat(frames, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error processing keyword chunk {self.processed_chunks}: {e}")
            return None
    
    def process_chunk_emerging_terms(self, chunk: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
        """Process a single chunk for emerging terms detection."""
        try:
            self.processed_chunks += 1
            logger.debug(f"Processing emerging terms chunk {self.processed_chunks}/{self.total_chunks}")
            
            p = prepare_text_df_optimized(chunk)
            if p is None or p.empty:
                return None
            
            dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
            
            # Extract terms with limits for performance
            if dataset_type == 'videos':
                p["all_terms"] = p["text_clean"].apply(
                    lambda x: extract_all_terms_optimized(x, max_terms=25)
                )
            else:
                p["all_terms"] = p["text_clean"].apply(
                    lambda x: extract_all_terms_optimized(x, max_terms=15)
                )
            
            if label in _FREQ_MAP:
                p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
            else:
                p['bin'] = assign_time_bin_optimized(p['ts'], label)
            
            # Explode and aggregate
            pe = p[["bin", "all_terms"]].explode("all_terms").dropna(subset=["all_terms"])
            if pe.empty:
                return None
            
            pe = pe.rename(columns={"all_terms": "feature"})
            
            # Add engagement context
            if 'engagement_score' in p.columns:
                engagement_map = p.groupby('bin')['engagement_score'].mean().to_dict()
                pe['weight'] = pe['bin'].map(engagement_map).fillna(1.0)
                if dataset_type == 'videos':
                    pe['weight'] = np.log1p(pe['weight'])
            else:
                pe['weight'] = 1.0
            
            result = pe.groupby(["bin", "feature"], as_index=False)['weight'].sum().rename(columns={'weight': 'count'})
            result['count'] = result['count'].round().astype(int).clip(lower=1)
            result['source_type'] = dataset_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing emerging terms chunk {self.processed_chunks}: {e}")
            return None
    
    def process_dataset_chunked(self, df: pd.DataFrame, processing_func: callable, 
                              label: str = '6h', feature_type: str = 'hashtags') -> Optional[pd.DataFrame]:
        """
        Process a large dataset using chunked approach.
        
        Args:
            df: DataFrame to process
            processing_func: Function to apply to each chunk
            label: Time label for aggregation
            feature_type: Type of features being processed
        
        Returns:
            Combined results from all chunks
        """
        if df.empty:
            return None
        
        logger.info(f"Starting chunked processing for {feature_type} ({label}): {len(df):,} rows")
        
        # Reset chunk counter
        self.processed_chunks = 0
        
        # Create chunks
        chunks = self.create_chunks(df)
        if not chunks:
            return None
        
        # Process chunks
        chunk_results = []
        with tqdm(total=len(chunks), desc=f"Processing {feature_type} chunks") as pbar:
            for i, chunk in enumerate(chunks):
                result = processing_func(chunk, label)
                if result is not None and not result.empty:
                    chunk_results.append(result)
                
                pbar.update(1)
                
                # Periodic garbage collection
                if (i + 1) % 5 == 0:
                    gc.collect()
        
        if not chunk_results:
            logger.warning(f"No valid results from chunked processing of {feature_type}")
            return None
        
        # Combine results
        logger.info(f"Combining {len(chunk_results)} chunk results for {feature_type}")
        combined_result = pd.concat(chunk_results, ignore_index=True)
        
        # Final aggregation to combine overlapping features across chunks
        if 'bin' in combined_result.columns and 'feature' in combined_result.columns:
            agg_cols = ['bin', 'feature']
            if 'source_type' in combined_result.columns:
                agg_cols.append('source_type')
            
            final_result = combined_result.groupby(agg_cols, as_index=False)['count'].sum()
            
            # Re-add other columns
            for col in ['category', 'source_type']:
                if col in combined_result.columns and col not in final_result.columns:
                    col_map = combined_result.groupby(agg_cols[:-1] if col == 'source_type' else agg_cols)[col].first().to_dict()
                    if col == 'source_type':
                        final_result[col] = final_result[agg_cols[:-1]].apply(lambda x: col_map.get(tuple(x)), axis=1)
                    else:
                        final_result[col] = final_result[agg_cols].apply(lambda x: col_map.get(tuple(x)), axis=1)
        else:
            final_result = combined_result
        
        logger.info(f"Chunked processing complete: {len(final_result):,} final {feature_type} features")
        return final_result

# Global chunked processor instance
chunked_processor = ChunkedDataProcessor(
    chunk_size=CONFIG.get("chunk_size", 50_000),
    memory_limit_mb=CONFIG.get("chunk_memory_limit_mb", 300)
)

def process_dataset_with_chunking(name: str, df: pd.DataFrame, 
                                processing_type: str = 'hashtags', 
                                label: str = '6h') -> Optional[pd.DataFrame]:
    """
    Process a dataset using chunked approach if it's large enough.
    
    Args:
        name: Dataset name for logging
        df: DataFrame to process
        processing_type: Type of processing ('hashtags', 'keywords', 'emerging_terms')
        label: Time label for aggregation
    
    Returns:
        Processed results
    """
    if df.empty:
        return None
    
    # Determine if chunking is beneficial
    dataset_size_mb = get_dataset_size_mb(df)
    threshold_rows = CONFIG.get("chunking_threshold_rows", 100_000)
    should_chunk = (len(df) > threshold_rows) or (dataset_size_mb > 200)  # Configurable threshold
    
    if not should_chunk:
        # Use original functions for smaller datasets
        if processing_type == 'hashtags':
            return aggregate_hashtags_dataset_aware(name, df, label)
        elif processing_type == 'keywords':
            return aggregate_keywords_dataset_aware(name, df, label)
        else:
            logger.warning(f"Unsupported processing type for small dataset: {processing_type}")
            return None
    
    # Use chunked processing for large datasets
    logger.info(f"Using chunked processing for {name}: {len(df):,} rows, {dataset_size_mb:.1f}MB")
    
    if processing_type == 'hashtags':
        return chunked_processor.process_dataset_chunked(
            df, chunked_processor.process_chunk_hashtags, label, 'hashtags'
        )
    elif processing_type == 'keywords':
        return chunked_processor.process_dataset_chunked(
            df, chunked_processor.process_chunk_keywords, label, 'keywords'
        )
    elif processing_type == 'emerging_terms':
        return chunked_processor.process_dataset_chunked(
            df, chunked_processor.process_chunk_emerging_terms, label, 'emerging_terms'
        )
    else:
        logger.error(f"Unsupported processing type: {processing_type}")
        return None

def aggregate_emerging_terms_chunked(dfs: List[Tuple[str, pd.DataFrame]], 
                                   label: str = '6h',
                                   min_growth_rate: float = 2.0) -> pd.DataFrame:
    """
    Enhanced emerging terms detection with chunked processing for large datasets.
    
    Args:
        dfs: List of (name, dataframe) tuples to process
        label: Time label for aggregation
        min_growth_rate: Minimum growth rate to consider a term emerging
    
    Returns:
        DataFrame with emerging trends, using chunked processing for large datasets
    """
    min_frequency = _MIN_FREQ_BY_LABEL.get(label, 3)
    
    logger.info(f"Computing chunked emerging terms for {label} with {len(dfs)} sources")
    
    # Process each source with chunked approach if needed
    source_results = []
    for name, df in dfs:
        if CONFIG["sample_rows_per_source"] and len(df) > CONFIG["sample_rows_per_source"]:
            df = sample_large_dataset(df, CONFIG["sample_rows_per_source"])
        
        # Use chunked processing for large datasets
        if CONFIG.get("enable_chunking", True):
            result = process_dataset_with_chunking(name, df, 'emerging_terms', label)
        else:
            # Fallback to original method for smaller datasets or when chunking disabled
            p = prepare_text_df_optimized(df)
            if p is None or p.empty:
                continue
            
            dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
            
            if dataset_type == 'videos':
                p["all_terms"] = p["text_clean"].apply(
                    lambda x: extract_all_terms_optimized(x, max_terms=25)
                )
            else:
                p["all_terms"] = p["text_clean"].apply(
                    lambda x: extract_all_terms_optimized(x, max_terms=15)
                )
            
            if label in _FREQ_MAP:
                p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
            else:
                p['bin'] = assign_time_bin_optimized(p['ts'], label)
            
            pe = p[["bin", "all_terms"]].explode("all_terms").dropna(subset=["all_terms"])
            if pe.empty:
                continue
            
            pe = pe.rename(columns={"all_terms": "feature"})
            
            if 'engagement_score' in p.columns:
                engagement_map = p.groupby('bin')['engagement_score'].mean().to_dict()
                pe['weight'] = pe['bin'].map(engagement_map).fillna(1.0)
                if dataset_type == 'videos':
                    pe['weight'] = np.log1p(pe['weight'])
            else:
                pe['weight'] = 1.0
            
            result = pe.groupby(["bin", "feature"], as_index=False)['weight'].sum().rename(columns={'weight': 'count'})
            result['count'] = result['count'].round().astype(int).clip(lower=1)
            result['source_type'] = dataset_type
        
        if result is not None and not result.empty:
            source_results.append(result)
    
    if not source_results:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging", "velocity", "category", "source_type"])
    
    # Combine all results
    allg = pd.concat(source_results, ignore_index=True)
    allg = allg.groupby(["bin", "feature", "source_type"], as_index=False)["count"].sum()
    
    # Filter rare terms early for performance
    total_counts = allg.groupby(["feature", "source_type"])["count"].sum()
    frequent_terms = total_counts[total_counts >= min_frequency].index
    
    # Convert to set for faster lookup
    frequent_set = set((feat, src) for feat, src in frequent_terms)
    allg = allg[allg.apply(lambda row: (row['feature'], row['source_type']) in frequent_set, axis=1)]
    
    # Apply feature limits for performance
    if CONFIG["max_features_emerging"] and len(allg['feature'].unique()) > CONFIG["max_features_emerging"]:
        top_features = allg.groupby('feature')['count'].sum().nlargest(CONFIG["max_features_emerging"]).index
        allg = allg[allg['feature'].isin(top_features)]
        logger.info(f"Limited to top {len(top_features)} emerging terms for performance")
    
    if allg.empty:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging", "velocity", "category", "source_type"])
    
    # Optimized growth rate calculation by source type
    allg = allg.sort_values(["feature", "source_type", "bin"]).reset_index(drop=True)
    allg['prev_count'] = allg.groupby(['feature', 'source_type'])['count'].shift(1)
    allg['growth_rate'] = allg['count'] / (allg['prev_count'] + 1e-6)
    allg.loc[allg['prev_count'].isna(), 'growth_rate'] = 1.0
    
    # Identify emerging terms with dataset-aware thresholds
    video_mask = allg['source_type'] == 'videos'
    comment_mask = allg['source_type'] == 'comments'
    
    allg.loc[video_mask, "is_emerging"] = allg.loc[video_mask, "growth_rate"] >= (min_growth_rate * 1.2)
    allg.loc[comment_mask, "is_emerging"] = allg.loc[comment_mask, "growth_rate"] >= min_growth_rate
    
    allg["category"] = allg["feature"].apply(categorize_feature)
    
    # Apply relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair
    logger.info("Applying chunked relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair trends")
    allg = filter_relevant_emerging_terms(allg)
    
    # Velocity calculation (expensive, controlled by config)
    if CONFIG["compute_velocity"] and label == '6h':
        logger.info("Computing chunked velocities for emerging terms")
        velocities = {}
        emerging_features = allg[allg["is_emerging"]]["feature"].unique()
        
        for term in tqdm(emerging_features, desc="Computing velocities"):
            term_data = allg[allg['feature'] == term]
            velocity = calculate_term_velocity_optimized(term_data, term)
            velocities[term] = velocity
        
        allg["velocity"] = allg["feature"].map(velocities).fillna(0.0)
    else:
        allg["velocity"] = 0.0
    
    return allg

# -----------------------------
# Enhanced Data Preparation
# -----------------------------

def prepare_text_df_optimized(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Optimized text DataFrame preparation for standardized comment/video datasets."""
    if df.empty:
        return None
        
    # Use standardized columns from preprocessing
    if 'timestamp' not in df.columns or 'text' not in df.columns:
        logger.warning("Dataset missing standardized 'timestamp' or 'text' columns")
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

# -----------------------------
# Parallel Processing Functions
# -----------------------------

def process_source_chunk(args: Tuple) -> Optional[pd.DataFrame]:
    """Process a single data source chunk for parallel processing."""
    name, df, processing_func = args
    try:
        return processing_func(name, df)
    except Exception as e:
        logger.error(f"Error processing chunk {name}: {e}")
        return None

def parallel_process_sources(sources: List[Tuple[str, pd.DataFrame]], 
                           processing_func: callable,
                           max_workers: Optional[int] = None) -> List[pd.DataFrame]:
    """Process sources in parallel when beneficial."""
    if not CONFIG["enable_parallel"] or len(sources) < 2:
        # Fall back to sequential processing
        results = []
        for name, df in sources:
            result = processing_func(name, df)
            if result is not None:
                results.append(result)
        return results
    
    max_workers = max_workers or min(MAX_WORKERS, len(sources))
    results = []
    
    logger.info(f"Processing {len(sources)} sources with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for parallel processing
        args_list = [(name, df, lambda n, d: processing_func(n, d)) for name, df in sources]
        
        # Submit all tasks
        future_to_source = {executor.submit(process_source_chunk, args): args[0] for args in args_list}
        
        # Collect results
        for future in tqdm(future_to_source, desc="Parallel processing"):
            try:
                result = future.result(timeout=60)  # 1 minute timeout per chunk
                if result is not None:
                    results.append(result)
            except Exception as e:
                source_name = future_to_source[future]
                logger.error(f"Failed to process source {source_name}: {e}")
    
    return results

# -----------------------------
# Dataset-Specific Aggregation Functions
# -----------------------------

def aggregate_hashtags_dataset_aware(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Dataset-aware hashtag aggregation optimized for comments vs videos."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
    
    # Efficient time binning
    if label in _FREQ_MAP:
        p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
    else:
        p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    # Explode hashtags efficiently
    pe = p[['bin', 'hashtags']].explode('hashtags').dropna(subset=['hashtags'])
    if pe.empty:
        return None
    
    pe = pe.rename(columns={'hashtags': 'feature'})
    
    # Add engagement weighting if available
    if 'engagement_score' in p.columns:
        # Merge engagement scores back
        engagement_map = p.groupby('bin')['engagement_score'].mean().to_dict()
        pe['engagement_weight'] = pe['bin'].map(engagement_map).fillna(1.0)
        
        # Apply engagement weighting (comments get linear weight, videos get log weight)
        if dataset_type == 'comments':
            pe['weighted_count'] = pe['engagement_weight'].apply(lambda x: min(x / 10, 5))  # Cap at 5x
        else:  # videos
            pe['weighted_count'] = pe['engagement_weight'].apply(lambda x: np.log1p(x / 1000))  # Log scale for views
    else:
        pe['weighted_count'] = 1.0
    
    # Group and aggregate with weighting
    g = pe.groupby(['bin', 'feature'], as_index=False).agg({
        'weighted_count': 'sum'
    }).rename(columns={'weighted_count': 'count'})
    
    # Round counts and ensure minimum of 1
    g['count'] = g['count'].round().astype(int).clip(lower=1)
    
    g['category'] = g['feature'].apply(categorize_feature)
    g['source_type'] = dataset_type
    
    return g

def aggregate_keywords_dataset_aware(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Dataset-aware keyword aggregation optimized for comments vs videos."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
    
    # Pre-compile patterns for better performance
    kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
    
    if label in _FREQ_MAP:
        p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
    else:
        p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    frames = []
    for kw, pat in kw_patterns:
        mask = p['text_clean'].str.contains(pat, na=False)
        if not mask.any():
            continue
        
        matched_data = p.loc[mask].copy()
        
        # Apply engagement weighting for dataset-specific aggregation
        if 'engagement_score' in matched_data.columns:
            if dataset_type == 'comments':
                # For comments, weight by like count (linear)
                matched_data['weight'] = matched_data['engagement_score'].apply(lambda x: min(x / 5, 10))
            else:
                # For videos, weight by comprehensive engagement (log scale)
                matched_data['weight'] = matched_data['engagement_score'].apply(lambda x: np.log1p(x / 1000))
        else:
            matched_data['weight'] = 1.0
        
        g = matched_data.groupby('bin', as_index=False)['weight'].sum().rename(columns={'weight': 'count'})
        g['count'] = g['count'].round().astype(int).clip(lower=1)
        g['feature'] = kw
        g['category'] = categorize_feature(kw)
        g['source_type'] = dataset_type
        frames.append(g)
    
    if not frames:
        return None
    
    return pd.concat(frames, ignore_index=True)

def aggregate_emerging_terms_dataset_aware(dfs: List[Tuple[str, pd.DataFrame]], 
                                         label: str = '6h',
                                         min_growth_rate: float = 2.0) -> pd.DataFrame:
    """Dataset-aware emerging terms detection with separate processing for comments and videos."""
    
    min_frequency = _MIN_FREQ_BY_LABEL.get(label, 3)
    
    logger.info(f"Computing dataset-aware emerging terms for {label} with {len(dfs)} sources")
    
    def process_emerging_source_aware(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if CONFIG["sample_rows_per_source"] and len(df) > CONFIG["sample_rows_per_source"]:
            df = sample_large_dataset(df, CONFIG["sample_rows_per_source"])
        
        p = prepare_text_df_optimized(df)
        if p is None or p.empty:
            return None
        
        dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
        
        # Extract terms with limits for performance, adjusted by dataset type
        if dataset_type == 'videos':
            # Videos have richer content, extract more terms
            p["all_terms"] = p["text_clean"].apply(
                lambda x: extract_all_terms_optimized(x, max_terms=40)
            )
        else:
            # Comments are typically shorter
            p["all_terms"] = p["text_clean"].apply(
                lambda x: extract_all_terms_optimized(x, max_terms=20)
            )
        
        if label in _FREQ_MAP:
            p["bin"] = p["ts"].dt.floor(_FREQ_MAP[label])
        else:
            p["bin"] = assign_time_bin_optimized(p["ts"], label)
        
        # Explode and aggregate with engagement weighting
        pe = p[["bin", "all_terms"]].explode("all_terms").dropna(subset=["all_terms"])
        if pe.empty:
            return None
        
        pe = pe.rename(columns={"all_terms": "feature"})
        
        # Add engagement context back
        if 'engagement_score' in p.columns:
            engagement_map = p.set_index('bin')['engagement_score'].to_dict()
            pe['engagement'] = pe['bin'].map(engagement_map).fillna(1.0)
            
            # Weight by engagement differently for each dataset type
            if dataset_type == 'comments':
                pe['weight'] = pe['engagement'].apply(lambda x: min(np.sqrt(x), 5))
            else:  # videos
                pe['weight'] = pe['engagement'].apply(lambda x: np.log1p(x / 100))
        else:
            pe['weight'] = 1.0
        
        result = pe.groupby(["bin", "feature"], as_index=False)['weight'].sum().rename(columns={'weight': 'count'})
        result['count'] = result['count'].round().astype(int).clip(lower=1)
        result['source_type'] = dataset_type
        
        return result
    
    # Process sources (potentially in parallel)
    frames = parallel_process_sources(dfs, process_emerging_source_aware)
    
    if not frames:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging", "velocity", "category", "source_type"])
    
    # Combine all data
    allg = pd.concat(frames, ignore_index=True)
    
    # Aggregate by source type to preserve dataset characteristics
    allg = allg.groupby(["bin", "feature", "source_type"], as_index=False)["count"].sum()
    
    # Filter rare terms early for performance
    total_counts = allg.groupby(["feature", "source_type"])["count"].sum()
    frequent_terms = total_counts[total_counts >= min_frequency].index
    
    # Convert to set for faster lookup
    frequent_set = set((feat, src) for feat, src in frequent_terms)
    allg = allg[allg.apply(lambda row: (row['feature'], row['source_type']) in frequent_set, axis=1)]
    
    # Apply feature limits for performance
    if CONFIG["max_features_emerging"] and len(allg['feature'].unique()) > CONFIG["max_features_emerging"]:
        top_features = allg.groupby('feature')['count'].sum().nlargest(CONFIG["max_features_emerging"]).index
        allg = allg[allg['feature'].isin(top_features)]
        logger.info(f"Limited to top {len(top_features)} emerging terms for performance")
    
    if allg.empty:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging", "velocity", "category", "source_type"])
    
    # Optimized growth rate calculation by source type
    allg = allg.sort_values(["feature", "source_type", "bin"]).reset_index(drop=True)
    allg['prev_count'] = allg.groupby(['feature', 'source_type'])['count'].shift(1)
    allg['growth_rate'] = allg['count'] / (allg['prev_count'] + 1e-6)
    allg.loc[allg['prev_count'].isna(), 'growth_rate'] = 1.0
    
    # Identify emerging terms with dataset-aware thresholds
    # Videos typically have more stable trends, so use slightly higher thresholds
    video_mask = allg['source_type'] == 'videos'
    comment_mask = allg['source_type'] == 'comments'
    
    allg.loc[video_mask, "is_emerging"] = allg.loc[video_mask, "growth_rate"] >= (min_growth_rate * 1.2)
    allg.loc[comment_mask, "is_emerging"] = allg.loc[comment_mask, "growth_rate"] >= min_growth_rate
    
    allg["category"] = allg["feature"].apply(categorize_feature)
    
    # Apply relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair
    logger.info("Applying dataset-aware relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair trends")
    allg = filter_relevant_emerging_terms(allg)
    
    # Velocity calculation (expensive, controlled by config)
    if CONFIG["compute_velocity"] and label == '6h':
        logger.info("Computing dataset-aware velocities for emerging terms")
        velocities = {}
        emerging_features = allg[allg["is_emerging"]]["feature"].unique()
        
        for term in tqdm(emerging_features, desc="Computing velocities"):
            velocities[term] = calculate_term_velocity_optimized(allg, term)
        
        allg["velocity"] = allg["feature"].map(velocities).fillna(0.0)
    else:
        allg["velocity"] = 0.0
    
    return allg

# -----------------------------
# Optimized Aggregation Functions
# -----------------------------

def aggregate_hashtags_optimized(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Optimized hashtag aggregation for a single source."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    # Efficient time binning
    if label in _FREQ_MAP:
        p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
    else:
        p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    # Explode hashtags efficiently
    pe = p[['bin', 'hashtags']].explode('hashtags').dropna(subset=['hashtags'])
    if pe.empty:
        return None
    
    pe = pe.rename(columns={'hashtags': 'feature'})
    
    # Group and aggregate
    g = pe.groupby(['bin', 'feature'], as_index=False).size().rename(columns={'size': 'count'})
    g['category'] = g['feature'].apply(categorize_feature)
    
    return g

def aggregate_keywords_optimized(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Optimized keyword aggregation for a single source."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    # Pre-compile patterns for better performance
    kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
    
    if label in _FREQ_MAP:
        p['bin'] = p['ts'].dt.floor(_FREQ_MAP[label])
    else:
        p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    frames = []
    for kw, pat in kw_patterns:
        mask = p['text_clean'].str.contains(pat, na=False)
        if not mask.any():
            continue
        
        g = p.loc[mask].groupby('bin', as_index=False).size().rename(columns={'size': 'count'})
        g['feature'] = kw
        g['category'] = categorize_feature(kw)
        frames.append(g)
    
    if not frames:
        return None
    
    return pd.concat(frames, ignore_index=True)

def assign_time_bin_optimized(ts: pd.Series, label: str) -> pd.Series:
    """Optimized time bin assignment."""
    if label in _FREQ_MAP:
        return ts.dt.floor(_FREQ_MAP[label])
    elif label == '1m':
        return ts.dt.to_period('M').dt.to_timestamp()
    elif label == '3m':
        return ts.dt.to_period('Q').dt.to_timestamp()
    elif label == '6m':
        years = ts.dt.year
        months = ts.dt.month
        start_month = np.where(months <= 6, 1, 7)
        return pd.to_datetime({'year': years, 'month': start_month, 'day': 1})
    else:
        raise ValueError(f"Unsupported timeframe label: {label}")

def add_rolling_stats_optimized(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Optimized rolling statistics calculation."""
    if df.empty:
        return df
    
    window = _ROLLING_WINDOW.get(label, 4)
    df = df.sort_values(["feature", "bin"]).reset_index(drop=True)
    
    # Vectorized rolling calculation
    df["rolling_mean_24h"] = (
        df.groupby("feature")["count"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    df["delta_vs_mean"] = df["count"] - df["rolling_mean_24h"]
    
    return df

# -----------------------------
# Enhanced Emerging Terms Detection
# -----------------------------

def aggregate_emerging_terms_optimized(dfs: List[Tuple[str, pd.DataFrame]], 
                                     label: str = '6h',
                                     min_growth_rate: float = 2.0) -> pd.DataFrame:
    """Optimized emerging terms detection with caching and performance improvements."""
    
    min_frequency = _MIN_FREQ_BY_LABEL.get(label, 3)
    
    # Create cache key
    data_fingerprint = f"{len(dfs)}_{label}_{min_frequency}_{min_growth_rate}"
    cache_key = f"emerging_terms_{data_fingerprint}"
    
    # Check cache first
    # if CONFIG["enable_caching"]:
    #     cached_result = cache.get(cache_key, data_fingerprint)
    #     if cached_result is not None:
    #         logger.info(f"Using cached emerging terms for {label}")
    #         return cached_result
    
    logger.info(f"Computing emerging terms for {label} with {len(dfs)} sources")
    
    def process_emerging_source(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if CONFIG["sample_rows_per_source"] and len(df) > CONFIG["sample_rows_per_source"]:
            df = sample_large_dataset(df, CONFIG["sample_rows_per_source"])
        
        p = prepare_text_df_optimized(df)
        if p is None or p.empty:
            return None
        
        # Extract terms with limits for performance
        p["all_terms"] = p["text_clean"].apply(
            lambda x: extract_all_terms_optimized(x, max_terms=20)
        )
        
        if label in _FREQ_MAP:
            p["bin"] = p["ts"].dt.floor(_FREQ_MAP[label])
        else:
            p["bin"] = assign_time_bin_optimized(p["ts"], label)
        
        # Explode and aggregate
        pe = p[["bin", "all_terms"]].explode("all_terms").dropna(subset=["all_terms"])
        if pe.empty:
            return None
        
        pe = pe.rename(columns={"all_terms": "feature"})
        return pe.groupby(["bin", "feature"], as_index=False).size().rename(columns={"size": "count"})
    
    # Process sources (potentially in parallel)
    frames = parallel_process_sources(dfs, process_emerging_source)
    
    if not frames:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging", "velocity", "category"])
    
    # Combine all data
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin", "feature"], as_index=False)["count"].sum()
    
    # Filter rare terms early for performance
    total_counts = allg.groupby("feature")["count"].sum()
    frequent_terms = total_counts[total_counts >= min_frequency].index
    
    # Apply feature limits for performance
    if CONFIG["max_features_emerging"] and len(frequent_terms) > CONFIG["max_features_emerging"]:
        top_terms = total_counts.nlargest(CONFIG["max_features_emerging"]).index
        frequent_terms = frequent_terms.intersection(top_terms)
        logger.info(f"Limited to top {len(frequent_terms)} emerging terms for performance")
    
    allg = allg[allg["feature"].isin(frequent_terms)]
    
    if allg.empty:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging", "velocity", "category"])
    
    # Optimized growth rate calculation
    allg = allg.sort_values(["feature", "bin"]).reset_index(drop=True)
    allg['prev_count'] = allg.groupby('feature')['count'].shift(1)
    allg['growth_rate'] = allg['count'] / (allg['prev_count'] + 1e-6)
    allg.loc[allg['prev_count'].isna(), 'growth_rate'] = 1.0
    
    # Identify emerging terms
    allg["is_emerging"] = allg["growth_rate"] >= min_growth_rate
    allg["category"] = allg["feature"].apply(categorize_feature)
    
    # Apply relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair
    logger.info("Applying relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair trends")
    allg = filter_relevant_emerging_terms(allg)
    
    # Velocity calculation (expensive, controlled by config)
    if CONFIG["compute_velocity"] and label == '6h':
        logger.info("Computing velocities for emerging terms")
        velocities = {}
        emerging_features = allg[allg["is_emerging"]]["feature"].unique()
        
        for term in tqdm(emerging_features, desc="Computing velocities"):
            velocities[term] = calculate_term_velocity_optimized(allg, term)
        
        allg["velocity"] = allg["feature"].map(velocities).fillna(0.0)
    else:
        allg["velocity"] = 0.0
    
    # Cache results
    if CONFIG["enable_caching"]:
        cache.set(cache_key, allg, data_fingerprint)
    
    return allg

def calculate_term_velocity_optimized(term_counts: pd.DataFrame, term: str, window_hours: int = 24) -> float:
    """Optimized velocity calculation."""
    term_data = term_counts[term_counts['feature'] == term].sort_values('bin')
    
    if len(term_data) < 2:
        return 0.0
    
    # Use only recent data for performance
    recent_data = term_data.tail(min(10, len(term_data)))
    
    if len(recent_data) < 2:
        return 0.0
    
    # Simple linear regression
    x = np.arange(len(recent_data))
    y = recent_data['count'].values
    
    if np.std(y) == 0:
        return 0.0
    
    try:
        slope, _, _, _, _ = stats.linregress(x, y)
        return max(0, slope)  # Only positive velocities
    except Exception:
        return 0.0

# -----------------------------
# Enhanced Phase 2 Reporting with ALL Timeframes
# -----------------------------

def write_enhanced_phase2_report(all_timeframe_data: Dict[str, pd.DataFrame]):
    """Enhanced Phase 2 report using ALL timeframe aggregation data passed in (no longer just 6h baseline)."""
    logger.info("Generating enhanced Phase 2 report from provided multi-timeframe data")

    path = INTERIM_DIR / 'phase2_enhanced_features_report.md'
    lines = ["# Enhanced Phase 2 Feature Engineering Report", ""]
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Performance Mode: {PERFORMANCE_MODE}")
    lines.append("")

    # Build unified summary from all provided dataframes
    lines.append("## Overall Summary (All Timeframes)")
    summary_rows = []
    for ftype, df in all_timeframe_data.items():
        if df is None or df.empty:
            summary_rows.append({
                'Feature_Type': ftype,
                'Total_Rows': 0,
                'Unique_Features': 0,
                'Timeframes': 0
            })
            continue
        tf_count = df['timeframe'].nunique() if 'timeframe' in df.columns else 1
        summary_rows.append({
            'Feature_Type': ftype,
            'Total_Rows': len(df),
            'Unique_Features': df['feature'].nunique() if 'feature' in df.columns else 0,
            'Timeframes': tf_count
        })
    if summary_rows:
        try:
            lines.append(pd.DataFrame(summary_rows).to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render summary table: {e})")
    lines.append("")

    # Timeframe comparison across all feature types (if timeframe column present)
    combined_for_timeframe = []
    for ftype, df in all_timeframe_data.items():
        if df is not None and not df.empty and 'timeframe' in df.columns:
            tmp = df[['timeframe','feature','count']].copy()
            tmp['feature_type'] = ftype
            combined_for_timeframe.append(tmp)
    if combined_for_timeframe:
        lines.append("## Timeframe Comparison (All Feature Types)")
        combo = pd.concat(combined_for_timeframe, ignore_index=True)
        timeframe_summary = combo.groupby('timeframe').agg(
            Rows=('feature','count'),
            Unique_Features=('feature','nunique'),
            Total_Count=('count','sum')
        ).reset_index().sort_values('timeframe')
        try:
            lines.append(timeframe_summary.to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render timeframe comparison: {e})")
        lines.append("")

    # Top performers per timeframe (merge all feature types)
    lines.append("## Top Performers by Timeframe")
    if combined_for_timeframe:
        # We need delta_vs_mean; ensure present by recomputing if missing
        for tf in TIMEFRAME_LABELS:
            lines.append(f"### {tf.upper()} Timeframe")
            tf_frames = []
            for ftype, df in all_timeframe_data.items():
                if df is None or df.empty:
                    continue
                if 'timeframe' not in df.columns:
                    continue
                subset = df[df['timeframe'] == tf]
                if subset.empty:
                    continue
                # Ensure rolling_mean_24h & delta_vs_mean
                if 'rolling_mean_24h' not in subset.columns or 'delta_vs_mean' not in subset.columns:
                    subset = subset.sort_values(['feature','bin']) if 'bin' in subset.columns else subset
                    if 'feature' in subset.columns and 'count' in subset.columns and 'bin' in subset.columns:
                        subset['rolling_mean_24h'] = subset.groupby('feature')['count'].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
                        subset['delta_vs_mean'] = subset['count'] - subset['rolling_mean_24h']
                subset['source_type'] = ftype
                tf_frames.append(subset)
            if not tf_frames:
                lines.append("No data available for this timeframe.")
                lines.append("")
                continue
            merged_tf = pd.concat(tf_frames, ignore_index=True)
            if 'bin' in merged_tf.columns and not merged_tf['bin'].empty:
                latest_bin = merged_tf['bin'].max()
                latest_slice = merged_tf[merged_tf['bin'] == latest_bin]
            else:
                latest_slice = merged_tf
                latest_bin = 'N/A'
            if latest_slice.empty:
                lines.append("No latest period data.")
                lines.append("")
                continue
            top = latest_slice.sort_values('delta_vs_mean', ascending=False).head(15) if 'delta_vs_mean' in latest_slice.columns else latest_slice.head(15)
            lines.append(f"Latest period: {latest_bin}")
            lines.append(f"Total features (latest): {len(latest_slice):,}")
            try:
                cols = ['source_type','feature','count','rolling_mean_24h','delta_vs_mean','category']
                avail = [c for c in cols if c in top.columns]
                lines.append(top[avail].to_markdown(index=False))
            except Exception as e:
                lines.append(f"(Could not render top table: {e})")
            lines.append("")

    # Performance insights
    lines.append("## Performance Insights")
    lines.append(f"- Processing mode: {PERFORMANCE_MODE}")
    lines.append(f"- Parallel processing: {'Enabled' if CONFIG['enable_parallel'] else 'Disabled'}")
    lines.append(f"- Caching: {'Enabled' if CONFIG['enable_caching'] else 'Disabled'}")
    lines.append(f"- Sample limit per source: {CONFIG['sample_rows_per_source']:,}" if CONFIG['sample_rows_per_source'] else "- No sampling applied")
    lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logger.info(f"Enhanced Phase 2 report written to {path} (multi-timeframe input)")

# -----------------------------
# Phase 3 Emerging Trends Reporting
# -----------------------------

def write_phase3_emerging_trends_report(emerging_df: pd.DataFrame,
                                        hashtags_6h: pd.DataFrame,
                                        keywords_6h: pd.DataFrame) -> None:
    """Generate Phase 3 emerging trends report. Supports single or multi-timeframe input.

    If 'timeframe' column exists, produce multi-timeframe summary & per-timeframe sections.
    """
    path = INTERIM_DIR / 'phase3_emerging_trends_report.md'
    lines: List[str] = ["# Phase 3 Emerging Trends Report", ""]
    lines.append(f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

    if emerging_df.empty:
        lines.append("No emerging term data available.")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"Phase 3 emerging trends report written to {path}")
        return

    multi_timeframe = 'timeframe' in emerging_df.columns

    # Executive summary
    lines.append("## Executive Summary")
    total_terms = emerging_df['feature'].nunique()
    lines.append(f"- Total unique tracked terms: {total_terms:,}")
    if 'is_emerging' in emerging_df.columns:
        current_mask_df = emerging_df.copy()
        # For multi-timeframe, consider latest bin per timeframe for 'current emerging'
        if multi_timeframe:
            latest_per_tf = emerging_df.groupby('timeframe')['bin'].transform('max')
            current_mask_df = emerging_df[emerging_df['bin'] == latest_per_tf]
        else:
            latest_bin_global = emerging_df['bin'].max()
            current_mask_df = emerging_df[emerging_df['bin'] == latest_bin_global]
        emerg_current = current_mask_df[current_mask_df['is_emerging'] == True]
        lines.append(f"- Emerging terms in latest window(s): {emerg_current['feature'].nunique():,}")
        if 'growth_rate' in emerg_current.columns and not emerg_current.empty:
            lines.append(f"- Median growth rate (latest emerging): {emerg_current['growth_rate'].median():.2f}x")
            lines.append(f"- Max growth rate (latest emerging): {emerg_current['growth_rate'].max():.2f}x")
    if 'velocity' in emerging_df.columns and emerging_df['velocity'].abs().sum() > 0:
        lines.append(f"- Mean positive velocity: {emerging_df[emerging_df['velocity']>0]['velocity'].mean():.2f}")
    if multi_timeframe:
        lines.append(f"- Timeframes covered: {emerging_df['timeframe'].nunique()}")
    lines.append("")

    if multi_timeframe:
        # Per-timeframe summary table
        lines.append("### Per-Timeframe Emerging Summary (latest window each)")
        summ_rows = []
        for tf in sorted(emerging_df['timeframe'].unique(), key=lambda x: TIMEFRAME_LABELS.index(x) if x in TIMEFRAME_LABELS else 999):
            sub = emerging_df[emerging_df['timeframe']==tf]
            latest_bin = sub['bin'].max()
            sub_latest = sub[sub['bin']==latest_bin]
            emerg_latest = sub_latest[sub_latest.get('is_emerging', False)==True]
            top_term = None
            if not emerg_latest.empty:
                if 'growth_rate' in emerg_latest.columns:
                    top_term = emerg_latest.sort_values('growth_rate', ascending=False)['feature'].iloc[0]
                else:
                    top_term = emerg_latest.sort_values('count', ascending=False)['feature'].iloc[0]
            summ_rows.append({
                'timeframe': tf,
                'latest_bin': latest_bin,
                'emerging_terms': emerg_latest['feature'].nunique(),
                'median_growth': emerg_latest['growth_rate'].median() if 'growth_rate' in emerg_latest.columns and not emerg_latest.empty else np.nan,
                'top_term': top_term or '-'})
        try:
            lines.append(pd.DataFrame(summ_rows).to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render summary table: {e})")
        lines.append("")

        # Detailed per timeframe sections
        for tf in sorted(emerging_df['timeframe'].unique(), key=lambda x: TIMEFRAME_LABELS.index(x) if x in TIMEFRAME_LABELS else 999):
            sub = emerging_df[emerging_df['timeframe']==tf].copy()
            lines.append(f"## Timeframe: {tf}")
            latest_bin = sub['bin'].max()
            lines.append(f"Latest bin: {latest_bin}")
            sub_latest = sub[sub['bin']==latest_bin].copy()
            # NEW vs SUSTAINED classification within timeframe
            appearances_tf = sub.groupby('feature')['bin'].nunique()
            new_flags_tf = appearances_tf[appearances_tf <= 2].index
            sub_latest['trend_type'] = np.where(sub_latest['feature'].isin(new_flags_tf), 'NEW', 'SUSTAINED')
            # Top emerging
            if 'growth_rate' in sub_latest.columns:
                top_emerg = sub_latest.sort_values('growth_rate', ascending=False).head(15)
            else:
                top_emerg = sub_latest.sort_values('count', ascending=False).head(15)
            lines.append("### Top Emerging (Latest Window)")
            try:
                cols_pref = ['trend_type','feature','category','count','prev_count','growth_rate','velocity']
                cols = [c for c in cols_pref if c in top_emerg.columns]
                lines.append(top_emerg[cols].to_markdown(index=False))
            except Exception as e:
                lines.append(f"(Could not render top table: {e})")
            lines.append("")
            # Category distribution
            if 'category' in sub_latest.columns and not sub_latest.empty:
                lines.append("#### Category Distribution (Emerging Latest Bin)")
                cat_counts = sub_latest[sub_latest.get('is_emerging', False)==True]['category'].value_counts().reset_index()
                cat_counts.columns = ['category','emerging_terms']
                try:
                    lines.append(cat_counts.to_markdown(index=False))
                except Exception as e:
                    lines.append(f"(Could not render category distribution: {e})")
                lines.append("")
            # Persistence metrics within timeframe
            lines.append("#### Persistence")
            persistence = appearances_tf.describe()
            lines.append(f"- Avg bins per term: {persistence['mean']:.2f}")
            lines.append(f"- 75th percentile bins: {persistence['75%']:.0f}")
            if 'growth_rate' in sub.columns:
                grp = sub.sort_values(['feature','bin']).copy()
                grp['prev_growth'] = grp.groupby('feature')['growth_rate'].shift(1)
                latest_join = grp[grp['bin']==latest_bin]
                accel = latest_join[(latest_join['growth_rate'] > latest_join['prev_growth']) & (latest_join['prev_growth'].notna())]
                decel = latest_join[(latest_join['growth_rate'] < latest_join['prev_growth']) & (latest_join['prev_growth'].notna())]
                lines.append(f"- Accelerating: {len(accel)} | Decelerating: {len(decel)}")
            lines.append("")
    else:
        # Single timeframe legacy logic
        latest_bin = emerging_df['bin'].max()
        lines.append(f"## Latest Window: {latest_bin}")
        latest_slice = emerging_df[emerging_df['bin']==latest_bin].copy()
        appearances = emerging_df.groupby('feature')['bin'].nunique()
        new_flags = appearances[appearances <= 2].index
        latest_slice['trend_type'] = np.where(latest_slice['feature'].isin(new_flags), 'NEW', 'SUSTAINED')
        if 'growth_rate' in latest_slice.columns:
            top_growth = latest_slice.sort_values('growth_rate', ascending=False).head(25)
        else:
            top_growth = latest_slice.sort_values('count', ascending=False).head(25)
        lines.append("### Top Emerging Terms")
        try:
            cols_pref = ['trend_type','feature','category','count','prev_count','growth_rate','velocity']
            cols = [c for c in cols_pref if c in top_growth.columns]
            lines.append(top_growth[cols].to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render top emerging table: {e})")
        lines.append("")

    # Overlap with baseline (still useful context)
    lines.append("## Overlap With Baseline 6h Aggregates")
    def overlap_section(label: str, base_df: pd.DataFrame):
        if base_df is None or base_df.empty or emerging_df.empty:
            return f"- {label}: No data"
        # Use 6h subset if multi-timeframe present
        if 'timeframe' in emerging_df.columns:
            e6 = emerging_df[emerging_df['timeframe']=='6h']
            if e6.empty:
                return f"- {label}: No 6h emerging data"
            latest_bin_loc = e6['bin'].max()
            emerg_latest = set(e6[e6['bin']==latest_bin_loc]['feature'])
        else:
            latest_bin_loc = emerging_df['bin'].max()
            emerg_latest = set(emerging_df[emerging_df['bin']==latest_bin_loc]['feature'])
        base_latest = set(base_df[base_df['bin']==base_df['bin'].max()]['feature']) if 'bin' in base_df.columns else set()
        inter = emerg_latest & base_latest
        return (f"- {label}: overlap {len(inter)} ({len(inter)/max(1,len(emerg_latest))*100:.1f}% of latest emerging)")
    lines.append(overlap_section('Hashtags 6h', hashtags_6h))
    lines.append(overlap_section('Keywords 6h', keywords_6h))
    lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Phase 3 emerging trends report written to {path}")

# -----------------------------
# Main Optimized Pipeline
# -----------------------------

def main_optimized():
    """Main optimized data processing pipeline with dataset-aware processing."""
    start_time = time.time()
    logger.info(f"Starting dataset-aware optimized data processing pipeline in {PERFORMANCE_MODE} mode")
    
    # Load data with optimization and dataset type separation
    samples = load_samples_optimized()
    if not samples:
        logger.error("No data samples found")
        return
    
    # Separate comment and video sources for optimized processing
    comment_sources = []
    video_sources = []
    
    for name, df in samples.items():
        if 'dataset_type' in df.columns:
            if df['dataset_type'].iloc[0] == 'comments':
                comment_sources.append((name, df))
            else:
                video_sources.append((name, df))
        else:
            # Fallback based on filename
            if any(tok in name.lower() for tok in ["comment"]):
                comment_sources.append((name, df))
            elif any(tok in name.lower() for tok in ["video"]):
                video_sources.append((name, df))
            else:
                # Default to comment processing for unknown types
                comment_sources.append((name, df))
    
    logger.info(f"Processing {len(comment_sources)} comment sources and {len(video_sources)} video sources")
    
    # Phase 2: Baseline 6h aggregations with dataset awareness and chunked processing
    logger.info("Phase 2: Running dataset-aware baseline 6h aggregations with chunked processing")
    
    def process_hashtags_6h_chunked(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process hashtags with automatic chunking for large datasets."""
        if CONFIG.get("enable_chunking", True):
            return process_dataset_with_chunking(name, df, 'hashtags', '6h')
        else:
            return aggregate_hashtags_dataset_aware(name, df, '6h')
    
    def process_keywords_6h_chunked(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process keywords with automatic chunking for large datasets."""
        if CONFIG.get("enable_chunking", True):
            return process_dataset_with_chunking(name, df, 'keywords', '6h')
        else:
            return aggregate_keywords_dataset_aware(name, df, '6h')
    
    # Process baseline aggregations for comments and videos separately
    all_sources = comment_sources + video_sources
    
    hashtag_frames = parallel_process_sources(all_sources, process_hashtags_6h_chunked)
    keyword_frames = parallel_process_sources(all_sources, process_keywords_6h_chunked)
    
    # Combine and process results
    ts_hashtags = pd.concat(hashtag_frames, ignore_index=True) if hashtag_frames else pd.DataFrame()
    ts_keywords = pd.concat(keyword_frames, ignore_index=True) if keyword_frames else pd.DataFrame()
    
    if not ts_hashtags.empty:
        # Group by bin, feature, category, and source_type to preserve dataset distinctions
        ts_hashtags = ts_hashtags.groupby(['bin', 'feature', 'category', 'source_type'], as_index=False)['count'].sum()
        ts_hashtags = add_rolling_stats_optimized(ts_hashtags, '6h')
    
    if not ts_keywords.empty:
        ts_keywords = ts_keywords.groupby(['bin', 'feature', 'category', 'source_type'], as_index=False)['count'].sum()
        ts_keywords = add_rolling_stats_optimized(ts_keywords, '6h')
    
    # Empty audio for sample data (not applicable to this dataset)
    ts_audio = pd.DataFrame(columns=['bin', 'feature', 'count', 'rolling_mean_24h', 'delta_vs_mean', 'category', 'source_type'])
    
    # Save baseline results with dataset type information
    if not ts_hashtags.empty:
        ts_hashtags.to_parquet(PROC_DIR / 'features_hashtags_6h.parquet', index=False)
        logger.info(f"Saved hashtags: {len(ts_hashtags):,} rows across {ts_hashtags['source_type'].nunique()} source types")
    
    if not ts_keywords.empty:
        ts_keywords.to_parquet(PROC_DIR / 'features_keywords_6h.parquet', index=False)
        logger.info(f"Saved keywords: {len(ts_keywords):,} rows across {ts_keywords['source_type'].nunique()} source types")
    
    # Multi-timeframe processing with dataset awareness and chunked processing
    logger.info("Phase 2: Running multi-timeframe dataset-aware aggregations with chunked processing")
    
    for label in TIMEFRAME_LABELS:
        if label == '6h':
            continue  # Already processed
        
        logger.info(f"Processing timeframe: {label}")
        
        def process_hashtags_tf_chunked(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
            """Process hashtags with automatic chunking for current timeframe."""
            if CONFIG.get("enable_chunking", True):
                return process_dataset_with_chunking(name, df, 'hashtags', label)
            else:
                return aggregate_hashtags_dataset_aware(name, df, label)
        
        def process_keywords_tf_chunked(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
            """Process keywords with automatic chunking for current timeframe."""
            if CONFIG.get("enable_chunking", True):
                return process_dataset_with_chunking(name, df, 'keywords', label)
            else:
                return aggregate_keywords_dataset_aware(name, df, label)
        
        # Check if files already exist
        h_file = f'features_hashtags_{label}.parquet'
        k_file = f'features_keywords_{label}.parquet'
        
        if not (PROC_DIR / h_file).exists():
            h_frames = parallel_process_sources(all_sources, process_hashtags_tf_chunked)
            if h_frames:
                h_combined = pd.concat(h_frames, ignore_index=True)
                h_combined = h_combined.groupby(['bin', 'feature', 'category', 'source_type'], as_index=False)['count'].sum()
                h_combined = add_rolling_stats_optimized(h_combined, label)
                h_combined.to_parquet(PROC_DIR / h_file, index=False)
                logger.info(f"Saved {h_file}: {len(h_combined):,} rows")
        
        if not (PROC_DIR / k_file).exists():
            k_frames = parallel_process_sources(all_sources, process_keywords_tf_chunked)
            if k_frames:
                k_combined = pd.concat(k_frames, ignore_index=True)
                k_combined = k_combined.groupby(['bin', 'feature', 'category', 'source_type'], as_index=False)['count'].sum()
                k_combined = add_rolling_stats_optimized(k_combined, label)
                k_combined.to_parquet(PROC_DIR / k_file, index=False)
                logger.info(f"Saved {k_file}: {len(k_combined):,} rows")
    
    # Build combined multi-timeframe datasets for report (include 6h inside each)
    def load_all_timeframes(feature_type: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for pth in PROC_DIR.glob(f'features_{feature_type}_*.parquet'):
            if 'emerging_terms' in pth.name or 'statistical_anomalies' in pth.name:
                continue
            try:
                dfp = pd.read_parquet(pth)
                if dfp.empty:
                    continue
                tf_label = pth.stem.split('_')[-1]
                dfp = dfp.copy()
                dfp['timeframe'] = tf_label
                frames.append(dfp)
            except Exception as e:
                logger.warning(f"Failed loading {pth.name} for report merge: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    all_timeframe_data = {
        'hashtags': load_all_timeframes('hashtags'),
        'keywords': load_all_timeframes('keywords'),
        'audio': load_all_timeframes('audio')  # may be empty
    }
    write_enhanced_phase2_report(all_timeframe_data)
    
    # Phase 3: Multi-timeframe emerging trends detection with chunked processing
    logger.info("Phase 3: Running multi-timeframe dataset-aware emerging trends detection with chunked processing")
    emerging_frames = []
    for label in TIMEFRAME_LABELS:
        # Use growth rate threshold per timeframe
        growth_thr = MIN_GROWTH_RATE_BY_LABEL.get(label, 2.0)
        fname = PROC_DIR / f'features_emerging_terms_{label}.parquet'
        if fname.exists():
            df_em = pd.read_parquet(fname)
            logger.info(f"Loaded existing emerging terms {label}: {len(df_em):,} rows")
        else:
            # Use chunked emerging terms processing for large datasets
            if CONFIG.get("enable_chunking", True):
                df_em = aggregate_emerging_terms_chunked(all_sources, label, min_growth_rate=growth_thr)
            else:
                df_em = aggregate_emerging_terms_dataset_aware(all_sources, label, min_growth_rate=growth_thr)
            
            if not df_em.empty:
                df_em.to_parquet(fname, index=False)
                logger.info(f"Saved emerging terms {label}: {len(df_em):,} rows (thr={growth_thr})")
        
        if not df_em.empty:
            df_em = df_em.copy()
            df_em['timeframe'] = label
            emerging_frames.append(df_em)
    
    if emerging_frames:
        ts_emerging_multi = pd.concat(emerging_frames, ignore_index=True)
    else:
        ts_emerging_multi = pd.DataFrame()

    # Write Phase 3 emerging trends report (multi timeframe and dataset aware)
    try:
        write_phase3_emerging_trends_report(ts_emerging_multi, ts_hashtags, ts_keywords)
    except Exception as e:
        logger.warning(f"Failed to write Phase 3 emerging trends report: {e}")
    
    # Performance summary with dataset insights
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("Dataset-aware pipeline completed successfully!")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    logger.info(f"Performance mode: {PERFORMANCE_MODE}")
    logger.info(f"Chunked processing: {'Enabled' if CONFIG.get('enable_chunking', True) else 'Disabled'}")
    logger.info(f"Chunk size: {CONFIG.get('chunk_size', 50_000):,} rows")
    logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    logger.info(f"Processed {len(comment_sources)} comment sources and {len(video_sources)} video sources")
    
    # Enhanced performance report with dataset breakdown and chunked processing info
    perf_report = {
        'execution_time_seconds': duration,
        'performance_mode': PERFORMANCE_MODE,
        'parallel_processing': CONFIG['enable_parallel'],
        'caching_enabled': CONFIG['enable_caching'],
        'chunked_processing_enabled': CONFIG.get('enable_chunking', True),
        'chunk_size': CONFIG.get('chunk_size', 50_000),
        'chunk_memory_limit_mb': CONFIG.get('chunk_memory_limit_mb', 300),
        'chunking_threshold_rows': CONFIG.get('chunking_threshold_rows', 100_000),
        'total_sources_processed': len(all_sources),
        'comment_sources': len(comment_sources),
        'video_sources': len(video_sources),
        'files_generated': len(list(PROC_DIR.glob('features_*.parquet'))),
        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(INTERIM_DIR / 'performance_report.json', 'w') as f:
        json.dump(perf_report, f, indent=2)
    
    logger.info(f"Performance report saved to {INTERIM_DIR / 'performance_report.json'}")

if __name__ == '__main__':
    main_optimized()