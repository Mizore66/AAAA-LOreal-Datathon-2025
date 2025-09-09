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
        "streaming_threshold_mb": 100
    },
    "BALANCED": {
        "max_features_emerging": 400,
        "max_features_anomaly": 400,
        "max_cluster_features": 250,
        "sample_rows_per_source": 100_000,
        "compute_velocity": True,
        "enable_parallel": True,
        "enable_caching": True,
        "streaming_threshold_mb": 200
    },
    "THOROUGH": {
        "max_features_emerging": None,
        "max_features_anomaly": None,
        "max_cluster_features": None,
        "sample_rows_per_source": None,
        "compute_velocity": True,
        "enable_parallel": False,
        "enable_caching": True,
        "streaming_threshold_mb": 500
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
    """Optimized data loading with memory management."""
    logger.info("Loading processed datasets with memory optimization")
    out: Dict[str, pd.DataFrame] = {}
    
    # Prefer full files over samples
    files = [p for p in PROC_DIR.glob('*.parquet') if not p.name.endswith('.sample.parquet')]
    if not files:
        files = list(PROC_DIR.glob('*.parquet'))
    
    logger.info(f"Found {len(files)} parquet files to process")
    
    total_memory_mb = 0
    for p in files:
        name = p.stem
        try:
            # Check file size before loading
            file_size_mb = p.stat().st_size / (1024 * 1024)
            
            df = pd.read_parquet(p)
            dataset_size_mb = get_dataset_size_mb(df)
            total_memory_mb += dataset_size_mb
            
            # Apply sampling if needed
            if CONFIG["sample_rows_per_source"] and len(df) > CONFIG["sample_rows_per_source"]:
                df = sample_large_dataset(df, CONFIG["sample_rows_per_source"])
                dataset_size_mb = get_dataset_size_mb(df)
            
            out[name] = df
            logger.info(f"Loaded {p.name}: {len(df):,} rows, {dataset_size_mb:.1f}MB")
            
            # Memory management
            if total_memory_mb > 500:  # 500MB threshold
                logger.warning("High memory usage detected, forcing garbage collection")
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to load {p}: {e}")
    
    logger.info(f"Total memory usage: {total_memory_mb:.1f}MB across {len(out)} datasets")
    return out

# -----------------------------
# Enhanced Data Preparation
# -----------------------------

def prepare_text_df_optimized(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Optimized text DataFrame preparation."""
    if df.empty:
        return None
        
    # Find timestamp column efficiently
    ts_candidates = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "created", "upload", "posted"])]
    ts_col = None
    
    for c in ts_candidates:
        try:
            pd.to_datetime(df[c].iloc[:100])  # Test with sample
            ts_col = c
            break
        except Exception:
            continue
    
    if not ts_col:
        return None
    
    # Find text columns efficiently
    txt_candidates = ["text", "comment", "caption", "title", "description", "body", "content"]
    txt_cols = [c for c in txt_candidates if c in df.columns]
    if not txt_cols:
        txt_cols = [c for c in df.columns if df[c].dtype == 'object'][:2]
    
    if not txt_cols:
        return None
    
    # Optimize column selection
    use = df[[ts_col] + txt_cols].copy()
    
    # Vectorized timestamp conversion
    use["ts"] = pd.to_datetime(use[ts_col], errors='coerce')
    use = use.dropna(subset=["ts"])
    
    if use.empty:
        return None
    
    # Optimized text processing
    use["text_raw"] = use[txt_cols].fillna('').astype(str).agg(" ".join, axis=1)
    
    # Batch text cleaning for better performance
    logger.info(f"Processing {len(use):,} text records")
    use["text_clean"] = use["text_raw"].apply(clean_text_optimized)
    use["hashtags"] = use["text_raw"].apply(extract_hashtags_optimized)
    
    return use[["ts", "text_clean", "hashtags"]]

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
    if CONFIG["enable_caching"]:
        cached_result = cache.get(cache_key, data_fingerprint)
        if cached_result is not None:
            logger.info(f"Using cached emerging terms for {label}")
            return cached_result
    
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
    """Main optimized data processing pipeline."""
    start_time = time.time()
    logger.info(f"Starting optimized data processing pipeline in {PERFORMANCE_MODE} mode")
    
    # Load data with optimization
    samples = load_samples_optimized()
    if not samples:
        logger.error("No data samples found")
        return
    
    # Choose text sources
    text_sources = []
    for name, df in samples.items():
        if any(tok in name.lower() for tok in ["comment", "video"]):
            text_sources.append((name, df))
    
    if not text_sources:
        text_sources = list(samples.items())
    
    logger.info(f"Processing {len(text_sources)} text sources")
    
    # Phase 2: Baseline 6h aggregations
    logger.info("Phase 2: Running baseline 6h aggregations")
    
    def process_hashtags_6h(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return aggregate_hashtags_optimized(name, df, '6h')
    
    def process_keywords_6h(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return aggregate_keywords_optimized(name, df, '6h')
    
    # Process baseline aggregations
    hashtag_frames = parallel_process_sources(text_sources, process_hashtags_6h)
    keyword_frames = parallel_process_sources(text_sources, process_keywords_6h)
    
    # Combine and process results
    ts_hashtags = pd.concat(hashtag_frames, ignore_index=True) if hashtag_frames else pd.DataFrame()
    ts_keywords = pd.concat(keyword_frames, ignore_index=True) if keyword_frames else pd.DataFrame()
    
    if not ts_hashtags.empty:
        ts_hashtags = ts_hashtags.groupby(['bin', 'feature', 'category'], as_index=False)['count'].sum()
        ts_hashtags = add_rolling_stats_optimized(ts_hashtags, '6h')
    
    if not ts_keywords.empty:
        ts_keywords = ts_keywords.groupby(['bin', 'feature', 'category'], as_index=False)['count'].sum()
        ts_keywords = add_rolling_stats_optimized(ts_keywords, '6h')
    
    # Empty audio for sample data
    ts_audio = pd.DataFrame(columns=['bin', 'feature', 'count', 'rolling_mean_24h', 'delta_vs_mean', 'category'])
    
    # Save baseline results
    if not ts_hashtags.empty:
        ts_hashtags.to_parquet(PROC_DIR / 'features_hashtags_6h.parquet', index=False)
        logger.info(f"Saved hashtags: {len(ts_hashtags):,} rows")
    
    if not ts_keywords.empty:
        ts_keywords.to_parquet(PROC_DIR / 'features_keywords_6h.parquet', index=False)
        logger.info(f"Saved keywords: {len(ts_keywords):,} rows")
    
    # Multi-timeframe processing
    logger.info("Phase 2: Running multi-timeframe aggregations")
    all_sources = list(samples.items())
    
    for label in TIMEFRAME_LABELS:
        if label == '6h':
            continue  # Already processed
        
        logger.info(f"Processing timeframe: {label}")
        
        def process_hashtags_tf(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
            return aggregate_hashtags_optimized(name, df, label)
        
        def process_keywords_tf(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
            return aggregate_keywords_optimized(name, df, label)
        
        # Check if files already exist
        h_file = f'features_hashtags_{label}.parquet'
        k_file = f'features_keywords_{label}.parquet'
        
        if not (PROC_DIR / h_file).exists():
            h_frames = parallel_process_sources(all_sources, process_hashtags_tf)
            if h_frames:
                h_combined = pd.concat(h_frames, ignore_index=True)
                h_combined = h_combined.groupby(['bin', 'feature', 'category'], as_index=False)['count'].sum()
                h_combined = add_rolling_stats_optimized(h_combined, label)
                h_combined.to_parquet(PROC_DIR / h_file, index=False)
                logger.info(f"Saved {h_file}: {len(h_combined):,} rows")
        
        if not (PROC_DIR / k_file).exists():
            k_frames = parallel_process_sources(all_sources, process_keywords_tf)
            if k_frames:
                k_combined = pd.concat(k_frames, ignore_index=True)
                k_combined = k_combined.groupby(['bin', 'feature', 'category'], as_index=False)['count'].sum()
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
    
    # Phase 3: Multi-timeframe emerging trends detection
    logger.info("Phase 3: Running multi-timeframe emerging trends detection")
    emerging_frames = []
    for label in TIMEFRAME_LABELS:
        # Use growth rate threshold per timeframe
        growth_thr = MIN_GROWTH_RATE_BY_LABEL.get(label, 2.0)
        fname = PROC_DIR / f'features_emerging_terms_{label}.parquet'
        if fname.exists():
            df_em = pd.read_parquet(fname)
            logger.info(f"Loaded existing emerging terms {label}: {len(df_em):,} rows")
        else:
            df_em = aggregate_emerging_terms_optimized(all_sources, label, min_growth_rate=growth_thr)
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

    # Write Phase 3 emerging trends report (multi timeframe aware)
    try:
        write_phase3_emerging_trends_report(ts_emerging_multi, ts_hashtags, ts_keywords)
    except Exception as e:
        logger.warning(f"Failed to write Phase 3 emerging trends report: {e}")
    
    # Performance summary
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    logger.info(f"Performance mode: {PERFORMANCE_MODE}")
    logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Performance report
    perf_report = {
        'execution_time_seconds': duration,
        'performance_mode': PERFORMANCE_MODE,
        'parallel_processing': CONFIG['enable_parallel'],
        'caching_enabled': CONFIG['enable_caching'],
        'total_sources_processed': len(all_sources),
        'files_generated': len(list(PROC_DIR.glob('features_*.parquet'))),
        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(INTERIM_DIR / 'performance_report.json', 'w') as f:
        json.dump(perf_report, f, indent=2)
    
    logger.info(f"Performance report saved to {INTERIM_DIR / 'performance_report.json'}")

if __name__ == '__main__':
    main_optimized()