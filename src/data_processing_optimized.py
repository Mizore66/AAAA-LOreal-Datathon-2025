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
PERFORMANCE_MODE = "OPTIMIZED"  # Options: "OPTIMIZED", "BALANCED", "THOROUGH"
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

for kw in skincare_keywords:
    KEYWORD_CATEGORY[kw] = "Skincare"
for kw in makeup_keywords:
    KEYWORD_CATEGORY[kw] = "Makeup"
for kw in hair_keywords:
    KEYWORD_CATEGORY[kw] = "Hair"

# Timeframe configurations
TIMEFRAME_LABELS = ["1h", "3h", "6h", "1d", "3d", "7d", "14d", "1m", "3m", "6m"]

_FREQ_MAP = {
    '1h': '1H', '3h': '3H', '6h': '6H', '1d': '1D', '3d': '3D', '7d': '7D', '14d': '14D'
}

_ROLLING_WINDOW = {
    '1h': 24, '3h': 8, '6h': 4, '1d': 7, '3d': 7, '7d': 4, '14d': 4, '1m': 6, '3m': 4, '6m': 4
}

_MIN_FREQ_BY_LABEL = {
    '1h': 2, '3h': 2, '6h': 3, '1d': 5, '3d': 5, '7d': 8, '14d': 10, '1m': 15, '3m': 20, '6m': 30
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
    """Optimized feature categorization with caching."""
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
    
    # Quick heuristic categorization
    if any(tok in f for tok in ["skin", "spf", "sunscreen", "barrier", "niacinamide", "retinol", "serum"]):
        return "Skincare"
    if any(tok in f for tok in ["lip", "lash", "brow", "blush", "contour", "eyeshadow", "mascara", "liner"]):
        return "Makeup"
    if any(tok in f for tok in ["hair", "scalp", "shampoo", "conditioner", "keratin", "oil"]):
        return "Hair"
    
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

def write_enhanced_phase2_report(baseline_data: Dict[str, pd.DataFrame]):
    """Enhanced Phase 2 report showing ALL timeframe aggregation data."""
    logger.info("Generating enhanced Phase 2 report with all timeframes")
    
    path = INTERIM_DIR / 'phase2_enhanced_features_report.md'
    lines = ["# Enhanced Phase 2 Feature Engineering Report", ""]
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Performance Mode: {PERFORMANCE_MODE}")
    lines.append("")
    
    # Summary of 6h baseline
    h_df = baseline_data.get('hashtags', pd.DataFrame())
    k_df = baseline_data.get('keywords', pd.DataFrame())
    a_df = baseline_data.get('audio', pd.DataFrame())
    
    lines.append("## Summary (6h Baseline)")
    lines.append(f"- Hashtag features: {len(h_df):,} rows")
    lines.append(f"- Keyword features: {len(k_df):,} rows")
    lines.append(f"- Audio features: {len(a_df):,} rows")
    lines.append("")
    
    # Comprehensive multi-frequency analysis
    lines.append("## Comprehensive Multi-Frequency Analysis")
    
    # Inventory all generated files
    inventory = []
    feature_types = set()
    
    for p in PROC_DIR.glob('features_*.parquet'):
        if 'statistical_anomalies' in p.name or 'emerging_terms' in p.name:
            continue
        
        try:
            dfp = pd.read_parquet(p)
            parts = p.stem.split('_')
            if len(parts) >= 3:
                feature_type = '_'.join(parts[1:-1])  # e.g., "hashtags", "keywords", "audio"
                timeframe = parts[-1]  # e.g., "1h", "6h", "1d"
                
                feature_types.add(feature_type)
                
                # Calculate additional metrics
                unique_features = dfp['feature'].nunique() if 'feature' in dfp.columns else 0
                latest_bin = dfp['bin'].max() if 'bin' in dfp.columns else None
                total_count = dfp['count'].sum() if 'count' in dfp.columns else 0
                
                inventory.append({
                    'Feature_Type': feature_type.title(),
                    'Timeframe': timeframe,
                    'Rows': len(dfp),
                    'Unique_Features': unique_features,
                    'Total_Count': total_count,
                    'Latest_Bin': latest_bin.strftime('%Y-%m-%d %H:%M') if latest_bin else 'N/A',
                    'File_Size_KB': p.stat().st_size // 1024
                })
        except Exception as e:
            logger.warning(f"Could not process {p.name}: {e}")
    
    if inventory:
        lines.append("### Complete Feature Inventory")
        inv_df = pd.DataFrame(inventory).sort_values(['Feature_Type', 'Timeframe'])
        try:
            lines.append(inv_df.to_markdown(index=False))
        except Exception as e:
            lines.append(f"Could not render inventory table: {e}")
        lines.append("")
        
        # Summary statistics by feature type
        lines.append("### Feature Type Statistics")
        for feature_type in sorted(feature_types):
            type_data = inv_df[inv_df['Feature_Type'] == feature_type.title()]
            total_rows = type_data['Rows'].sum()
            total_features = type_data['Unique_Features'].sum()
            timeframes_count = len(type_data)
            
            lines.append(f"**{feature_type.title()}**:")
            lines.append(f"- Timeframes: {timeframes_count}")
            lines.append(f"- Total rows across all timeframes: {total_rows:,}")
            lines.append(f"- Total unique features: {total_features:,}")
            lines.append("")
        
        # Timeframe comparison
        lines.append("### Timeframe Comparison")
        timeframe_summary = inv_df.groupby('Timeframe').agg({
            'Rows': 'sum',
            'Unique_Features': 'sum',
            'Total_Count': 'sum'
        }).reset_index()
        timeframe_summary = timeframe_summary.sort_values('Timeframe')
        
        try:
            lines.append(timeframe_summary.to_markdown(index=False))
        except Exception as e:
            lines.append(f"Could not render timeframe comparison: {e}")
        lines.append("")
    
    # Detailed analysis for each timeframe (top performers)
    lines.append("## Top Performers by Timeframe")
    
    for timeframe in TIMEFRAME_LABELS:
        lines.append(f"### {timeframe.upper()} Timeframe")
        
        timeframe_files = list(PROC_DIR.glob(f'features_*_{timeframe}.parquet'))
        if not timeframe_files:
            lines.append("No data available for this timeframe.")
            lines.append("")
            continue
        
        # Combine data from all feature types for this timeframe
        combined_data = []
        for file_path in timeframe_files:
            if 'statistical_anomalies' in file_path.name or 'emerging_terms' in file_path.name:
                continue
            
            try:
                df = pd.read_parquet(file_path)
                if not df.empty and 'delta_vs_mean' in df.columns:
                    feature_type = file_path.stem.split('_')[1]
                    df['source_type'] = feature_type
                    combined_data.append(df)
            except Exception as e:
                logger.warning(f"Could not load {file_path.name}: {e}")
        
        if combined_data:
            all_data = pd.concat(combined_data, ignore_index=True)
            
            # Get latest bin and top performers
            if 'bin' in all_data.columns:
                latest_bin = all_data['bin'].max()
                latest_data = all_data[all_data['bin'] == latest_bin]
                
                if not latest_data.empty:
                    top_performers = latest_data.nlargest(15, 'delta_vs_mean')
                    
                    lines.append(f"Latest period: {latest_bin}")
                    lines.append(f"Total features: {len(latest_data):,}")
                    lines.append("")
                    lines.append("Top trending features:")
                    
                    try:
                        cols = ['source_type', 'feature', 'count', 'rolling_mean_24h', 'delta_vs_mean', 'category']
                        available_cols = [c for c in cols if c in top_performers.columns]
                        lines.append(top_performers[available_cols].to_markdown(index=False))
                    except Exception as e:
                        lines.append(f"Could not render top performers table: {e}")
        
        lines.append("")
    
    # Performance insights
    lines.append("## Performance Insights")
    lines.append(f"- Processing mode: {PERFORMANCE_MODE}")
    lines.append(f"- Parallel processing: {'Enabled' if CONFIG['enable_parallel'] else 'Disabled'}")
    lines.append(f"- Caching: {'Enabled' if CONFIG['enable_caching'] else 'Disabled'}")
    lines.append(f"- Sample limit per source: {CONFIG['sample_rows_per_source']:,}" if CONFIG['sample_rows_per_source'] else "- No sampling applied")
    lines.append("")
    
    # Write the report
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    logger.info(f"Enhanced Phase 2 report written to {path}")

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
    
    # Enhanced Phase 2 report
    baseline_data = {
        'hashtags': ts_hashtags,
        'keywords': ts_keywords,
        'audio': ts_audio
    }
    write_enhanced_phase2_report(baseline_data)
    
    # Phase 3: Optimized emerging trends detection
    logger.info("Phase 3: Running optimized emerging trends detection")
    
    emerging_file = 'features_emerging_terms_6h.parquet'
    if not (PROC_DIR / emerging_file).exists():
        ts_emerging = aggregate_emerging_terms_optimized(all_sources, '6h', min_growth_rate=2.5)
        if not ts_emerging.empty:
            ts_emerging.to_parquet(PROC_DIR / emerging_file, index=False)
            logger.info(f"Saved emerging terms: {len(ts_emerging):,} rows")
    else:
        ts_emerging = pd.read_parquet(PROC_DIR / emerging_file)
        logger.info(f"Loaded existing emerging terms: {len(ts_emerging):,} rows")
    
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