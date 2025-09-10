#!/usr/bin/env python3
"""
Configuration module for L'Oréal Datathon 2025 Data Processing Pipeline
Contains all configuration settings, constants, and performance parameters.
"""

import psutil
from pathlib import Path

# -----------------------------
# Performance Configuration
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

# -----------------------------
# Directory Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed" / "dataset"
INTERIM_DIR = ROOT / "data" / "interim"
CACHE_DIR = ROOT / "data" / "cache"

# -----------------------------
# Timeframe Configurations
# -----------------------------
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
# Keywords and Categories
# -----------------------------
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
KEYWORD_CATEGORY = {}

# Define category lists
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

# Build category mapping
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

# Target categories for relevance filtering
RELEVANT_CATEGORIES = {"Beauty", "Fashion", "Skincare", "Makeup", "Hair"}

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
