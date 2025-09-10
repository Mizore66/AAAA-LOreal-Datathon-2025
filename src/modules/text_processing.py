#!/usr/bin/env python3
"""
Text processing utilities for L'Oréal Datathon 2025
Handles text cleaning, feature extraction, and categorization.
"""

import re
import pandas as pd
from typing import List, Optional
from .config import KEYWORD_CATEGORY, RELEVANT_CATEGORIES, STOPWORDS

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
            if word_count >= max_terms // 2:
                break
    
    # Add bigrams (limited)
    terms.extend(extract_ngrams_optimized(text, 2, max_terms // 4))
    
    # Only add trigrams if we have space and they're valuable
    if len(terms) < max_terms * 0.75:
        terms.extend(extract_ngrams_optimized(text, 3, max_terms // 6))
    
    return terms[:max_terms]  # Hard limit

def categorize_feature(feature: str) -> str:
    """Optimized feature categorization with enhanced Beauty/Fashion support."""
    if not isinstance(feature, str):
        return "Other"
    
    f = feature.lstrip('#').lower()
    
    # Direct keyword match (fastest)
    if f in KEYWORD_CATEGORY:
        return KEYWORD_CATEGORY[f]
    
    # Heuristic contains-based match
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
        if indicator in f and len(f) > 5:
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
    
    return relevant_df
