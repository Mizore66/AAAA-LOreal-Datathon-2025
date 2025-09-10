#!/usr/bin/env python3
"""
Aggregation functions for L'Oréal Datathon 2025
Handles hashtag, keyword, and emerging terms aggregation.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Tuple, Optional
from .config import (
    _FREQ_MAP, _ROLLING_WINDOW, _MIN_FREQ_BY_LABEL, 
    MIN_GROWTH_RATE_BY_LABEL, KEYWORDS, CONFIG
)
from .data_loading import prepare_text_df_optimized
from .text_processing import categorize_feature, extract_all_terms_optimized, filter_relevant_emerging_terms

logger = logging.getLogger(__name__)

def assign_time_bin_optimized(ts: pd.Series, label: str) -> pd.Series:
    """Optimized time bin assignment."""
    if label in _FREQ_MAP:
        return ts.dt.floor(_FREQ_MAP[label])
    elif label == '1m':
        return ts.dt.to_period('M').dt.start_time
    elif label == '3m':
        return (ts.dt.to_period('Q').dt.start_time)
    elif label == '6m':
        return (ts.dt.year.astype(str) + '-' + 
                ((ts.dt.month - 1) // 6 * 6 + 1).astype(str).str.zfill(2)).apply(
                    lambda x: pd.to_datetime(x, format='%Y-%m'))
    else:
        return ts.dt.floor('1D')

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

def aggregate_hashtags_optimized(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Optimized hashtag aggregation for a single source."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
    
    # Efficient time binning
    p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    # Explode hashtags efficiently
    pe = p[['bin', 'hashtags']].explode('hashtags').dropna(subset=['hashtags'])
    if pe.empty:
        return None
    
    pe = pe.rename(columns={'hashtags': 'feature'})
    
    # Add engagement weighting if available
    if 'engagement_score' in p.columns:
        # Merge engagement scores back
        pe = pe.merge(p[['bin', 'engagement_score']], on='bin', how='left')
        pe['weighted_count'] = 1 + (pe['engagement_score'] / 100).fillna(0)
    else:
        pe['weighted_count'] = 1
    
    # Group and aggregate with weighting
    g = pe.groupby(['bin', 'feature'], as_index=False).agg({
        'weighted_count': 'sum'
    }).rename(columns={'weighted_count': 'count'})
    
    # Round counts and ensure minimum of 1
    g['count'] = g['count'].round().astype(int).clip(lower=1)
    
    g['category'] = g['feature'].apply(categorize_feature)
    g['source_type'] = dataset_type
    
    return g

def aggregate_keywords_optimized(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Optimized keyword aggregation for a single source."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
    
    # Pre-compile patterns for better performance
    kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
    
    p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    frames = []
    for kw, pat in kw_patterns:
        matches = p[p['text_clean'].str.contains(pat, na=False)]
        if not matches.empty:
            kw_counts = matches.groupby('bin', as_index=False).size().rename(columns={'size': 'count'})
            kw_counts['feature'] = kw
            kw_counts['category'] = categorize_feature(kw)
            kw_counts['source_type'] = dataset_type
            frames.append(kw_counts)
    
    if not frames:
        return None
    
    return pd.concat(frames, ignore_index=True)

def process_emerging_source(name: str, df: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
    """Process a single source for emerging terms."""
    p = prepare_text_df_optimized(df)
    if p is None or p.empty:
        return None
    
    dataset_type = p['dataset_type'].iloc[0] if 'dataset_type' in p.columns else 'unknown'
    
    # Efficient time binning
    p['bin'] = assign_time_bin_optimized(p['ts'], label)
    
    # Extract all terms from text
    all_terms = []
    for _, row in p.iterrows():
        terms = extract_all_terms_optimized(row['text_clean'])
        for term in terms:
            all_terms.append({
                'bin': row['bin'],
                'feature': term,
                'source_type': dataset_type
            })
    
    if not all_terms:
        return None
    
    # Convert to DataFrame and aggregate
    terms_df = pd.DataFrame(all_terms)
    g = terms_df.groupby(['bin', 'feature', 'source_type'], as_index=False).size().rename(columns={'size': 'count'})
    g['category'] = g['feature'].apply(categorize_feature)
    
    return g

def aggregate_emerging_terms_optimized(dfs: List[Tuple[str, pd.DataFrame]], 
                                     label: str = '6h',
                                     min_growth_rate: float = 2.0) -> pd.DataFrame:
    """Optimized emerging terms detection with performance improvements."""
    
    min_frequency = _MIN_FREQ_BY_LABEL.get(label, 3)
    
    logger.info(f"Computing emerging terms for {label} with {len(dfs)} sources")
    
    # Process sources
    frames = []
    for name, df in dfs:
        result = process_emerging_source(name, df, label)
        if result is not None:
            frames.append(result)
    
    if not frames:
        logger.warning("No emerging terms data generated")
        return pd.DataFrame()
    
    # Combine all data
    allg = pd.concat(frames, ignore_index=True)
    
    # Handle source_type aggregation
    if 'source_type' in allg.columns:
        allg = allg.groupby(["bin", "feature", "source_type"], as_index=False)["count"].sum()
    else:
        allg = allg.groupby(["bin", "feature"], as_index=False)["count"].sum()
    
    # Filter rare terms early for performance
    if 'source_type' in allg.columns:
        total_counts = allg.groupby(["feature", "source_type"])["count"].sum()
        frequent_terms = total_counts[total_counts >= min_frequency].index
        # Convert to set for faster lookup
        frequent_set = set((feat, src) for feat, src in frequent_terms)
        allg = allg[allg.apply(lambda row: (row['feature'], row['source_type']) in frequent_set, axis=1)]
    else:
        total_counts = allg.groupby("feature")["count"].sum()
        frequent_terms = total_counts[total_counts >= min_frequency].index
        allg = allg[allg["feature"].isin(frequent_terms)]
    
    # Apply feature limits for performance
    if CONFIG["max_features_emerging"] and len(allg['feature'].unique()) > CONFIG["max_features_emerging"]:
        top_features = (allg.groupby('feature')['count'].sum()
                       .nlargest(CONFIG["max_features_emerging"]).index)
        allg = allg[allg['feature'].isin(top_features)]
    
    if allg.empty:
        logger.warning("No frequent emerging terms found")
        return pd.DataFrame()
    
    # Optimized growth rate calculation
    if 'source_type' in allg.columns:
        allg = allg.sort_values(["feature", "source_type", "bin"]).reset_index(drop=True)
        allg['prev_count'] = allg.groupby(['feature', 'source_type'])['count'].shift(1)
    else:
        allg = allg.sort_values(["feature", "bin"]).reset_index(drop=True)
        allg['prev_count'] = allg.groupby('feature')['count'].shift(1)
    
    allg['growth_rate'] = allg['count'] / (allg['prev_count'] + 1e-6)
    allg.loc[allg['prev_count'].isna(), 'growth_rate'] = 1.0
    
    # Identify emerging terms with dataset-aware thresholds
    if 'source_type' in allg.columns:
        video_mask = allg['source_type'] == 'videos'
        comment_mask = allg['source_type'] == 'comments'
        
        allg.loc[video_mask, "is_emerging"] = allg.loc[video_mask, "growth_rate"] >= (min_growth_rate * 1.2)
        allg.loc[comment_mask, "is_emerging"] = allg.loc[comment_mask, "growth_rate"] >= min_growth_rate
    else:
        allg["is_emerging"] = allg["growth_rate"] >= min_growth_rate
    
    allg["category"] = allg["feature"].apply(categorize_feature)
    
    # Apply relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair
    logger.info("Applying relevance filtering for Beauty/Fashion/Skincare/Makeup/Hair trends")
    allg = filter_relevant_emerging_terms(allg)
    
    # Simple velocity calculation
    allg['velocity'] = 0.0
    
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
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception:
        return 0.0
