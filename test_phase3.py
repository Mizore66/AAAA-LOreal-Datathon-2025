#!/usr/bin/env python3
"""
Test script for Phase 3 enhanced trend detection functionality.
Creates synthetic data to test the new keyword-independent trend detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing import (
    extract_ngrams, extract_all_terms, calculate_term_velocity,
    detect_statistical_anomalies, aggregate_emerging_terms_6h,
    identify_trend_clusters, PROC_DIR
)

def create_synthetic_data():
    """Create synthetic social media data for testing."""
    
    # Create data directory
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sample texts with emerging beauty trends
    base_time = datetime.now() - timedelta(days=7)
    
    texts = [
        # Traditional beauty content
        "Just tried this new foundation! Love the coverage #makeup #beauty",
        "Niacinamide serum is amazing for my skin #skincare #glow",
        "Sunday skincare routine with retinol and vitamin c #selfcare",
        
        # Emerging trends (will be repeated more over time)
        "Peptide eye cream is the new holy grail #skincare #antiaging",
        "Glass skin technique with slug gel is trending #glassskin #kbeauty", 
        "Barrier repair with ceramide slug method #barrierrepair #slugging",
        "Peptide slug combo for ultimate hydration #peptides #hydration",
        "New viral slug cream technique everyone's trying #viral #skincare",
        
        # More emerging content that grows over time
        "Cica slug treatment changed my skin completely #cica #skincare",
        "Peptide barrier slug routine is my new obsession #peptides #barrier",
        "Glass slug method for dewy skin goals #glassskin #dewyskin",
    ]
    
    # Create time series data with emerging trends
    data = []
    
    for day in range(7):
        current_time = base_time + timedelta(days=day)
        
        # Base frequency for all content
        for _ in range(50):
            text = np.random.choice(texts[:3])  # Traditional content
            data.append({
                'created_time': current_time + timedelta(hours=np.random.randint(0, 24)),
                'text': text,
                'engagement': np.random.randint(10, 100)
            })
        
        # Emerging content - frequency increases over time
        emerging_multiplier = 1 + (day * 2)  # Grows over time
        for _ in range(int(20 * emerging_multiplier)):
            text = np.random.choice(texts[3:])  # Emerging content
            data.append({
                'created_time': current_time + timedelta(hours=np.random.randint(0, 24)),
                'text': text,
                'engagement': np.random.randint(50, 500)  # Higher engagement
            })
    
    df = pd.DataFrame(data)
    
    # Save as parquet
    df.to_parquet(PROC_DIR / 'synthetic_comments.parquet', index=False)
    print(f"Created synthetic data with {len(df)} rows")
    
    return df

def test_trend_detection():
    """Test the new trend detection functionality."""
    print("Testing Phase 3 trend detection...")
    
    # Create synthetic data
    df = create_synthetic_data()
    
    # Test n-gram extraction
    sample_text = "peptide barrier slug routine is amazing"
    bigrams = extract_ngrams(sample_text, 2)
    trigrams = extract_ngrams(sample_text, 3)
    all_terms = extract_all_terms(sample_text)
    
    print(f"Sample text: '{sample_text}'")
    print(f"Bigrams: {bigrams}")
    print(f"Trigrams: {trigrams}")
    print(f"All terms: {all_terms}")
    
    print("\nTesting completed successfully!")
    print("Run 'python src/data_processing.py' to perform full analysis")

if __name__ == '__main__':
    test_trend_detection()