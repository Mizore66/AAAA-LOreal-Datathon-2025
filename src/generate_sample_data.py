#!/usr/bin/env python3
"""
Generate synthetic beauty trend data for testing the TrendSpotter pipeline.
This creates realistic-looking data that matches the expected structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

# Create directories
for dir_path in [RAW_DIR, PROC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Beauty industry terms and hashtags
TRENDING_TERMS = [
    "niacinamide", "retinol", "vitamin c", "hyaluronic acid", "sunscreen",
    "skincare", "makeup", "foundation", "concealer", "blush", "mascara",
    "hair mask", "scalp serum", "k beauty", "glass skin", "slugging",
    "skin cycling", "double cleanse", "chemical exfoliant", "dewy skin"
]

HASHTAGS = [
    "skincare", "makeup", "beauty", "kbeauty", "skincareroutine", "glowup",
    "makeuptutorial", "beautytips", "skincareaddicts", "naturalskincare",
    "antiaging", "acne", "dryskin", "sensitiveskin", "combination skin"
]

CATEGORIES = ["Skincare", "Makeup", "Hair", "Other"]

def generate_time_series_data(start_date: datetime, days: int = 30, freq_hours: int = 6) -> pd.DataFrame:
    """Generate time series data with realistic beauty trends."""
    
    # Create time bins (6-hour intervals)
    time_bins = pd.date_range(
        start=start_date,
        periods=days * 24 // freq_hours,
        freq=f'{freq_hours}H'
    )
    
    data = []
    
    # Generate data for trending terms
    for term in TRENDING_TERMS:
        base_frequency = random.randint(5, 50)
        trend_strength = random.uniform(0.8, 1.5)
        
        for i, bin_time in enumerate(time_bins):
            # Add some seasonality and trend
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / (24 // freq_hours))  # Daily pattern
            trend_factor = 1 + (i / len(time_bins)) * trend_strength * random.uniform(-0.1, 0.2)
            
            # Add random spikes for viral trends
            spike_factor = 1
            if random.random() < 0.05:  # 5% chance of viral spike
                spike_factor = random.uniform(3, 8)
            
            count = max(1, int(base_frequency * seasonal_factor * trend_factor * spike_factor * random.uniform(0.7, 1.3)))
            
            # Calculate rolling mean (simulate previous periods)
            rolling_mean = base_frequency * seasonal_factor * trend_factor
            delta_vs_mean = count - rolling_mean
            
            # Calculate growth rate (simulate change from previous period)
            if i > 0:
                prev_count = max(1, base_frequency * (1 + 0.3 * np.sin(2 * np.pi * (i-1) / (24 // freq_hours))))
                growth_rate = count / prev_count
            else:
                growth_rate = 1.0
            
            # Calculate velocity (rate of change)
            velocity = (count - rolling_mean) / max(rolling_mean, 1)
            
            data.append({
                'bin': bin_time,
                'feature': term,
                'count': count,
                'rolling_mean_24h': rolling_mean,
                'delta_vs_mean': delta_vs_mean,
                'growth_rate': growth_rate,
                'velocity': velocity,
                'category': 'Skincare' if any(x in term.lower() for x in ['skin', 'acid', 'serum', 'vitamin', 'retinol', 'niacinamide']) 
                          else 'Makeup' if any(x in term.lower() for x in ['makeup', 'foundation', 'concealer', 'mascara', 'blush'])
                          else 'Hair' if any(x in term.lower() for x in ['hair', 'scalp'])
                          else 'Other'
            })
    
    # Generate hashtag data
    for hashtag in HASHTAGS:
        base_frequency = random.randint(10, 100)
        
        for i, bin_time in enumerate(time_bins):
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / (24 // freq_hours))
            
            # Hashtags tend to have more viral spikes
            spike_factor = 1
            if random.random() < 0.08:  # 8% chance of viral spike
                spike_factor = random.uniform(2, 5)
            
            count = max(1, int(base_frequency * seasonal_factor * spike_factor * random.uniform(0.8, 1.2)))
            rolling_mean = base_frequency * seasonal_factor
            delta_vs_mean = count - rolling_mean
            
            data.append({
                'bin': bin_time,
                'feature': f"#{hashtag}",
                'count': count,
                'rolling_mean_24h': rolling_mean,
                'delta_vs_mean': delta_vs_mean,
                'category': 'Skincare' if any(x in hashtag.lower() for x in ['skin', 'care', 'anti', 'glow'])
                          else 'Makeup' if any(x in hashtag.lower() for x in ['makeup', 'beauty', 'tutorial'])
                          else 'Other'
            })
    
    return pd.DataFrame(data)


def generate_emerging_terms_data(start_date: datetime, days: int = 30) -> pd.DataFrame:
    """Generate emerging terms data with growth indicators."""
    
    # Simulate emerging beauty terms
    emerging_terms = [
        "peptide serum", "bakuchiol", "tranexamic acid", "azelaic acid",
        "snail mucin", "centella asiatica", "niacinamide serum", "copper peptides",
        "latte makeup", "soft glam", "underpainting", "cream blush technique",
        "rosemary oil", "bond builder", "scalp scrub", "rice water"
    ]
    
    time_bins = pd.date_range(
        start=start_date,
        periods=days * 4,  # 6-hour intervals
        freq='6H'
    )
    
    data = []
    
    for term in emerging_terms:
        # Emerging terms start small and grow rapidly
        initial_count = random.randint(1, 5)
        growth_rate_base = random.uniform(1.1, 2.5)  # 10% to 150% growth
        
        for i, bin_time in enumerate(time_bins):
            # Exponential growth with some noise
            growth_factor = growth_rate_base ** (i / 10)  # Slower growth over time
            count = max(1, int(initial_count * growth_factor * random.uniform(0.7, 1.4)))
            
            # Calculate if it's considered "emerging"
            is_emerging = growth_rate_base > 1.5 and i >= 5  # After some initial periods
            
            # Calculate velocity
            if i > 0:
                prev_count = max(1, initial_count * (growth_rate_base ** ((i-1) / 10)))
                velocity = (count - prev_count) / max(prev_count, 1)
                actual_growth_rate = count / prev_count
            else:
                velocity = 0
                actual_growth_rate = 1.0
            
            data.append({
                'bin': bin_time,
                'feature': term,
                'count': count,
                'growth_rate': actual_growth_rate,
                'velocity': velocity,
                'is_emerging': is_emerging,
                'category': 'Skincare' if any(x in term.lower() for x in ['serum', 'acid', 'mucin', 'peptide', 'oil'])
                          else 'Makeup' if any(x in term.lower() for x in ['makeup', 'glam', 'blush', 'latte'])
                          else 'Hair' if any(x in term.lower() for x in ['hair', 'scalp', 'bond', 'rosemary', 'rice'])
                          else 'Other'
            })
    
    return pd.DataFrame(data)


def generate_audio_data(start_date: datetime, days: int = 30) -> pd.DataFrame:
    """Generate audio/music trend data."""
    
    # Simulate popular audio tracks for beauty content
    audio_tracks = [
        "trending_audio_001", "trending_audio_002", "trending_audio_003",
        "viral_sound_beauty", "makeup_trend_audio", "skincare_routine_sound",
        "get_ready_with_me_music", "transformation_audio", "beauty_hack_sound"
    ]
    
    time_bins = pd.date_range(
        start=start_date,
        periods=days * 4,
        freq='6H'
    )
    
    data = []
    
    for audio in audio_tracks:
        base_frequency = random.randint(5, 30)
        
        for i, bin_time in enumerate(time_bins):
            # Audio trends can be very spiky
            spike_factor = 1
            if random.random() < 0.12:  # 12% chance of going viral
                spike_factor = random.uniform(5, 15)
            
            count = max(1, int(base_frequency * spike_factor * random.uniform(0.6, 1.4)))
            rolling_mean = base_frequency
            delta_vs_mean = count - rolling_mean
            
            data.append({
                'bin': bin_time,
                'feature': audio,
                'count': count,
                'rolling_mean_24h': rolling_mean,
                'delta_vs_mean': delta_vs_mean,
                'category': 'Other'  # Audio doesn't fit beauty categories directly
            })
    
    return pd.DataFrame(data)


def main():
    """Generate synthetic datasets for testing."""
    print("Generating synthetic beauty trend data...")
    
    start_date = datetime.now() - timedelta(days=30)
    
    # Generate different types of features
    hashtag_data = generate_time_series_data(start_date, days=30)
    hashtag_features = hashtag_data[hashtag_data['feature'].str.startswith('#')]
    keyword_features = hashtag_data[~hashtag_data['feature'].str.startswith('#')]
    
    emerging_data = generate_emerging_terms_data(start_date, days=30)
    audio_data = generate_audio_data(start_date, days=30)
    
    # Save as parquet files
    hashtag_features.to_parquet(PROC_DIR / 'features_hashtags_6h.parquet', index=False)
    keyword_features.to_parquet(PROC_DIR / 'features_keywords_6h.parquet', index=False)
    emerging_data.to_parquet(PROC_DIR / 'features_emerging_terms_6h.parquet', index=False)
    audio_data.to_parquet(PROC_DIR / 'features_audio_6h.parquet', index=False)
    
    print(f"Generated datasets:")
    print(f"  - Hashtags: {len(hashtag_features)} rows")
    print(f"  - Keywords: {len(keyword_features)} rows")
    print(f"  - Emerging terms: {len(emerging_data)} rows")
    print(f"  - Audio: {len(audio_data)} rows")
    
    print(f"\nData saved to: {PROC_DIR}")
    
    # Also create some sample raw data to simulate the ingestion process
    print("\nGenerating sample raw data...")
    
    # Create comments data
    comments_data = []
    for i in range(1000):
        timestamp = start_date + timedelta(hours=random.randint(0, 30*24))
        
        # Generate realistic beauty comments
        beauty_terms = random.sample(TRENDING_TERMS + [h.replace('#', '') for h in HASHTAGS], k=random.randint(1, 3))
        hashtags = random.sample(HASHTAGS, k=random.randint(0, 2))
        
        comment_text = f"Love this {' and '.join(beauty_terms)} routine! " + ' '.join([f'#{h}' for h in hashtags])
        
        comments_data.append({
            'comment_id': f'comment_{i:04d}',
            'text': comment_text,
            'created_at': timestamp,
            'platform': random.choice(['tiktok', 'instagram', 'youtube'])
        })
    
    comments_df = pd.DataFrame(comments_data)
    comments_df.to_parquet(PROC_DIR / 'comments.parquet', index=False)
    
    # Create videos data
    videos_data = []
    for i in range(500):
        timestamp = start_date + timedelta(hours=random.randint(0, 30*24))
        
        beauty_terms = random.sample(TRENDING_TERMS, k=random.randint(1, 2))
        title = f"{' '.join(beauty_terms).title()} Tutorial - Must Try!"
        
        videos_data.append({
            'video_id': f'video_{i:04d}',
            'title': title,
            'description': f"Check out this amazing {' and '.join(beauty_terms)} tutorial!",
            'upload_date': timestamp,
            'platform': random.choice(['tiktok', 'instagram', 'youtube']),
            'audio_track': random.choice(['trending_audio_001', 'trending_audio_002', 'viral_sound_beauty'])
        })
    
    videos_df = pd.DataFrame(videos_data)
    videos_df.to_parquet(PROC_DIR / 'videos.parquet', index=False)
    
    print(f"  - Comments: {len(comments_df)} rows")
    print(f"  - Videos: {len(videos_df)} rows")
    
    print("\nSynthetic data generation complete!")


if __name__ == '__main__':
    main()