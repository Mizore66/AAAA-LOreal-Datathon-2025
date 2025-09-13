#!/usr/bin/env python3
"""
Test script to validate the pipeline execution without real data.
Creates sample data to test the pipeline functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

def create_sample_data():
    """Create sample data for testing the pipeline."""
    print("Creating sample data for testing...")
    
    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create sample video data
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(hours=i*6) for i in range(120)]  # 30 days of 6-hour intervals
    
    video_data = []
    hashtags = ["#skincare", "#makeup", "#beauty", "#routine", "#glowup", "#selfcare", "#aesthetic"]
    keywords = ["niacinamide", "retinol", "vitamin c", "hyaluronic acid", "sunscreen", "moisturizer"]
    
    for i, date in enumerate(dates):
        # Create some video records
        for j in range(np.random.randint(1, 10)):
            title = f"Beauty video {i}_{j}"
            if np.random.random() > 0.7:
                title += f" {np.random.choice(keywords)}"
            
            caption = f"Check out this amazing routine! {np.random.choice(hashtags)}"
            if np.random.random() > 0.5:
                caption += f" {np.random.choice(hashtags)}"
            
            video_data.append({
                'video_id': f'video_{i}_{j}',
                'title': title,
                'caption': caption,
                'upload_time': date,
                'view_count': np.random.randint(100, 10000),
                'like_count': np.random.randint(10, 1000)
            })
    
    video_df = pd.DataFrame(video_data)
    video_df.to_parquet(PROC_DIR / 'sample_videos.parquet', index=False)
    print(f"Created sample video data: {len(video_df)} records")
    
    # Create sample comment data
    comment_data = []
    for i, video_row in video_df.iterrows():
        # Create comments for each video
        for k in range(np.random.randint(0, 5)):
            comment_text = f"Great video! {np.random.choice(hashtags)} love this"
            if np.random.random() > 0.6:
                comment_text += f" {np.random.choice(keywords)} works so well"
                
            comment_data.append({
                'comment_id': f'comment_{i}_{k}',
                'video_id': video_row['video_id'],
                'text': comment_text,
                'created_time': video_row['upload_time'] + timedelta(hours=np.random.randint(1, 24)),
                'like_count': np.random.randint(0, 100)
            })
    
    comment_df = pd.DataFrame(comment_data)
    comment_df.to_parquet(PROC_DIR / 'sample_comments.parquet', index=False)
    print(f"Created sample comment data: {len(comment_df)} records")
    
    return len(video_df), len(comment_df)

def test_data_processing():
    """Test the data processing pipeline with sample data."""
    print("\nTesting data processing pipeline...")
    
    try:
        # Import and run data processing
        from data_processing import load_samples, aggregate_hashtags_6h, aggregate_keywords_6h
        
        samples = load_samples()
        print(f"Loaded {len(samples)} sample datasets")
        
        if samples:
            text_sources = list(samples.items())
            
            # Test hashtag aggregation
            hashtags_result = aggregate_hashtags_6h(text_sources)
            print(f"Hashtag aggregation result: {len(hashtags_result)} rows")
            
            # Test keyword aggregation  
            keywords_result = aggregate_keywords_6h(text_sources)
            print(f"Keyword aggregation result: {len(keywords_result)} rows")
            
            return True
        else:
            print("No sample data found for processing")
            return False
            
    except Exception as e:
        print(f"Data processing test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("PIPELINE TEST - SAMPLE DATA VALIDATION")
    print("=" * 60)
    
    # Create sample data
    video_count, comment_count = create_sample_data()
    
    # Test data processing
    processing_success = test_data_processing()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Sample videos created: {video_count}")
    print(f"Sample comments created: {comment_count}")
    print(f"Data processing test: {'PASSED' if processing_success else 'FAILED'}")
    print("=" * 60)
    
    return processing_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)