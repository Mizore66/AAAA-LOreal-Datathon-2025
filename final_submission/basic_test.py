#!/usr/bin/env python3
"""
Basic functionality test that creates sample data and runs simple analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

def create_test_data():
    """Create minimal test data"""
    print("📊 Creating test data...")
    
    # Create sample comments
    np.random.seed(42)
    comments_data = []
    
    beauty_terms = ['skincare', 'makeup', 'foundation', 'mascara', 'lipstick']
    hashtags = ['#beauty', '#makeup', '#skincare', '#fashion', '#style']
    
    for i in range(100):
        term = np.random.choice(beauty_terms)
        hashtag = np.random.choice(hashtags)
        comment = f"Love this {term}! {hashtag}"
        
        comments_data.append({
            'textOriginal': comment,
            'likeCount': np.random.poisson(25),
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i%30)
        })
    
    df = pd.DataFrame(comments_data)
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    df.to_parquet('data/test_comments.parquet', index=False)
    
    print(f"✅ Created test data: {len(df)} comments")
    return df

def simple_analysis(df):
    """Perform simple analysis"""
    print("\n🔍 Running simple analysis...")
    
    # Basic statistics
    stats = {
        'total_comments': len(df),
        'avg_likes': df['likeCount'].mean(),
        'date_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat()
        }
    }
    
    # Simple sentiment analysis
    positive_words = ['love', 'amazing', 'great', 'perfect']
    df['has_positive'] = df['textOriginal'].str.lower().str.contains('|'.join(positive_words))
    stats['positive_sentiment_rate'] = df['has_positive'].mean()
    
    # Extract hashtags
    hashtags = []
    for text in df['textOriginal']:
        import re
        tags = re.findall(r'#\w+', text.lower())
        hashtags.extend(tags)
    
    if hashtags:
        hashtag_counts = pd.Series(hashtags).value_counts()
        stats['top_hashtags'] = hashtag_counts.head().to_dict()
    
    print(f"✅ Analysis complete:")
    print(f"   📊 Total comments: {stats['total_comments']}")
    print(f"   👍 Average likes: {stats['avg_likes']:.1f}")
    print(f"   😊 Positive sentiment: {stats['positive_sentiment_rate']:.1%}")
    
    # Save results
    with open('data/analysis_results.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    return stats

def main():
    """Run basic functionality test"""
    print("🚀 BASIC FUNCTIONALITY TEST")
    print("=" * 40)
    
    try:
        # Create test data
        df = create_test_data()
        
        # Run analysis
        results = simple_analysis(df)
        
        print(f"\n🎉 Basic functionality test PASSED!")
        print(f"📁 Results saved to data/analysis_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)