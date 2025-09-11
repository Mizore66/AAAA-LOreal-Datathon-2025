#!/usr/bin/env python3
"""
🎭 L'Oréal Datathon 2025 - Data Analysis Results
==============================================

Analyze processed data results and extract insights from beauty trend data.
"""

import pandas as pd
import json
import numpy as np
from collections import Counter
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_processed_data():
    """Analyze all processed data files and extract insights."""
    
    logger.info("🎭 L'Oréal Datathon 2025 - Data Analysis Results")
    logger.info("=" * 60)
    
    # Load processing results
    results_file = Path("data/interim/data_processing_results.json")
    if not results_file.exists():
        logger.error("❌ No processing results found. Run pipeline first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    logger.info(f"📊 Found {len(results['processed_datasets'])} processed datasets")
    
    # Analyze each dataset
    all_data = []
    category_stats = {}
    engagement_stats = {}
    hashtag_analysis = {}
    
    for file_path in results['processed_datasets']:
        if not Path(file_path).exists():
            logger.warning(f"⚠️  File not found: {file_path}")
            continue
            
        logger.info(f"📈 Analyzing: {Path(file_path).name}")
        df = pd.read_parquet(file_path)
        
        # Basic stats
        dataset_name = Path(file_path).stem.replace('_processed', '')
        logger.info(f"   • Rows: {len(df):,}")
        logger.info(f"   • Columns: {len(df.columns)}")
        
        # Category analysis
        if 'category' in df.columns:
            cat_counts = df['category'].value_counts()
            category_stats[dataset_name] = cat_counts.to_dict()
            logger.info(f"   • Top category: {cat_counts.index[0]} ({cat_counts.iloc[0]:,} posts)")
        
        # Engagement analysis
        if 'engagement_score' in df.columns:
            eng_stats = {
                'mean': df['engagement_score'].mean(),
                'median': df['engagement_score'].median(),
                'max': df['engagement_score'].max()
            }
            engagement_stats[dataset_name] = eng_stats
            logger.info(f"   • Avg engagement: {eng_stats['mean']:.2f}")
        
        # Hashtag analysis
        if 'tokens' in df.columns:
            # Extract hashtags from tokens
            hashtags = []
            for tokens_str in df['tokens'].dropna():
                if isinstance(tokens_str, str):
                    # Parse tokens (could be string representation of list)
                    tokens = eval(tokens_str) if tokens_str.startswith('[') else tokens_str.split()
                elif isinstance(tokens_str, list):
                    tokens = tokens_str
                else:
                    continue
                    
                # Find hashtags
                for token in tokens:
                    if isinstance(token, str) and token.startswith('#'):
                        hashtags.append(token.lower())
            
            if hashtags:
                hashtag_counts = Counter(hashtags)
                hashtag_analysis[dataset_name] = dict(hashtag_counts.most_common(10))
                logger.info(f"   • Top hashtag: {hashtag_counts.most_common(1)[0][0]} ({hashtag_counts.most_common(1)[0][1]} times)")
        
        all_data.append(df)
    
    # Generate comprehensive analysis
    logger.info("\n🎯 COMPREHENSIVE ANALYSIS")
    logger.info("=" * 60)
    
    # Overall category distribution
    logger.info("\n📊 Overall Category Distribution:")
    total_categories = Counter()
    for dataset, cats in category_stats.items():
        for cat, count in cats.items():
            total_categories[cat] += count
    
    for cat, count in total_categories.most_common():
        percentage = (count / sum(total_categories.values())) * 100
        logger.info(f"   • {cat.title()}: {count:,} posts ({percentage:.1f}%)")
    
    # Top hashtags across all datasets
    logger.info("\n🏷️  Top Hashtags Across All Datasets:")
    all_hashtags = Counter()
    for dataset, hashtags in hashtag_analysis.items():
        for hashtag, count in hashtags.items():
            all_hashtags[hashtag] += count
    
    for hashtag, count in all_hashtags.most_common(15):
        logger.info(f"   • {hashtag}: {count} times")
    
    # Engagement insights
    logger.info("\n💡 Engagement Insights:")
    avg_engagement_by_dataset = {k: v['mean'] for k, v in engagement_stats.items()}
    sorted_engagement = sorted(avg_engagement_by_dataset.items(), key=lambda x: x[1], reverse=True)
    
    for dataset, avg_eng in sorted_engagement:
        logger.info(f"   • {dataset}: {avg_eng:.2f} avg engagement")
    
    # Save detailed analysis
    analysis_results = {
        "summary": {
            "total_datasets": len(results['processed_datasets']),
            "total_rows": sum(len(df) for df in all_data),
            "analysis_date": pd.Timestamp.now().isoformat()
        },
        "category_distribution": dict(total_categories),
        "top_hashtags": dict(all_hashtags.most_common(20)),
        "engagement_by_dataset": engagement_stats,
        "category_by_dataset": category_stats,
        "hashtag_by_dataset": hashtag_analysis
    }
    
    output_file = Path("data/interim/analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Detailed analysis saved to: {output_file}")
    logger.info("\n✅ Analysis completed successfully!")
    
    return analysis_results

def extract_beauty_trends():
    """Extract beauty trends from processed data."""
    
    logger.info("\n🌟 BEAUTY TREND EXTRACTION")
    logger.info("=" * 60)
    
    # Load all processed data
    results_file = Path("data/interim/data_processing_results.json")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    beauty_keywords = {
        'skincare': ['skincare', 'moisturizer', 'serum', 'cleanser', 'toner', 'retinol', 'hyaluronic', 'vitamin c', 'sunscreen', 'acne'],
        'makeup': ['makeup', 'lipstick', 'foundation', 'mascara', 'eyeshadow', 'blush', 'concealer', 'highlighter', 'contour', 'bronzer'],
        'haircare': ['hair', 'shampoo', 'conditioner', 'styling', 'curly', 'straight', 'color', 'blonde', 'brunette', 'treatment'],
        'trends': ['trending', 'viral', 'popular', 'new', 'latest', 'must-have', 'obsessed', 'love', 'favorite', 'amazing']
    }
    
    trend_data = {category: Counter() for category in beauty_keywords.keys()}
    
    for file_path in results['processed_datasets']:
        if not Path(file_path).exists():
            continue
            
        df = pd.read_parquet(file_path)
        
        # Analyze text content for trends
        text_columns = ['cleaned_text', 'textOriginal'] if 'textOriginal' in df.columns else ['cleaned_text']
        
        for _, row in df.iterrows():
            text_content = ""
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    text_content += str(row[col]).lower() + " "
            
            # Count keyword mentions
            for category, keywords in beauty_keywords.items():
                for keyword in keywords:
                    if keyword in text_content:
                        trend_data[category][keyword] += 1
    
    # Report trends
    for category, keyword_counts in trend_data.items():
        if keyword_counts:
            logger.info(f"\n🔥 Top {category.title()} Trends:")
            for keyword, count in keyword_counts.most_common(10):
                logger.info(f"   • {keyword}: {count} mentions")
    
    return trend_data

if __name__ == "__main__":
    try:
        # Analyze processed data
        analysis_results = analyze_processed_data()
        
        # Extract beauty trends
        trend_results = extract_beauty_trends()
        
        print("\n🎉 Analysis completed! Check the log output above for detailed insights.")
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        raise
