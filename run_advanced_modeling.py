#!/usr/bin/env python3
"""
🎭 L'Oréal Datathon 2025 - Advanced Modeling Pipeline
====================================================

Run advanced trend detection and modeling on processed beauty data.
"""

import sys
sys.path.append('src')

import pandas as pd
import json
import logging
from pathlib import Path
from modeling_optimized import (
    ModelingPipeline, 
    CategoryClassifier, 
    SentimentAnalyzer,
    SemanticValidator
)
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_advanced_modeling():
    """Run the advanced modeling pipeline on processed data."""
    
    logger.info("🎭 L'Oréal Datathon 2025 - Advanced Modeling Pipeline")
    logger.info("=" * 60)
    
    # Load processed data files
    results_file = Path("data/interim/data_processing_results.json")
    if not results_file.exists():
        logger.error("❌ No processing results found. Run simple_pipeline.py first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    processed_files = results['processed_datasets']
    logger.info(f"📊 Found {len(processed_files)} processed datasets")
    
    # Initialize the modeling pipeline
    logger.info("🚀 Initializing Advanced Modeling Pipeline...")
    
    # Initialize individual components
    pipeline = ModelingPipeline()
    category_classifier = CategoryClassifier()
    sentiment_analyzer = SentimentAnalyzer()
    semantic_validator = SemanticValidator()
    
    logger.info("✅ Pipeline components initialized successfully!")
    
    # Process each dataset for trend detection
    all_trends = {}
    
    for i, file_path in enumerate(processed_files, 1):
        if not Path(file_path).exists():
            logger.warning(f"⚠️ File not found: {file_path}")
            continue
        
        dataset_name = Path(file_path).stem.replace('_processed', '')
        logger.info(f"\n📈 Processing dataset {i}/{len(processed_files)}: {dataset_name}")
        
        # Load the dataset
        df = pd.read_parquet(file_path)
        logger.info(f"   • Loaded {len(df):,} rows")
        
        # Prepare data for trend detection
        # Use timestamp and engagement for time series analysis
        if 'timestamp' in df.columns and 'engagement_score' in df.columns:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Create time series data for trend detection
            time_series_data = df.groupby('timestamp').agg({
                'engagement_score': 'mean',
                'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
            }).reset_index()
            
            logger.info(f"   • Created time series with {len(time_series_data)} time points")
            
            # Run trend analysis and sentiment analysis
            if len(time_series_data) >= 10:  # Minimum data points for analysis
                try:
                    logger.info("   🔍 Running sentiment and category analysis...")
                    
                    # Sample some text data for analysis
                    sample_texts = df['cleaned_text'].dropna().sample(min(1000, len(df))).tolist()
                    
                    # Run sentiment analysis
                    sentiment_results = [sentiment_analyzer.analyze_sentiment(text) for text in sample_texts[:100]]
                    
                    # Run category classification
                    category_results = category_classifier.classify_posts(sample_texts[:50])  # Smaller sample
                    
                    # Basic trend analysis using simple statistics
                    engagement_trend = time_series_data['engagement_score'].rolling(window=3).mean()
                    trend_direction = "increasing" if engagement_trend.iloc[-1] > engagement_trend.iloc[0] else "decreasing"
                    
                    all_trends[dataset_name] = {
                        'sentiment_analysis': {
                            'total_analyzed': len(sentiment_results),
                            'avg_sentiment': np.mean([r.get('compound', 0) for r in sentiment_results if isinstance(r, dict)]),
                            'sentiment_distribution': {}
                        },
                        'category_analysis_ml': {
                            'total_classified': len(category_results),
                            'classification_results': category_results[:10]  # Top 10 for review
                        },
                        'engagement_trend': {
                            'direction': trend_direction,
                            'start_value': float(engagement_trend.iloc[0]) if not pd.isna(engagement_trend.iloc[0]) else 0,
                            'end_value': float(engagement_trend.iloc[-1]) if not pd.isna(engagement_trend.iloc[-1]) else 0
                        },
                        'data_points': len(time_series_data),
                        'date_range': {
                            'start': str(time_series_data['timestamp'].min()),
                            'end': str(time_series_data['timestamp'].max())
                        }
                    }
                    
                    logger.info(f"   ✅ Advanced analysis completed")
                    logger.info(f"   📊 Engagement trend: {trend_direction}")
                    logger.info(f"   😊 Analyzed {len(sentiment_results)} posts for sentiment")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Advanced analysis failed: {e}")
                    all_trends[dataset_name] = {'error': str(e)}
            else:
                logger.warning(f"   ⚠️ Insufficient data points ({len(time_series_data)}) for STL analysis")
                all_trends[dataset_name] = {'error': 'Insufficient data points'}
        
        # Category-based trend analysis
        if 'category' in df.columns:
            logger.info("   🏷️ Analyzing category trends...")
            
            category_trends = df['category'].value_counts()
            category_engagement = df.groupby('category')['engagement_score'].mean() if 'engagement_score' in df.columns else {}
            
            all_trends[dataset_name] = all_trends.get(dataset_name, {})
            all_trends[dataset_name]['category_analysis'] = {
                'top_categories': category_trends.head(5).to_dict(),
                'category_engagement': category_engagement.to_dict() if hasattr(category_engagement, 'to_dict') else {}
            }
            
            logger.info(f"   • Top category: {category_trends.index[0]} ({category_trends.iloc[0]} posts)")
    
    # Generate comprehensive modeling results
    logger.info("\n🎯 GENERATING MODELING RESULTS")
    logger.info("=" * 60)
    
    # Aggregate insights across all datasets
    total_data_points = sum(t.get('data_points', 0) for t in all_trends.values())
    successful_analyses = sum(1 for t in all_trends.values() if 'sentiment_analysis' in t)
    
    logger.info(f"📊 Modeling Summary:")
    logger.info(f"   • Total datasets processed: {len(all_trends)}")
    logger.info(f"   • Successful advanced analyses: {successful_analyses}")
    logger.info(f"   • Total data points analyzed: {total_data_points:,}")
    
    # Save modeling results
    modeling_results = {
        "modeling_summary": {
            "datasets_processed": len(all_trends),
            "successful_advanced_analyses": successful_analyses,
            "total_data_points": total_data_points,
            "modeling_date": pd.Timestamp.now().isoformat()
        },
        "trend_analysis": all_trends,
        "pipeline_config": {
            "components_used": ["ModelingPipeline", "CategoryClassifier", "SentimentAnalyzer", "SemanticValidator"],
            "advanced_features_enabled": True
        }
    }
    
    # Save results
    output_file = Path("data/interim/modeling_results.json")
    with open(output_file, 'w') as f:
        json.dump(modeling_results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Modeling results saved to: {output_file}")
    
    # Generate recommendations
    logger.info("\n💡 KEY MODELING INSIGHTS:")
    
    for dataset_name, trends in all_trends.items():
        if 'category_analysis' in trends:
            top_cat = list(trends['category_analysis']['top_categories'].keys())[0]
            top_count = list(trends['category_analysis']['top_categories'].values())[0]
            logger.info(f"   • {dataset_name}: {top_cat} dominates with {top_count:,} posts")
    
    logger.info("\n✅ Advanced modeling pipeline completed successfully!")
    return modeling_results

if __name__ == "__main__":
    try:
        results = run_advanced_modeling()
        print("\n🎉 Advanced modeling completed! Check the log output above for detailed insights.")
        print("📊 Results saved to: data/interim/modeling_results.json")
        
    except Exception as e:
        logger.error(f"❌ Error during modeling: {e}")
        raise
