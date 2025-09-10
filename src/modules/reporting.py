#!/usr/bin/env python3
"""
Reporting module for L'Oréal Datathon 2025
Generates comprehensive JSON reports for Phase 2 and Phase 3 analysis.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from .config import INTERIM_DIR, PERFORMANCE_MODE, CONFIG, KEYWORD_CATEGORY

logger = logging.getLogger(__name__)

def write_enhanced_phase2_report(all_timeframe_data: Dict[str, pd.DataFrame]):
    """Generate Phase 2 JSON report matching the provided format."""
    logger.info("🔄 Generating Phase 2 JSON report matching target format...")
    
    report_data = {
        "report_type": "phase2_enhanced_features",
        "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance_mode": PERFORMANCE_MODE,
        "overall_summary": {},
        "timeframe_comparison": [],
        "top_performers_by_timeframe": {},
        "performance_insights": {
            "Processing mode": PERFORMANCE_MODE,
            "Parallel processing": "Enabled" if CONFIG.get('enable_parallel', True) else "Disabled",
            "Caching": "Enabled" if CONFIG.get('enable_caching', False) else "Disabled",
            "Sample limit per source": f"{CONFIG.get('sample_size', 100000):,}" if CONFIG.get('sample_size') else "No limit"
        },
        "total_unique_features": 0,
        "beauty_relevant_features": [],
        "all_features": []
    }
    
    # Build overall summary
    for ftype, df in all_timeframe_data.items():
        if not df.empty:
            total_rows = len(df)
            unique_features = df['feature'].nunique() if 'feature' in df.columns else 0
            timeframes = df['timeframe'].nunique() if 'timeframe' in df.columns else 0
            
            report_data["overall_summary"][ftype] = {
                "total_rows": total_rows,
                "unique_features": unique_features,
                "timeframes": timeframes
            }
    
    # Build timeframe comparison
    combined_timeframe_data = []
    for ftype, df in all_timeframe_data.items():
        if not df.empty and 'timeframe' in df.columns:
            for timeframe in df['timeframe'].unique():
                tf_data = df[df['timeframe'] == timeframe]
                combined_timeframe_data.append({
                    'timeframe': timeframe,
                    'feature_type': ftype,
                    'rows': len(tf_data),
                    'unique_features': tf_data['feature'].nunique() if 'feature' in tf_data.columns else 0,
                    'total_count': tf_data['count'].sum() if 'count' in tf_data.columns else 0
                })
    
    # Aggregate by timeframe
    if combined_timeframe_data:
        tf_df = pd.DataFrame(combined_timeframe_data)
        timeframe_summary = tf_df.groupby('timeframe').agg({
            'rows': 'sum',
            'unique_features': 'max',  # Take max since features are shared across types
            'total_count': 'sum'
        }).reset_index()
        
        for _, row in timeframe_summary.iterrows():
            report_data["timeframe_comparison"].append({
                "timeframe": row['timeframe'],
                "rows": int(row['rows']),
                "unique_features": int(row['unique_features']),
                "total_count": int(row['total_count'])
            })
    
    # Build top performers by timeframe
    all_features_list = []
    beauty_features_list = []
    
    for ftype, df in all_timeframe_data.items():
        if df.empty or 'timeframe' not in df.columns:
            continue
            
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe]
            if tf_data.empty:
                continue
                
            # Sort by count and get top features
            top_features = tf_data.nlargest(15, 'count') if 'count' in tf_data.columns else tf_data.head(15)
            
            # Get latest period for this timeframe
            latest_period = None
            if 'time_bin' in tf_data.columns:
                latest_period = tf_data['time_bin'].max()
                latest_data = tf_data[tf_data['time_bin'] == latest_period]
            else:
                latest_data = tf_data
            
            timeframe_key = timeframe.upper()
            if timeframe_key not in report_data["top_performers_by_timeframe"]:
                report_data["top_performers_by_timeframe"][timeframe_key] = {
                    "latest_period": str(latest_period) if latest_period else "Unknown",
                    "total_features_latest": len(latest_data),
                    "features": []
                }
            
            # Add features to the timeframe
            for _, feature_row in top_features.head(15).iterrows():
                feature_dict = {
                    "source_type": ftype,
                    "feature": feature_row.get('feature', ''),
                    "count": float(feature_row.get('count', 0)),
                    "rolling_mean_24h": float(feature_row.get('rolling_mean_24h', 0)),
                    "delta_vs_mean": float(feature_row.get('delta_vs_mean', 0)),
                    "category": _categorize_feature(feature_row.get('feature', ''))
                }
                
                # Check if beauty relevant
                if _is_beauty_relevant_feature(feature_row.get('feature', ''), feature_dict["category"]):
                    beauty_features_list.append({
                        "feature": feature_row.get('feature', ''),
                        "category": feature_dict["category"],
                        "source_type": ftype,
                        "timeframe": timeframe_key
                    })
                
                # Add to all features
                all_features_list.append({
                    "feature": feature_row.get('feature', ''),
                    "category": feature_dict["category"], 
                    "source_type": ftype,
                    "timeframe": timeframe_key
                })
                
                report_data["top_performers_by_timeframe"][timeframe_key]["features"].append(feature_dict)
    
    # Set unique features count and lists
    report_data["total_unique_features"] = len(set(f["feature"] for f in all_features_list))
    report_data["beauty_relevant_features"] = beauty_features_list
    report_data["all_features"] = all_features_list
    
    # Write JSON report
    output_path = INTERIM_DIR / 'phase2_enhanced_features_comprehensive.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Phase 2 JSON report generated: {output_path}")
    logger.info(f"   📊 Total features: {report_data['total_unique_features']}")
    logger.info(f"   🎯 Beauty relevant: {len(beauty_features_list)}")
    logger.info(f"   ⏱️  Timeframes: {len(report_data['timeframe_comparison'])}")

def write_phase3_emerging_trends_report(emerging_df: pd.DataFrame, ts_hashtags: pd.DataFrame = None, ts_keywords: pd.DataFrame = None):
    """Generate Phase 3 JSON report matching the provided format."""
    logger.info("🔄 Generating Phase 3 emerging trends JSON report...")
    
    if emerging_df.empty:
        logger.warning("Empty emerging trends data - generating minimal report")
        report_data = {
            "phase3_emerging_trends_detailed": {
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "executive_summary": {
                    "total_unique_tracked_terms": 0,
                    "emerging_terms_latest_windows": 0,
                    "mean_positive_velocity": 0.0,
                    "timeframes_covered": 0
                },
                "timeframe_summary": {},
                "detailed_timeframes": {},
                "all_emerging_terms_with_data": [],
                "category_analysis": {},
                "overlap_analysis": {
                    "hashtags_6h_overlap": 0,
                    "keywords_6h_overlap": 0
                },
                "metadata": {
                    "total_unique_terms": 0,
                    "beauty_relevant_terms": 0,
                    "categories_found": [],
                    "timeframes_analyzed": [],
                    "parsing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        }
    else:
        # Build comprehensive emerging trends report
        timeframes = sorted(emerging_df['timeframe'].unique()) if 'timeframe' in emerging_df.columns else ['default']
        
        # Executive summary
        total_unique = emerging_df['feature'].nunique()
        mean_velocity = emerging_df['velocity'].mean() if 'velocity' in emerging_df.columns else 0.0
        
        report_data = {
            "phase3_emerging_trends_detailed": {
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "executive_summary": {
                    "total_unique_tracked_terms": total_unique,
                    "emerging_terms_latest_windows": 0,  # Will be calculated
                    "mean_positive_velocity": float(mean_velocity),
                    "timeframes_covered": len(timeframes)
                },
                "timeframe_summary": {},
                "detailed_timeframes": {},
                "all_emerging_terms_with_data": [],
                "category_analysis": {},
                "overlap_analysis": {
                    "hashtags_6h_overlap": _calculate_overlap(emerging_df, ts_hashtags, 'hashtags'),
                    "keywords_6h_overlap": _calculate_overlap(emerging_df, ts_keywords, 'keywords')
                },
                "metadata": {
                    "total_unique_terms": total_unique,
                    "beauty_relevant_terms": 0,  # Will be calculated
                    "categories_found": [],
                    "timeframes_analyzed": timeframes,
                    "parsing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        }
        
        # Process each timeframe
        for timeframe in timeframes:
            tf_data = emerging_df[emerging_df['timeframe'] == timeframe] if 'timeframe' in emerging_df.columns else emerging_df
            
            if tf_data.empty:
                continue
                
            # Latest bin for this timeframe
            latest_bin = None
            if 'time_bin' in tf_data.columns:
                latest_bin = tf_data['time_bin'].max()
                latest_data = tf_data[tf_data['time_bin'] == latest_bin]
            else:
                latest_data = tf_data
                latest_bin = "Unknown"
            
            # Get top term and median growth
            top_term = latest_data.nlargest(1, 'count')['feature'].iloc[0] if len(latest_data) > 0 and 'count' in latest_data.columns else "Unknown"
            median_growth = tf_data['growth_rate'].median() if 'growth_rate' in tf_data.columns else 0.0
            
            # Timeframe summary
            report_data["phase3_emerging_trends_detailed"]["timeframe_summary"][timeframe] = {
                "latest_bin": str(latest_bin),
                "emerging_terms_count": len(latest_data),
                "top_term": top_term,
                "median_growth": float(median_growth)
            }
            
            # Detailed timeframe data
            emerging_terms_list = []
            for _, row in tf_data.head(15).iterrows():  # Top 15 for detail
                emerging_terms_list.append({
                    "feature": row.get('feature', ''),
                    "category": _categorize_feature(row.get('feature', '')),
                    "count": float(row.get('count', 0)),
                    "growth_rate": float(row.get('growth_rate', 0)),
                    "velocity": float(row.get('velocity', 0)),
                    "source_type": row.get('source_type', 'unknown')
                })
            
            # Persistence analysis
            avg_bins = tf_data.groupby('feature').size().mean() if len(tf_data) > 0 else 0
            percentile_75_bins = tf_data.groupby('feature').size().quantile(0.75) if len(tf_data) > 0 else 0
            accelerating = len(tf_data[tf_data['velocity'] > 0]) if 'velocity' in tf_data.columns else 0
            decelerating = len(tf_data[tf_data['velocity'] <= 0]) if 'velocity' in tf_data.columns else 0
            
            report_data["phase3_emerging_trends_detailed"]["detailed_timeframes"][timeframe] = {
                "latest_bin": str(latest_bin),
                "emerging_terms": emerging_terms_list,
                "total_emerging_terms": len(tf_data),
                "persistence": {
                    "avg_bins_per_term": float(avg_bins),
                    "percentile_75_bins": float(percentile_75_bins),
                    "accelerating": accelerating,
                    "decelerating": decelerating
                }
            }
        
        # All emerging terms with data
        all_terms_data = []
        beauty_relevant_count = 0
        categories_found = set()
        
        for feature in emerging_df['feature'].unique():
            feature_data = emerging_df[emerging_df['feature'] == feature]
            category = _categorize_feature(feature)
            categories_found.add(category)
            
            is_beauty_relevant = _is_beauty_relevant_feature(feature, category)
            if is_beauty_relevant:
                beauty_relevant_count += 1
                
            # Build timeframes dict
            timeframes_dict = {}
            for _, row in feature_data.iterrows():
                tf = row.get('timeframe', 'default')
                timeframes_dict[tf] = {
                    "trend_type": "SUSTAINED" if row.get('count', 0) > 1 else "EMERGING"
                }
            
            all_terms_data.append({
                "feature": feature,
                "category": category,
                "timeframes": timeframes_dict,
                "total_appearances": len(feature_data),
                "avg_growth_rate": float(feature_data['growth_rate'].mean()) if 'growth_rate' in feature_data.columns else 0.0,
                "max_count": float(feature_data['count'].max()) if 'count' in feature_data.columns else 0.0,
                "is_beauty_relevant": is_beauty_relevant
            })
        
        report_data["phase3_emerging_trends_detailed"]["all_emerging_terms_with_data"] = all_terms_data
        report_data["phase3_emerging_trends_detailed"]["metadata"]["beauty_relevant_terms"] = beauty_relevant_count
        report_data["phase3_emerging_trends_detailed"]["metadata"]["categories_found"] = list(categories_found)
        
        # Category analysis
        category_analysis = {}
        for category in categories_found:
            category_terms = [term["feature"] for term in all_terms_data if term["category"] == category]
            category_analysis[category] = {
                "terms": category_terms[:5],  # Top 5 terms for brevity
                "total_count": len(category_terms)
            }
        
        report_data["phase3_emerging_trends_detailed"]["category_analysis"] = category_analysis
    
    # Write JSON report
    output_path = INTERIM_DIR / 'phase3_emerging_trends_comprehensive.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Phase 3 JSON report generated: {output_path}")
    logger.info(f"   📊 Total emerging terms: {report_data['phase3_emerging_trends_detailed']['metadata']['total_unique_terms']}")
    logger.info(f"   🎯 Beauty relevant: {report_data['phase3_emerging_trends_detailed']['metadata']['beauty_relevant_terms']}")
    logger.info(f"   ⏱️  Timeframes: {len(report_data['phase3_emerging_trends_detailed']['metadata']['timeframes_analyzed'])}")

def _categorize_feature(feature: str) -> str:
    """Categorize a feature using the keyword category mapping."""
    if not feature:
        return "Other"
    
    feature_lower = feature.lower().strip('#')
    
    # Check direct matches first
    if feature_lower in KEYWORD_CATEGORY:
        return KEYWORD_CATEGORY[feature_lower]
    
    # Check partial matches
    for keyword, category in KEYWORD_CATEGORY.items():
        if keyword.lower() in feature_lower or feature_lower in keyword.lower():
            return category
    
    return "Other"

def _is_beauty_relevant_feature(feature: str, category: str) -> bool:
    """Check if a feature is beauty/fashion relevant."""
    beauty_categories = {'Beauty', 'Makeup', 'Skincare', 'Hair', 'Fashion'}
    return category in beauty_categories

def _calculate_overlap(emerging_df: pd.DataFrame, baseline_df: pd.DataFrame, source_type: str) -> int:
    """Calculate overlap between emerging trends and baseline 6h data."""
    if emerging_df.empty or baseline_df is None or baseline_df.empty:
        return 0
    
    emerging_features = set(emerging_df['feature'].unique())
    baseline_features = set(baseline_df['feature'].unique()) if 'feature' in baseline_df.columns else set()
    
    return len(emerging_features.intersection(baseline_features))

def generate_performance_report(start_time: float, end_time: float, 
                               comment_sources: int, video_sources: int, 
                               total_files: int):
    """Generate performance summary JSON report."""
    duration = end_time - start_time
    
    performance_data = {
        "performance_summary": {
            "execution_time_seconds": duration,
            "execution_time_minutes": duration / 60,
            "performance_mode": PERFORMANCE_MODE,
            "data_sources": {
                "comment_sources": comment_sources,
                "video_sources": video_sources,
                "total_sources": comment_sources + video_sources
            },
            "output_files_generated": total_files,
            "configuration": CONFIG,
            "processing_features": {
                "parallel_processing": CONFIG.get('enable_parallel', True),
                "chunked_processing": CONFIG.get('enable_chunking', True),
                "caching_enabled": CONFIG.get('enable_caching', False)
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    output_path = INTERIM_DIR / 'performance_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📈 Performance report generated: {output_path}")
    logger.info(f"   ⏱️  Duration: {duration:.2f}s ({duration/60:.1f}m)")
    logger.info(f"   📁 Files: {total_files}")
    logger.info(f"   🔧 Mode: {PERFORMANCE_MODE}")
