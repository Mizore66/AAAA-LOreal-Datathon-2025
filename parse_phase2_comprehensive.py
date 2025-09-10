#!/usr/bin/env python3
"""
Comprehensive Phase 2 Enhanced Features Report Parser
Extracts detailed metrics from the markdown report and converts to structured JSON.
"""

import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any

def parse_phase2_report() -> Dict[str, Any]:
    """
    Parse the Phase 2 enhanced features markdown report and extract all data.
    
    Returns:
        Dictionary containing all structured report data
    """
    report_path = "data/interim/phase2_enhanced_features_report.md"
    
    if not os.path.exists(report_path):
        print(f"❌ Report file not found: {report_path}")
        return {}
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Initialize result structure
    result = {
        "report_type": "phase2_enhanced_features",
        "generated_timestamp": None,
        "performance_mode": None,
        "overall_summary": {},
        "timeframe_comparison": [],
        "top_performers_by_timeframe": {},
        "performance_insights": {},
        "total_unique_features": 0,
        "beauty_relevant_features": [],
        "all_features": []
    }
    
    # Extract generation timestamp and performance mode
    timestamp_match = re.search(r"Generated: ([\d\-: ]+)", content)
    if timestamp_match:
        result["generated_timestamp"] = timestamp_match.group(1)
    
    performance_match = re.search(r"Performance Mode: (\w+)", content)
    if performance_match:
        result["performance_mode"] = performance_match.group(1)
    
    # Parse Overall Summary table
    overall_summary_pattern = r"## Overall Summary.*?\n(\|.*?\n\|.*?\n(?:\|.*?\n)*)"
    overall_match = re.search(overall_summary_pattern, content, re.DOTALL)
    
    if overall_match:
        table_content = overall_match.group(1)
        rows = re.findall(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", table_content)
        
        for row in rows:
            if len(row) == 4:
                feature_type = row[0].strip()
                # Skip headers and separators
                if (feature_type and 
                    feature_type != 'Feature_Type' and 
                    not feature_type.startswith(':') and 
                    not feature_type.startswith('-')):
                    
                    try:
                        total_rows = int(row[1].strip()) if row[1].strip().isdigit() else 0
                        unique_features = int(row[2].strip()) if row[2].strip().isdigit() else 0
                        timeframes = int(row[3].strip()) if row[3].strip().isdigit() else 0
                        
                        result["overall_summary"][feature_type] = {
                            "total_rows": total_rows,
                            "unique_features": unique_features,
                            "timeframes": timeframes
                        }
                    except (ValueError, TypeError):
                        continue
    
    # Parse Timeframe Comparison table
    timeframe_comparison_pattern = r"## Timeframe Comparison.*?\n(\|.*?\n\|.*?\n(?:\|.*?\n)*)"
    timeframe_match = re.search(timeframe_comparison_pattern, content, re.DOTALL)
    
    if timeframe_match:
        table_content = timeframe_match.group(1)
        rows = re.findall(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", table_content)
        
        for row in rows:
            if len(row) == 4:
                timeframe = row[0].strip()
                # Skip headers and separators
                if (timeframe and 
                    timeframe != 'timeframe' and 
                    not timeframe.startswith(':') and 
                    not timeframe.startswith('-')):
                    
                    try:
                        rows_count = int(row[1].strip()) if row[1].strip().isdigit() else 0
                        unique_features = int(row[2].strip()) if row[2].strip().isdigit() else 0
                        total_count = int(row[3].strip()) if row[3].strip().isdigit() else 0
                        
                        result["timeframe_comparison"].append({
                            "timeframe": timeframe,
                            "rows": rows_count,
                            "unique_features": unique_features,
                            "total_count": total_count
                        })
                    except (ValueError, TypeError):
                        continue
    
    # Parse Top Performers by Timeframe sections
    timeframe_sections = re.findall(r"### (\w+) Timeframe\n(.*?)(?=###|##|\Z)", content, re.DOTALL)
    
    for timeframe, section_content in timeframe_sections:
        # Extract latest period
        latest_period_match = re.search(r"Latest period: ([^\n]+)", section_content)
        latest_period = latest_period_match.group(1) if latest_period_match else None
        
        # Extract total features count
        total_features_match = re.search(r"Total features \(latest\): ([\d,]+)", section_content)
        total_features = 0
        if total_features_match:
            total_features = int(total_features_match.group(1).replace(',', ''))
        
        # Parse the features table
        table_pattern = r"\|\s*source_type\s*\|\s*feature\s*\|\s*count\s*\|\s*rolling_mean_24h\s*\|\s*delta_vs_mean\s*\|\s*category\s*\|.*?\n\|.*?\n((?:\|.*?\n)*)"
        table_match = re.search(table_pattern, section_content, re.DOTALL)
        
        features = []
        if table_match:
            table_content = table_match.group(1)
            rows = re.findall(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", table_content)
            
            for row in rows:
                if len(row) == 6:
                    source_type = row[0].strip()
                    feature = row[1].strip()
                    category = row[5].strip()
                    
                    # Skip headers and separators
                    if (source_type and feature and category and 
                        source_type != 'source_type' and 
                        feature != 'feature' and 
                        category != 'category' and
                        not feature.startswith(':') and 
                        not feature.startswith('-') and
                        not all(c in ':-' for c in feature)):
                        
                        try:
                            count = float(row[2].strip()) if row[2].strip().replace('.', '').replace('-', '').isdigit() else 0
                            rolling_mean = float(row[3].strip()) if row[3].strip().replace('.', '').replace('-', '').isdigit() else 0
                            delta_vs_mean = float(row[4].strip()) if row[4].strip().replace('.', '').replace('-', '').isdigit() else 0
                            
                            feature_data = {
                                "source_type": source_type,
                                "feature": feature,
                                "count": count,
                                "rolling_mean_24h": rolling_mean,
                                "delta_vs_mean": delta_vs_mean,
                                "category": category
                            }
                            features.append(feature_data)
                            
                            # Add to all features list (avoid duplicates)
                            if feature not in [f["feature"] for f in result["all_features"]]:
                                result["all_features"].append({
                                    "feature": feature,
                                    "category": category,
                                    "source_type": source_type,
                                    "timeframe": timeframe
                                })
                            
                            # Check if beauty-relevant
                            beauty_categories = ["Beauty", "Makeup", "Skincare", "Hair", "Fashion"]
                            if category in beauty_categories:
                                if feature not in [f["feature"] for f in result["beauty_relevant_features"]]:
                                    result["beauty_relevant_features"].append({
                                        "feature": feature,
                                        "category": category,
                                        "source_type": source_type,
                                        "timeframe": timeframe
                                    })
                                    
                        except (ValueError, TypeError):
                            continue
        
        result["top_performers_by_timeframe"][timeframe] = {
            "latest_period": latest_period,
            "total_features_latest": total_features,
            "features": features
        }
    
    # Parse Performance Insights
    insights_pattern = r"## Performance Insights\n(.*?)(?=##|\Z)"
    insights_match = re.search(insights_pattern, content, re.DOTALL)
    
    if insights_match:
        insights_content = insights_match.group(1)
        insights = {}
        
        # Extract key-value pairs
        insight_lines = re.findall(r"- ([^:]+): (.+)", insights_content)
        for key, value in insight_lines:
            insights[key.strip()] = value.strip()
        
        result["performance_insights"] = insights
    
    # Calculate totals
    result["total_unique_features"] = len(result["all_features"])
    
    return result

def main():
    """Main execution function"""
    print("Parsing Phase 2 enhanced features report...")
    
    # Parse the report
    data = parse_phase2_report()
    
    if not data:
        print("❌ Failed to parse report")
        return
    
    # Save comprehensive JSON
    output_path = "data/interim/phase2_enhanced_features_comprehensive.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Comprehensive JSON created!")
    print(f"📊 Total unique features: {data['total_unique_features']}")
    print(f"🌟 Beauty-relevant features: {len(data['beauty_relevant_features'])}")
    
    # Show categories
    categories = set()
    timeframes = set()
    for feature in data['all_features']:
        categories.add(feature['category'])
        timeframes.add(feature['timeframe'])
    
    print(f"📂 Categories: {', '.join(sorted(categories))}")
    print(f"⏰ Timeframes: {', '.join(sorted(timeframes))}")
    print(f"💾 Saved to: {output_path}")
    
    # Show sample features
    print(f"\n📋 Sample feature data:")
    for i, feature in enumerate(data['beauty_relevant_features'][:5]):
        print(f"  - {feature['feature']} ({feature['category']}): {feature['source_type']} - {feature['timeframe']}")

if __name__ == "__main__":
    main()
