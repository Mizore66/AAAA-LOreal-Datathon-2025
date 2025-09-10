#!/usr/bin/env python3
"""
Parse the Phase 3 markdown report and create a comprehensive JSON version
with all the detailed metrics and data from the tables.
"""

import json
import re
from pathlib import Path
from datetime import datetime

def parse_phase3_report():
    """Parse the detailed Phase 3 markdown report into comprehensive JSON."""
    
    # Read the markdown report
    report_path = Path("data/interim/phase3_emerging_trends_report.md")
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Initialize the comprehensive report structure
    report_data = {
        "phase3_emerging_trends_detailed": {
            "generation_timestamp": "2025-09-10 02:11:20",
            "executive_summary": {
                "total_unique_tracked_terms": 171,
                "emerging_terms_latest_windows": 0,
                "mean_positive_velocity": 0.44,
                "timeframes_covered": 10
            },
            "timeframe_summary": {},
            "detailed_timeframes": {},
            "all_emerging_terms_with_data": [],
            "category_analysis": {},
            "overlap_analysis": {
                "hashtags_6h_overlap": 0,
                "keywords_6h_overlap": 0
            }
        }
    }
    
    # Parse timeframe sections
    timeframes = ['1h', '3h', '6h', '1d', '3d', '7d', '14d', '1m', '3m', '6m']
    
    for timeframe in timeframes:
        # Find the timeframe section
        pattern = f"## Timeframe: {timeframe}.*?(?=## Timeframe:|## Overlap|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            section = match.group(0)
            
            # Extract latest bin
            latest_bin_match = re.search(r"Latest bin: ([^\n]+)", section)
            latest_bin = latest_bin_match.group(1) if latest_bin_match else None
            
            # Parse the top emerging table
            table_pattern = r"\| trend_type\s+\| feature\s+\| category\s+\|\s+count\s+\|\s+prev_count\s+\|\s+growth_rate\s+\|\s+velocity\s+\|.*?\n\|.*?\n((?:\|.*?\n)*)"
            table_match = re.search(table_pattern, section, re.DOTALL)
            
            emerging_terms = []
            if table_match:
                table_content = table_match.group(1)
                rows = re.findall(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", table_content)
                
                for row in rows:
                    if len(row) == 7:
                        trend_type = row[0].strip()
                        feature = row[1].strip()
                        category = row[2].strip()
                        
                        # Skip table headers, separators, and empty rows
                        if (trend_type and feature and category and 
                            not feature.startswith(':') and 
                            not feature.startswith('-') and
                            feature != 'feature' and 
                            trend_type != 'trend_type' and
                            category != 'category' and
                            not all(c in ':-' for c in feature) and
                            feature != '-'):
                            
                            try:
                                count = float(row[3].strip()) if row[3].strip().replace('.', '').isdigit() else 0
                                prev_count = float(row[4].strip()) if row[4].strip().replace('.', '').isdigit() else 0
                                growth_rate = float(row[5].strip()) if row[5].strip().replace('.', '').replace('-', '').isdigit() else 0
                                velocity = float(row[6].strip()) if row[6].strip().replace('.', '').replace('-', '').isdigit() else 0
                                
                                term_data = {
                                    "trend_type": trend_type,
                                    "feature": feature,
                                    "category": category,
                                    "count": count,
                                    "prev_count": prev_count,
                                    "growth_rate": growth_rate,
                                    "velocity": velocity
                                }
                                emerging_terms.append(term_data)
                            except (ValueError, TypeError):
                                # Skip rows with invalid numeric data
                                continue
            
            # Extract persistence data
            persistence_data = {}
            avg_bins_match = re.search(r"Avg bins per term: ([\d.]+)", section)
            percentile_bins_match = re.search(r"75th percentile bins: ([\d.]+)", section)
            accel_decel_match = re.search(r"Accelerating: (\d+) \| Decelerating: (\d+)", section)
            
            if avg_bins_match:
                persistence_data["avg_bins_per_term"] = float(avg_bins_match.group(1))
            if percentile_bins_match:
                persistence_data["percentile_75_bins"] = float(percentile_bins_match.group(1))
            if accel_decel_match:
                persistence_data["accelerating"] = int(accel_decel_match.group(1))
                persistence_data["decelerating"] = int(accel_decel_match.group(2))
            
            # Store timeframe data
            report_data["phase3_emerging_trends_detailed"]["detailed_timeframes"][timeframe] = {
                "latest_bin": latest_bin,
                "emerging_terms": emerging_terms,
                "total_emerging_terms": len(emerging_terms),
                "persistence": persistence_data
            }
            
            # Add to summary
            report_data["phase3_emerging_trends_detailed"]["timeframe_summary"][timeframe] = {
                "latest_bin": latest_bin,
                "emerging_terms_count": len(emerging_terms),
                "top_term": emerging_terms[0]["feature"] if emerging_terms else "-",
                "median_growth": sum(t["growth_rate"] for t in emerging_terms) / len(emerging_terms) if emerging_terms else 0
            }
    
    # Compile all emerging terms with their data
    all_terms = {}
    categories = {}
    
    for timeframe, data in report_data["phase3_emerging_trends_detailed"]["detailed_timeframes"].items():
        for term_data in data["emerging_terms"]:
            feature = term_data["feature"]
            category = term_data["category"]
            
            if feature not in all_terms:
                all_terms[feature] = {
                    "feature": feature,
                    "category": category,
                    "timeframes": {},
                    "total_appearances": 0,
                    "avg_growth_rate": 0,
                    "max_count": 0,
                    "is_beauty_relevant": category in ["Beauty", "Skincare", "Makeup", "Hair", "Fashion"]
                }
            
            all_terms[feature]["timeframes"][timeframe] = {
                "count": term_data["count"],
                "prev_count": term_data["prev_count"],
                "growth_rate": term_data["growth_rate"],
                "velocity": term_data["velocity"],
                "trend_type": term_data["trend_type"]
            }
            all_terms[feature]["total_appearances"] += 1
            all_terms[feature]["max_count"] = max(all_terms[feature]["max_count"], term_data["count"])
            
            # Track categories
            if category not in categories:
                categories[category] = {"terms": [], "total_count": 0}
            if feature not in categories[category]["terms"]:
                categories[category]["terms"].append(feature)
                categories[category]["total_count"] += 1
    
    # Calculate average growth rates
    for feature, data in all_terms.items():
        growth_rates = [tf_data["growth_rate"] for tf_data in data["timeframes"].values()]
        data["avg_growth_rate"] = sum(growth_rates) / len(growth_rates) if growth_rates else 0
    
    # Convert to list and sort by importance
    all_terms_list = list(all_terms.values())
    all_terms_list.sort(key=lambda x: (x["total_appearances"], x["max_count"], x["avg_growth_rate"]), reverse=True)
    
    report_data["phase3_emerging_trends_detailed"]["all_emerging_terms_with_data"] = all_terms_list
    report_data["phase3_emerging_trends_detailed"]["category_analysis"] = categories
    
    # Add metadata
    report_data["phase3_emerging_trends_detailed"]["metadata"] = {
        "total_unique_terms": len(all_terms_list),
        "beauty_relevant_terms": len([t for t in all_terms_list if t["is_beauty_relevant"]]),
        "categories_found": list(categories.keys()),
        "timeframes_analyzed": timeframes,
        "parsing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return report_data

def main():
    """Create comprehensive JSON from Phase 3 report."""
    print("Parsing Phase 3 emerging trends report...")
    
    report_data = parse_phase3_report()
    
    # Save the comprehensive JSON
    output_path = Path("data/interim/phase3_emerging_trends_comprehensive.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    metadata = report_data["phase3_emerging_trends_detailed"]["metadata"]
    print(f"✅ Comprehensive JSON created!")
    print(f"📊 Total unique terms: {metadata['total_unique_terms']}")
    print(f"🌟 Beauty-relevant terms: {metadata['beauty_relevant_terms']}")
    print(f"📂 Categories: {', '.join(metadata['categories_found'])}")
    print(f"⏰ Timeframes: {', '.join(metadata['timeframes_analyzed'])}")
    print(f"💾 Saved to: {output_path}")
    
    # Show sample of detailed data
    print("\n📋 Sample term data:")
    sample_terms = report_data["phase3_emerging_trends_detailed"]["all_emerging_terms_with_data"][:5]
    for term in sample_terms:
        print(f"  - {term['feature']} ({term['category']}): {term['total_appearances']} timeframes, max count: {term['max_count']}")

if __name__ == "__main__":
    main()
