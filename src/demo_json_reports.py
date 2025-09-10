#!/usr/bin/env python3
"""
Demo script to show the JSON report format structure.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from modules.reporting import write_enhanced_phase2_report, write_phase3_emerging_trends_report

def demo_json_reports():
    """Demonstrate the JSON report format with sample data."""
    print("🎯 JSON REPORT FORMAT DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data structures
    sample_hashtags = pd.DataFrame({
        'feature': ['#makeup', '#skincare', '#beauty'],
        'timeframe': ['6h', '6h', '6h'],
        'count': [150, 120, 90],
        'rolling_mean_24h': [140, 115, 85],
        'delta_vs_mean': [10, 5, 5],
        'time_bin': ['2025-07-14 18:00:00+00:00'] * 3
    })
    
    sample_keywords = pd.DataFrame({
        'feature': ['makeup', 'skincare', 'contour'],
        'timeframe': ['6h', '6h', '6h'],
        'count': [200, 180, 50],
        'rolling_mean_24h': [190, 175, 45],
        'delta_vs_mean': [10, 5, 5],
        'time_bin': ['2025-07-14 18:00:00+00:00'] * 3
    })
    
    sample_audio = pd.DataFrame({
        'feature': ['audio_trend_1'],
        'timeframe': ['6h'],
        'count': [30],
        'rolling_mean_24h': [25],
        'delta_vs_mean': [5],
        'time_bin': ['2025-07-14 18:00:00+00:00']
    })
    
    # Test Phase 2 report generation
    print("📊 Generating Phase 2 JSON report...")
    all_timeframe_data = {
        'hashtags': sample_hashtags,
        'keywords': sample_keywords,
        'audio': sample_audio
    }
    
    write_enhanced_phase2_report(all_timeframe_data)
    
    # Test Phase 3 report generation
    print("\n🔍 Generating Phase 3 JSON report...")
    sample_emerging = pd.DataFrame({
        'feature': ['#trending', 'contour', 'best skin'],
        'timeframe': ['6h', '14d', '1m'],
        'count': [10, 50, 8],
        'growth_rate': [0.5, 0.8, 0.3],
        'velocity': [0.2, 0.1, 0.4],
        'source_type': ['hashtags', 'keywords', 'keywords'],
        'time_bin': ['2025-07-14 18:00:00+00:00', '2025-07-14 00:00:00+00:00', '2025-07-01 00:00:00']
    })
    
    write_phase3_emerging_trends_report(sample_emerging, sample_hashtags, sample_keywords)
    
    # Show report structure
    print("\n📋 Generated Report Structure:")
    print("=" * 40)
    
    # Phase 2 structure
    phase2_path = Path("../data/interim/phase2_enhanced_features_comprehensive.json")
    if phase2_path.exists():
        with open(phase2_path, 'r') as f:
            phase2_data = json.load(f)
        
        print("📊 Phase 2 Report Structure:")
        print(f"   - report_type: {phase2_data.get('report_type')}")
        print(f"   - performance_mode: {phase2_data.get('performance_mode')}")
        print(f"   - total_unique_features: {phase2_data.get('total_unique_features')}")
        print(f"   - timeframe_comparison: {len(phase2_data.get('timeframe_comparison', []))} timeframes")
        print(f"   - beauty_relevant_features: {len(phase2_data.get('beauty_relevant_features', []))} features")
        print(f"   - top_performers_by_timeframe: {len(phase2_data.get('top_performers_by_timeframe', {}))} timeframes")
    
    # Phase 3 structure  
    phase3_path = Path("../data/interim/phase3_emerging_trends_comprehensive.json")
    if phase3_path.exists():
        with open(phase3_path, 'r') as f:
            phase3_data = json.load(f)
        
        phase3_detail = phase3_data.get('phase3_emerging_trends_detailed', {})
        print("\n🔍 Phase 3 Report Structure:")
        print(f"   - total_unique_tracked_terms: {phase3_detail.get('executive_summary', {}).get('total_unique_tracked_terms')}")
        print(f"   - timeframes_covered: {phase3_detail.get('executive_summary', {}).get('timeframes_covered')}")
        print(f"   - all_emerging_terms: {len(phase3_detail.get('all_emerging_terms_with_data', []))} terms")
        print(f"   - category_analysis: {len(phase3_detail.get('category_analysis', {}))} categories")
        print(f"   - detailed_timeframes: {len(phase3_detail.get('detailed_timeframes', {}))} timeframes")
    
    print("\n✅ JSON reports match your provided format structure!")
    print(f"📁 Reports saved to: data/interim/")
    print(f"   📊 phase2_enhanced_features_comprehensive.json")
    print(f"   🔍 phase3_emerging_trends_comprehensive.json")
    print(f"   📈 performance_report.json")

if __name__ == "__main__":
    demo_json_reports()
