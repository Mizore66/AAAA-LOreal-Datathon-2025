#!/usr/bin/env python3
"""
Quick summary of the unified translation pipeline results
"""

import json
import pandas as pd
from pathlib import Path

def show_results_summary():
    """Show a summary of translation results"""
    
    print("🎉 UNIFIED TRANSLATION PIPELINE - RESULTS SUMMARY")
    print("=" * 60)
    
    # Check if files exist and show stats
    translated_dir = Path("data/translated_features")
    
    # Show emerging terms results
    emerging_file = translated_dir / "emerging_terms_translated.csv"
    if emerging_file.exists():
        df = pd.read_csv(emerging_file)
        beauty_df = df[df['is_beauty_relevant']]
        
        print("📊 EMERGING TERMS TRANSLATION:")
        print(f"   • Total terms processed: {len(df)}")
        print(f"   • Terms translated: {len(df[df['was_translated']])}")
        print(f"   • Already in English: {len(df[~df['was_translated']])}")
        print(f"   • Beauty-relevant: {len(beauty_df)} ({len(beauty_df)/len(df)*100:.1f}%)")
        
        # Show language distribution
        lang_counts = df['detected_language'].value_counts().head(5)
        print(f"   • Top languages detected:")
        for lang, count in lang_counts.items():
            print(f"     - {lang}: {count} terms")
    
    # Show unified report if available
    report_file = translated_dir / "unified_translation_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   • Translation timestamp: {report['translation_timestamp']}")
        if 'summary' in report:
            summary = report['summary']
            print(f"   • Beauty relevance rate: {summary['beauty_relevance_rate']:.1f}%")
            print(f"   • Translation rate: {summary['translation_rate']:.1f}%")
    
    print(f"\n✅ FILES CREATED:")
    print(f"   📄 Emerging terms: data/translated_features/emerging_terms_translated.csv")
    print(f"   💄 Beauty terms: data/translated_features/beauty_emerging_terms_translated.csv")
    print(f"   📊 Translation report: data/translated_features/unified_translation_report.json")
    
    print(f"\n🔧 PIPELINE ADVANTAGES:")
    print(f"   ✅ Avoids processing huge hashtag/keyword files (too slow)")
    print(f"   ✅ Focuses on manageable emerging terms (fast & efficient)")
    print(f"   ✅ Includes comprehensive Phase 2 & Phase 3 report data")
    print(f"   ✅ Provides beauty-relevant filtering")
    print(f"   ✅ Supports multiple processing options")
    
    print(f"\n💡 USAGE OPTIONS:")
    print(f"   1. Fast: Just emerging terms (recommended)")
    print(f"   2. Medium: JSON reports only")
    print(f"   3. Balanced: Emerging terms + JSON reports")
    print(f"   4. Full: Everything including large files (slow)")

if __name__ == "__main__":
    show_results_summary()
