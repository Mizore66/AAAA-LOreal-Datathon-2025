#!/usr/bin/env python3
"""
Summary of Translation Results
Shows key examples of translated features from both Phase 2 and Phase 3 reports.
"""

import json
import os
from pathlib import Path

def show_translation_examples():
    """Display translation examples from both reports"""
    
    print("🌍 COMPREHENSIVE REPORTS TRANSLATION SUMMARY")
    print("=" * 60)
    
    # Phase 2 results
    phase2_file = "data/interim/phase2_enhanced_features_comprehensive_translated.json"
    if os.path.exists(phase2_file):
        with open(phase2_file, 'r', encoding='utf-8') as f:
            phase2_data = json.load(f)
        
        print("\n📊 PHASE 2 ENHANCED FEATURES REPORT:")
        print("-" * 40)
        
        if 'translation_metadata' in phase2_data:
            stats = phase2_data['translation_metadata']['translation_stats']
            print(f"✅ Total features processed: {stats['total_features']}")
            print(f"🔄 Features translated: {stats['translated_features']}")
            print(f"🇺🇸 Already in English: {stats['english_features']}")
            print(f"❌ Failed translations: {stats['failed_translations']}")
            
            # Show some translation examples
            print(f"\n📋 Translation Examples:")
            translation_log = phase2_data['translation_metadata']['translation_log']
            examples_shown = 0
            for entry in translation_log:
                if entry['was_translated'] and entry['original'] != entry['translated'] and examples_shown < 5:
                    print(f"  • '{entry['original']}' ({entry['language']}) → '{entry['translated']}'")
                    examples_shown += 1
    
    # Phase 3 results
    phase3_file = "data/interim/phase3_emerging_trends_comprehensive_translated.json"
    if os.path.exists(phase3_file):
        with open(phase3_file, 'r', encoding='utf-8') as f:
            phase3_data = json.load(f)
        
        print("\n🔍 PHASE 3 EMERGING TRENDS REPORT:")
        print("-" * 40)
        
        if 'phase3_emerging_trends_detailed' in phase3_data:
            phase3_detailed = phase3_data['phase3_emerging_trends_detailed']
            
            if 'translation_metadata' in phase3_detailed:
                stats = phase3_detailed['translation_metadata']['translation_stats']
                print(f"✅ Total features processed: {stats['total_features']}")
                print(f"🔄 Features translated: {stats['translated_features']}")
                print(f"🇺🇸 Already in English: {stats['english_features']}")
                print(f"❌ Failed translations: {stats['failed_translations']}")
                
                # Show some translation examples
                print(f"\n📋 Translation Examples:")
                translation_log = phase3_detailed['translation_metadata']['translation_log']
                examples_shown = 0
                for entry in translation_log:
                    if entry['was_translated'] and entry['original'] != entry['translated'] and examples_shown < 5:
                        print(f"  • '{entry['original']}' ({entry['language']}) → '{entry['translated']}'")
                        examples_shown += 1
    
    print("\n📁 FILES CREATED:")
    print("-" * 20)
    print("✅ Phase 2 Translated: data/interim/phase2_enhanced_features_comprehensive_translated.json")
    print("✅ Phase 3 Translated: data/interim/phase3_emerging_trends_comprehensive_translated.json")
    
    print("\n🔍 KEY FEATURES:")
    print("-" * 20)
    print("• All feature names translated to English")
    print("• Original feature names preserved in 'feature_original' field")
    print("• Translation status tracked with 'was_translated' field")
    print("• Detected source language stored in 'detected_language' field")
    print("• Complete translation logs and statistics included")
    print("• All original structure and data preserved")
    
    print("\n💡 USAGE:")
    print("-" * 10)
    print("Use these translated files for:")
    print("• English-language analysis and reporting")
    print("• Cross-language trend comparison")
    print("• International beauty trend identification")
    print("• Standardized feature naming across datasets")

if __name__ == "__main__":
    show_translation_examples()
