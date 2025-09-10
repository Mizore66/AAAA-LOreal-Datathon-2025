#!/usr/bin/env python3
"""
Debug script to test the categorization function.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from modules.config import KEYWORD_CATEGORY
from modules.reporting import _categorize_feature

def debug_categorization():
    """Debug the categorization function."""
    print("🔍 DEBUGGING CATEGORIZATION FUNCTION")
    print("=" * 50)
    
    # Check KEYWORD_CATEGORY structure
    print(f"📊 KEYWORD_CATEGORY has {len(KEYWORD_CATEGORY)} entries")
    print("First 10 entries:")
    for i, (keyword, category) in enumerate(list(KEYWORD_CATEGORY.items())[:10]):
        print(f"  {i+1}. '{keyword}' → '{category}'")
    
    print("\nCategories found:")
    categories = set(KEYWORD_CATEGORY.values())
    for cat in sorted(categories):
        count = sum(1 for v in KEYWORD_CATEGORY.values() if v == cat)
        print(f"  {cat}: {count} keywords")
    
    # Test some features
    test_features = [
        "shorts",
        "makeup", 
        "skincare",
        "#makeup",
        "hyaluronic acid",
        "foundation",
        "unknown_feature",
        "hair",
        "beauty"
    ]
    
    print(f"\n🧪 Testing categorization for sample features:")
    print("-" * 40)
    for feature in test_features:
        category = _categorize_feature(feature)
        print(f"'{feature}' → '{category}'")
        
    # Debug the partial matching logic
    print(f"\n🔍 Debugging 'shorts' categorization:")
    feature = "shorts"
    feature_lower = feature.lower().strip('#')
    print(f"feature_lower: '{feature_lower}'")
    
    # Check direct matches
    for category, keywords in KEYWORD_CATEGORY.items():
        if feature_lower in [k.lower() for k in keywords]:
            print(f"Direct match found: '{feature_lower}' in {category}")
            break
    else:
        print("No direct match found")
    
    # Check partial matches
    for category, keywords in KEYWORD_CATEGORY.items():
        for keyword in keywords:
            if keyword.lower() in feature_lower or feature_lower in keyword.lower():
                print(f"Partial match found: '{keyword}' ↔ '{feature_lower}' in {category}")
                break
        else:
            continue
        break
    else:
        print("No partial match found")

if __name__ == "__main__":
    debug_categorization()
