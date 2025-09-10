#!/usr/bin/env python3
"""
Test script to verify all modular imports work correctly.
"""

import sys
import importlib
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

def test_module_imports():
    """Test that all modules can be imported successfully."""
    modules_to_test = [
        'modules.config',
        'modules.text_processing', 
        'modules.data_loading',
        'modules.aggregation',
        'modules.parallel_processing',
        'modules.reporting'
    ]
    
    print("🧪 Testing modular imports...")
    print("=" * 50)
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name:<30} - SUCCESS")
            
            # Test if key functions exist
            if module_name == 'modules.config':
                assert hasattr(module, 'PERF_CONFIGS')
                assert hasattr(module, 'KEYWORD_CATEGORY')
                print(f"   📊 Performance modes available: {list(module.PERF_CONFIGS.keys())}")
                
            elif module_name == 'modules.text_processing':
                assert hasattr(module, 'clean_text_optimized')
                assert hasattr(module, 'categorize_feature')
                print(f"   🔧 Key functions: clean_text_optimized, categorize_feature, is_trend_relevant")
                
            elif module_name == 'modules.data_loading':
                assert hasattr(module, 'load_samples_optimized')
                assert hasattr(module, 'preprocess_comments_dataset')
                print(f"   📂 Key functions: load_samples_optimized, preprocess_comments_dataset")
                
            elif module_name == 'modules.aggregation':
                assert hasattr(module, 'aggregate_hashtags_optimized')
                assert hasattr(module, 'aggregate_keywords_optimized')
                print(f"   📊 Key functions: aggregate_hashtags_optimized, aggregate_keywords_optimized")
                
            elif module_name == 'modules.parallel_processing':
                assert hasattr(module, 'parallel_process_sources')
                assert hasattr(module, 'ChunkedDataProcessor')
                print(f"   ⚡ Key functions: parallel_process_sources, ChunkedDataProcessor")
                
            elif module_name == 'modules.reporting':
                assert hasattr(module, 'write_enhanced_phase2_report')
                assert hasattr(module, 'write_phase3_emerging_trends_report')
                print(f"   📝 Key functions: write_enhanced_phase2_report, write_phase3_emerging_trends_report")
            
            success_count += 1
            
        except ImportError as e:
            print(f"❌ {module_name:<30} - FAILED: {e}")
        except AssertionError as e:
            print(f"⚠️  {module_name:<30} - MISSING FUNCTIONS: {e}")
        except Exception as e:
            print(f"💥 {module_name:<30} - UNEXPECTED ERROR: {e}")
    
    print("=" * 50)
    print(f"🎯 Import test results: {success_count}/{len(modules_to_test)} modules passed")
    
    if success_count == len(modules_to_test):
        print("🎉 All modules imported successfully!")
        print("\n🚀 Ready to run: python data_processing_modular.py")
        return True
    else:
        print("❌ Some modules failed to import. Please check the errors above.")
        return False

def test_main_driver():
    """Test that the main driver can be imported."""
    try:
        import data_processing_modular
        print("\n✅ Main driver (data_processing_modular.py) imports successfully!")
        print(f"   📊 Available functions: {[f for f in dir(data_processing_modular) if not f.startswith('_')]}")
        return True
    except Exception as e:
        print(f"\n❌ Main driver import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 MODULAR PIPELINE IMPORT VERIFICATION")
    print("=" * 60)
    
    modules_ok = test_module_imports()
    driver_ok = test_main_driver()
    
    print("\n" + "=" * 60)
    if modules_ok and driver_ok:
        print("🎉 ALL TESTS PASSED - MODULAR PIPELINE READY!")
        print("\n📋 Next steps:")
        print("   1. Run: python data_processing_modular.py")
        print("   2. Watch for progress bars and enhanced logging")
        print("   3. Check reports in data/interim/")
        print("   4. Verify features in data/processed/")
    else:
        print("❌ SOME TESTS FAILED - PLEASE FIX BEFORE RUNNING")
    print("=" * 60)
