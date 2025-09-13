#!/usr/bin/env python3
"""
Simple test to verify the final submission structure and basic functionality
"""

import sys
from pathlib import Path

def test_file_structure():
    """Test that all required files are present"""
    print("🧪 Testing final submission structure...")
    
    required_files = [
        'EDA_and_Model_Training.ipynb',
        'README.md', 
        'requirements.txt',
        'src/full_pipeline.py',
        'src/data_processing_optimized.py',
        'src/feature_text_processor.py',
        'src/modeling_optimized.py',
        'docs/FULL_PIPELINE_README.md',
        'docs/steps.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_imports():
    """Test that core modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Test imports
        from data_processing_optimized import OptimizedDataProcessor
        print("✅ OptimizedDataProcessor imported successfully")
        
        from feature_text_processor import FeatureTextProcessor  
        print("✅ FeatureTextProcessor imported successfully")
        
        from modeling_optimized import ModelingPipeline
        print("✅ ModelingPipeline imported successfully")
        
        from full_pipeline import FullPipeline
        print("✅ FullPipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_notebook():
    """Test that the notebook exists and has content"""
    print("\n🧪 Testing Jupyter notebook...")
    
    notebook_path = Path('EDA_and_Model_Training.ipynb')
    if not notebook_path.exists():
        print("❌ Notebook not found")
        return False
    
    # Check notebook size
    size = notebook_path.stat().st_size
    if size < 10000:  # Less than 10KB
        print("❌ Notebook appears to be empty or very small")
        return False
    
    print(f"✅ Notebook found with {size:,} bytes")
    return True

def main():
    """Run all tests"""
    print("🔬 FINAL SUBMISSION VALIDATION")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports, 
        test_notebook
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Final submission is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before submission.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)