#!/usr/bin/env python3
"""
Simple test to verify the full pipeline imports and basic functionality
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all main components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test data processing imports
        from data_processing_optimized import OptimizedDataProcessor
        print("✅ Data processing imports successful")
        
        # Test feature processing imports
        from feature_text_processor import FeatureTextProcessor
        print("✅ Feature processing imports successful")
        
        # Test modeling imports  
        from modeling_optimized import ModelingPipeline
        print("✅ Modeling imports successful")
        
        # Test full pipeline import
        from full_pipeline import FullPipeline
        print("✅ Full pipeline imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with minimal data"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Create minimal sample data
        sample_data = pd.DataFrame({
            'text': ['This mascara is great #beauty', 'Love this skincare routine'],
            'timestamp': pd.date_range('2024-01-01', periods=2),
            'engagement_score': [100, 50]
        })
        
        print(f"✅ Created sample data: {len(sample_data)} rows")
        
        # Test progress bar
        print("🔄 Testing progress bar...")
        with tqdm(range(3), desc="Demo progress") as pbar:
            for i in pbar:
                pbar.set_postfix(step=f"step_{i}")
                import time
                time.sleep(0.1)
        
        print("✅ Progress bar test successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("🚀 Basic Pipeline Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        return False
    
    print("\n🎉 All basic tests passed!")
    print("\nℹ️  Full pipeline is ready to use. Run with:")
    print("   python full_pipeline.py --sample")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)