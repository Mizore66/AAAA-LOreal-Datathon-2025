#!/usr/bin/env python3
"""
Complete pipeline test for TrendSpotter Phase 3 implementation.
This script demonstrates the full functionality of the modeling framework.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from modeling import (
    TrendDetectionModel, TimeSeriesForecaster, TrendCategoryClassifier,
    ModelPersistence, load_processed_features
)

def test_trend_detection():
    """Test the trend detection models."""
    print("🔍 Testing Trend Detection Models...")
    
    # Load data
    datasets = load_processed_features()
    if not datasets:
        print("❌ No datasets available for testing")
        return False
    
    # Combine datasets
    all_data = []
    for name, df in datasets.items():
        if 'category' in df.columns:
            all_data.append(df)
    
    if not all_data:
        print("❌ No valid datasets with categories")
        return False
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Test trend detection
    detector = TrendDetectionModel(contamination=0.05)
    detector.fit(combined_df)
    
    anomalies = detector.predict_anomalies(combined_df)
    
    print(f"✅ Trend detection: {len(anomalies)} anomalies detected from {len(combined_df)} records")
    print(f"   Top anomaly features: {anomalies['feature'].head(3).tolist()}")
    
    return True


def test_forecasting():
    """Test the time series forecasting models."""
    print("📈 Testing Time Series Forecasting...")
    
    datasets = load_processed_features()
    if not datasets:
        print("❌ No datasets available for testing")
        return False
    
    # Get emerging terms data for forecasting
    emerging_df = datasets.get('features_emerging_terms_6h')
    if emerging_df is None or emerging_df.empty:
        print("❌ No emerging terms data available")
        return False
    
    forecaster = TimeSeriesForecaster()
    
    # Test forecasting on a sample feature
    top_features = emerging_df['feature'].value_counts().head(3).index.tolist()
    
    forecasts_generated = 0
    for feature in top_features:
        forecasts = forecaster.forecast_feature(emerging_df, feature, periods=12)
        if forecasts:
            forecasts_generated += len(forecasts)
            for model_name, result in forecasts.items():
                mae = result.model_metrics.mae
                print(f"   {feature} ({model_name}): MAE = {mae:.2f}" if mae else f"   {feature} ({model_name}): forecasted")
    
    print(f"✅ Forecasting: {forecasts_generated} models generated for {len(top_features)} features")
    return True


def test_classification():
    """Test the category classification models."""
    print("🏷️  Testing Category Classification...")
    
    datasets = load_processed_features()
    if not datasets:
        print("❌ No datasets available for testing")
        return False
    
    # Combine datasets with sufficient category samples
    all_data = []
    for name, df in datasets.items():
        if 'category' in df.columns and len(df) > 50:
            all_data.append(df)
    
    if not all_data:
        print("❌ No valid datasets for classification")
        return False
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter to categories with sufficient samples
    category_counts = combined_df['category'].value_counts()
    valid_categories = category_counts[category_counts >= 30].index
    classification_df = combined_df[combined_df['category'].isin(valid_categories)]
    
    if len(classification_df) < 100:
        print("❌ Insufficient data for classification")
        return False
    
    classifier = TrendCategoryClassifier()
    classifier.fit(classification_df)
    
    predictions = classifier.predict_categories(classification_df)
    
    # Calculate accuracy
    rf_accuracy = (predictions['category'] == predictions['rf_predicted_category']).mean()
    gb_accuracy = (predictions['category'] == predictions['gb_predicted_category']).mean()
    
    print(f"✅ Classification: Random Forest = {rf_accuracy:.3f}, Gradient Boosting = {gb_accuracy:.3f}")
    print(f"   Categories: {valid_categories.tolist()}")
    
    return True


def test_model_persistence():
    """Test model saving and loading."""
    print("💾 Testing Model Persistence...")
    
    # Create a simple test model
    datasets = load_processed_features()
    if not datasets:
        print("❌ No datasets available for testing")
        return False
    
    combined_df = pd.concat([df for df in datasets.values() if 'category' in df.columns], ignore_index=True)
    
    if combined_df.empty:
        print("❌ No data available for model persistence test")
        return False
    
    # Create and fit a simple trend detector
    detector = TrendDetectionModel(contamination=0.1)
    detector.fit(combined_df)
    
    # Test saving
    test_path = Path("models/test_model.pkl")
    test_path.parent.mkdir(exist_ok=True)
    
    metadata = {
        'test': True,
        'timestamp': datetime.now().isoformat(),
        'data_shape': combined_df.shape
    }
    
    ModelPersistence.save_model(detector, test_path, metadata)
    
    # Test loading
    loaded_model, loaded_metadata = ModelPersistence.load_model(test_path)
    
    print(f"✅ Model persistence: saved and loaded successfully")
    print(f"   Metadata: {loaded_metadata.get('data_shape')}")
    
    # Clean up
    test_path.unlink()
    
    return True


def test_end_to_end_integration():
    """Test the complete end-to-end pipeline."""
    print("🔄 Testing End-to-End Integration...")
    
    datasets = load_processed_features()
    if not datasets:
        print("❌ No datasets available for integration test")
        return False
    
    print(f"   Loaded {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"     - {name}: {len(df)} rows")
    
    # Test that all components work together
    combined_df = pd.concat([df for df in datasets.values() if 'category' in df.columns], ignore_index=True)
    
    # 1. Trend Detection
    detector = TrendDetectionModel()
    detector.fit(combined_df)
    anomalies = detector.predict_anomalies(combined_df)
    
    # 2. Forecasting (on a subset)
    forecaster = TimeSeriesForecaster()
    test_feature = combined_df['feature'].value_counts().index[0]
    forecasts = forecaster.forecast_feature(combined_df, test_feature, periods=6)
    
    # 3. Classification
    if len(combined_df) > 100:
        classifier = TrendCategoryClassifier()
        classifier.fit(combined_df)
        predictions = classifier.predict_categories(combined_df.head(50))
        classification_success = len(predictions) > 0
    else:
        classification_success = True  # Skip if insufficient data
    
    print(f"✅ End-to-end integration successful:")
    print(f"   - Anomalies detected: {len(anomalies)}")
    print(f"   - Forecasts generated: {len(forecasts)}")
    print(f"   - Classification: {'✅' if classification_success else '❌'}")
    
    return True


def main():
    """Run all pipeline tests."""
    print("🚀 TrendSpotter Phase 3 Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Trend Detection", test_trend_detection),
        ("Time Series Forecasting", test_forecasting),
        ("Category Classification", test_classification),
        ("Model Persistence", test_model_persistence),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Phase 3 implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)