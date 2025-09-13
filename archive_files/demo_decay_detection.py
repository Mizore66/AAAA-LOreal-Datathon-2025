#!/usr/bin/env python3
"""
Decay Detection Demo for L'Oréal Datathon 2025
Demonstrates the decay detection functionality implemented in modeling_optimized.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from modeling_optimized import DecayDetector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_trend_data():
    """Create sample trend data to demonstrate decay detection."""
    
    # Create sample time series data
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    
    # Feature 1: Decaying trend (growth > 0, acceleration < 0) - STRONG DECAY PATTERN
    decaying_trend = [10, 25, 45, 70, 90, 105, 115, 120, 122, 123, 123.5, 123.8, 123.9, 124, 124, 124, 124, 124, 124, 124]
    
    # Feature 2: Accelerating trend (growth > 0, acceleration > 0)
    accelerating_trend = [5, 8, 12, 18, 26, 36, 48, 62, 78, 96, 116, 138, 162, 188, 216, 246, 278, 312, 348, 386]
    
    # Feature 3: Declining trend (growth < 0)
    declining_trend = [100, 95, 89, 82, 74, 65, 55, 44, 32, 19, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2]
    
    # Feature 4: Stable trend
    stable_trend = [50, 51, 49, 52, 48, 50, 51, 49, 50, 52, 49, 51, 50, 49, 52, 48, 50, 51, 49, 50]
    
    # Create DataFrame
    trend_data = []
    
    features = [
        ('skincare_serum', decaying_trend),
        ('makeup_foundation', accelerating_trend), 
        ('hair_product', declining_trend),
        ('beauty_routine', stable_trend)
    ]
    
    for feature_name, counts in features:
        for i, (date, count) in enumerate(zip(dates, counts)):
            trend_data.append({
                'feature': feature_name,
                'timestamp': date,
                'time_bin': date,
                'count': count
            })
    
    return pd.DataFrame(trend_data)

def demo_decay_detection():
    """Demonstrate decay detection functionality."""
    
    logger.info("🔬 DECAY DETECTION DEMONSTRATION")
    logger.info("=" * 50)
    
    # Create sample data
    logger.info("📊 Creating sample trend data...")
    trend_data = create_sample_trend_data()
    
    logger.info(f"Created data for {trend_data['feature'].nunique()} features over {trend_data['timestamp'].nunique()} time periods")
    
    # Initialize decay detector
    logger.info("🔧 Initializing DecayDetector...")
    decay_detector = DecayDetector(period_threshold=3)
    
    # Run decay detection
    logger.info("🔍 Running decay detection analysis...")
    results = decay_detector.detect_decay(trend_data, period_T=3)
    
    # Display results
    logger.info("📈 DECAY DETECTION RESULTS:")
    logger.info("=" * 50)
    
    for feature in results['feature'].unique():
        feature_data = results[results['feature'] == feature].iloc[0]
        
        logger.info(f"\n🏷️  Feature: {feature}")
        logger.info(f"   📊 Trend State: {feature_data.get('trend_state', 'N/A')}")
        logger.info(f"   🎯 Decay Confidence: {feature_data.get('decay_confidence', 0):.3f}")
        logger.info(f"   📈 Avg Growth Rate: {feature_data.get('avg_growth_rate', 0):.3f}")
        logger.info(f"   ⚡ Avg Acceleration: {feature_data.get('avg_acceleration', 0):.3f}")
        
        # Explain the trend state
        trend_state = feature_data.get('trend_state', 'Unknown')
        if trend_state == 'Decaying':
            logger.info(f"   💡 Analysis: Growing but slowing down (growth > 0, acceleration < 0)")
        elif trend_state == 'Accelerating':
            logger.info(f"   💡 Analysis: Growing and speeding up (growth > 0, acceleration > 0)")
        elif trend_state == 'Declining':
            logger.info(f"   💡 Analysis: Decreasing engagement (growth < 0)")
        elif trend_state == 'Stable':
            logger.info(f"   💡 Analysis: Minimal change, consistent engagement")
        elif trend_state == 'Growing':
            logger.info(f"   💡 Analysis: Steady growth with mixed acceleration")
    
    # Show the rule implementation
    logger.info(f"\n📋 DECAY DETECTION RULE (from steps.md):")
    logger.info(f"   Rule: if growth_rate > 0 and acceleration < 0 for period T")
    logger.info(f"   Then: trend_state = 'Decaying'")
    logger.info(f"   Period T used: 3 time periods")
    
    # Save detailed results
    output_file = "data/interim/decay_detection_demo_results.json"
    results_summary = {
        'demo_timestamp': datetime.now().isoformat(),
        'period_T': 3,
        'features_analyzed': len(results['feature'].unique()),
        'decay_rule': "if growth_rate > 0 and acceleration < 0 for period T: then trend_state = 'Decaying'",
        'results_by_feature': {}
    }
    
    for feature in results['feature'].unique():
        feature_data = results[results['feature'] == feature].iloc[0]
        results_summary['results_by_feature'][feature] = {
            'trend_state': feature_data.get('trend_state', 'Unknown'),
            'decay_confidence': float(feature_data.get('decay_confidence', 0)),
            'avg_growth_rate': float(feature_data.get('avg_growth_rate', 0)),
            'avg_acceleration': float(feature_data.get('avg_acceleration', 0)),
            'interpretation': {
                'Decaying': 'Growing but slowing down (growth > 0, acceleration < 0)',
                'Accelerating': 'Growing and speeding up (growth > 0, acceleration > 0)', 
                'Declining': 'Decreasing engagement (growth < 0)',
                'Stable': 'Minimal change, consistent engagement',
                'Growing': 'Steady growth with mixed acceleration'
            }.get(feature_data.get('trend_state', 'Unknown'), 'Unknown trend pattern')
        }
    
    # Save results
    import json
    os.makedirs("data/interim", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n💾 Detailed results saved to: {output_file}")
    logger.info("✅ Decay detection demonstration completed successfully!")
    
    return results

if __name__ == "__main__":
    demo_decay_detection()
