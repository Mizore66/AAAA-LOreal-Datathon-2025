#!/usr/bin/env python3
"""
Real Data Term-Level Decay Detection Analysis
Analyzes actual temporal TF-IDF patterns from the enhanced modeling results
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealDataTermDecayAnalyzer:
    """Analyzes decay patterns using actual temporal TF-IDF data"""
    
    def __init__(self):
        self.results = {}
        
    def load_enhanced_results(self, filepath):
        """Load the enhanced modeling results"""
        logger.info("📊 Loading enhanced modeling results...")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_real_term_timeseries(self, temporal_trends):
        """Extract actual time series data for each term"""
        logger.info("🔍 Extracting real term time series from temporal data...")
        
        term_timeseries = defaultdict(lambda: defaultdict(list))
        all_windows = set()
        
        # First pass: collect all time windows
        for category, data in temporal_trends.items():
            trending_terms_by_window = data.get('trending_terms_by_window', {})
            all_windows.update(trending_terms_by_window.keys())
        
        # Sort windows chronologically
        sorted_windows = sorted(all_windows)
        logger.info(f"📅 Found {len(sorted_windows)} time windows from {min(sorted_windows)} to {max(sorted_windows)}")
        
        # Second pass: build complete time series for each term
        for category, data in temporal_trends.items():
            trending_terms_by_window = data.get('trending_terms_by_window', {})
            
            # Collect all terms that appear in this category
            all_terms = set()
            for window_terms in trending_terms_by_window.values():
                for term_data in window_terms:
                    if isinstance(term_data, dict) and term_data.get('term'):
                        all_terms.add(term_data['term'])
            
            logger.info(f"    📂 {category}: Found {len(all_terms)} unique terms")
            
            # Build complete time series for each term
            for term in all_terms:
                term_scores = []
                
                for window in sorted_windows:
                    window_terms = trending_terms_by_window.get(window, [])
                    
                    # Find this term's score in this window
                    score = 0.0
                    for term_data in window_terms:
                        if isinstance(term_data, dict) and term_data.get('term') == term:
                            score = term_data.get('tfidf_score', 0.0)
                            break
                    
                    term_scores.append(score)
                
                term_timeseries[category][term] = {
                    'scores': term_scores,
                    'windows': sorted_windows,
                    'non_zero_count': sum(1 for s in term_scores if s > 0),
                    'max_score': max(term_scores) if term_scores else 0,
                    'total_score': sum(term_scores)
                }
        
        return term_timeseries
    
    def detect_term_decay(self, time_series_values, period_T=3):
        """
        Decay detection for actual time series data
        """
        if len(time_series_values) < period_T + 2:
            return {
                'trend_state': 'Insufficient_Data',
                'decay_confidence': 0.0,
                'avg_growth_rate': 0.0,
                'avg_acceleration': 0.0,
                'analysis_windows': len(time_series_values)
            }
        
        # Convert to numpy array
        values = np.array(time_series_values)
        
        # Calculate first derivative (growth rate)
        growth_rates = np.diff(values)
        
        # Calculate second derivative (acceleration)
        accelerations = np.diff(growth_rates)
        
        if len(accelerations) == 0:
            return {
                'trend_state': 'Insufficient_Data',
                'decay_confidence': 0.0,
                'avg_growth_rate': 0.0,
                'avg_acceleration': 0.0,
                'analysis_windows': len(time_series_values)
            }
        
        # Focus on the last period_T values for recent trend analysis
        recent_growth = growth_rates[-period_T:] if len(growth_rates) >= period_T else growth_rates
        recent_acceleration = accelerations[-period_T:] if len(accelerations) >= period_T else accelerations
        
        avg_growth_rate = np.mean(recent_growth) if len(recent_growth) > 0 else 0.0
        avg_acceleration = np.mean(recent_acceleration) if len(recent_acceleration) > 0 else 0.0
        
        # Apply decay detection rule: if growth_rate > 0 and acceleration < 0
        is_decaying = avg_growth_rate > 0 and avg_acceleration < 0
        
        # Calculate decay confidence
        decay_periods = 0
        total_periods = min(period_T, len(accelerations))
        
        for i in range(total_periods):
            growth_idx = -(i+1)
            accel_idx = -(i+1)
            if (growth_idx >= -len(growth_rates) and accel_idx >= -len(accelerations) and
                growth_rates[growth_idx] > 0 and accelerations[accel_idx] < 0):
                decay_periods += 1
        
        decay_confidence = decay_periods / total_periods if total_periods > 0 else 0.0
        
        # Determine trend state with appropriate thresholds for TF-IDF data
        threshold = 1e-6  # Very small threshold for TF-IDF scores
        
        if is_decaying and decay_confidence > 0:
            trend_state = 'Decaying'
        elif avg_growth_rate > threshold and avg_acceleration > threshold:
            trend_state = 'Accelerating'
        elif avg_growth_rate > threshold and abs(avg_acceleration) <= threshold:
            trend_state = 'Growing'
        elif abs(avg_growth_rate) <= threshold:
            trend_state = 'Stable'
        else:
            trend_state = 'Declining'
        
        return {
            'trend_state': trend_state,
            'decay_confidence': decay_confidence,
            'avg_growth_rate': float(avg_growth_rate),
            'avg_acceleration': float(avg_acceleration),
            'analysis_windows': len(time_series_values),
            'non_zero_windows': np.count_nonzero(values),
            'peak_score': float(np.max(values)),
            'latest_score': float(values[-1]) if len(values) > 0 else 0.0
        }
    
    def analyze_real_term_decay(self, term_timeseries, min_appearances=3):
        """Analyze decay patterns in real term data"""
        logger.info(f"🔬 Analyzing decay patterns for terms with ≥{min_appearances} appearances...")
        
        results = {}
        
        for category, terms in term_timeseries.items():
            logger.info(f"\n📊 Processing {category} category...")
            category_results = {}
            
            decaying_count = 0
            accelerating_count = 0
            total_analyzed = 0
            
            for term, term_data in terms.items():
                if term_data['non_zero_count'] >= min_appearances:
                    total_analyzed += 1
                    
                    # Analyze decay pattern
                    decay_info = self.detect_term_decay(term_data['scores'])
                    
                    category_results[term] = {
                        **decay_info,
                        'appearances': term_data['non_zero_count'],
                        'max_tfidf': term_data['max_score'],
                        'total_tfidf': term_data['total_score']
                    }
                    
                    if decay_info['trend_state'] == 'Decaying':
                        decaying_count += 1
                        logger.info(f"    🔍 {term}: DECAYING (confidence: {decay_info['decay_confidence']:.3f}, growth: {decay_info['avg_growth_rate']:.6f}, accel: {decay_info['avg_acceleration']:.6f})")
                    elif decay_info['trend_state'] == 'Accelerating':
                        accelerating_count += 1
                        if accelerating_count <= 3:  # Show first few
                            logger.info(f"    🚀 {term}: ACCELERATING (growth: {decay_info['avg_growth_rate']:.6f}, accel: {decay_info['avg_acceleration']:.6f})")
            
            results[category] = category_results
            logger.info(f"    📈 Found {decaying_count} decaying and {accelerating_count} accelerating terms out of {total_analyzed} analyzed")
        
        return results
    
    def generate_detailed_insights(self, decay_results):
        """Generate detailed insights from real data analysis"""
        logger.info("\n📋 Generating detailed insights from real data...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'decay_rule_applied': "if growth_rate > 0 and acceleration < 0 for period T: then trend_state = 'Decaying'",
            'data_source': 'actual_temporal_tfidf_scores',
            'categories_analyzed': len(decay_results),
            'detailed_category_insights': {}
        }
        
        total_decaying = 0
        total_terms = 0
        
        for category, terms in decay_results.items():
            if not terms:
                continue
                
            trend_states = [term_data['trend_state'] for term_data in terms.values()]
            decaying_terms = [(term, data) for term, data in terms.items() if data['trend_state'] == 'Decaying']
            accelerating_terms = [(term, data) for term, data in terms.items() if data['trend_state'] == 'Accelerating']
            
            total_decaying += len(decaying_terms)
            total_terms += len(terms)
            
            # Sort by decay confidence and TF-IDF scores
            decaying_by_confidence = sorted(decaying_terms, key=lambda x: x[1]['decay_confidence'], reverse=True)
            decaying_by_impact = sorted(decaying_terms, key=lambda x: x[1]['max_tfidf'], reverse=True)
            accelerating_by_impact = sorted(accelerating_terms, key=lambda x: x[1]['max_tfidf'], reverse=True)
            
            summary['detailed_category_insights'][category] = {
                'total_terms_analyzed': len(terms),
                'decaying_terms_count': len(decaying_terms),
                'accelerating_terms_count': len(accelerating_terms),
                'decay_percentage': (len(decaying_terms) / len(terms)) * 100 if terms else 0,
                'trend_state_distribution': {
                    state: trend_states.count(state) 
                    for state in ['Decaying', 'Accelerating', 'Growing', 'Stable', 'Declining']
                },
                'top_decaying_by_confidence': [
                    {
                        'term': term,
                        'decay_confidence': data['decay_confidence'],
                        'max_tfidf': data['max_tfidf'],
                        'avg_growth_rate': data['avg_growth_rate'],
                        'avg_acceleration': data['avg_acceleration'],
                        'appearances': data['appearances']
                    }
                    for term, data in decaying_by_confidence[:10]
                ],
                'high_impact_decaying_terms': [
                    {
                        'term': term,
                        'max_tfidf': data['max_tfidf'],
                        'decay_confidence': data['decay_confidence'],
                        'latest_score': data['latest_score']
                    }
                    for term, data in decaying_by_impact[:5]
                ],
                'top_accelerating_terms': [
                    {
                        'term': term,
                        'max_tfidf': data['max_tfidf'],
                        'avg_growth_rate': data['avg_growth_rate'],
                        'avg_acceleration': data['avg_acceleration']
                    }
                    for term, data in accelerating_by_impact[:5]
                ]
            }
        
        summary['overall_statistics'] = {
            'total_terms_analyzed': total_terms,
            'total_decaying_terms': total_decaying,
            'overall_decay_rate': (total_decaying / total_terms) * 100 if total_terms > 0 else 0
        }
        
        return summary
    
    def save_real_analysis_results(self, decay_results, summary, output_path):
        """Save the real data analysis results"""
        logger.info(f"💾 Saving real data analysis to {output_path}...")
        
        complete_results = {
            'real_data_term_decay_analysis': decay_results,
            'detailed_insights': summary,
            'methodology': {
                'decay_detection_rule': "if growth_rate > 0 and acceleration < 0 for period T: then trend_state = 'Decaying'",
                'period_T': 3,
                'minimum_appearances': 3,
                'data_source': 'temporal_tfidf_scores_from_enhanced_modeling_results',
                'analysis_scope': 'individual_terms_within_categories_real_data'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    def print_real_data_insights(self, summary):
        """Print insights from real data analysis"""
        logger.info("\n🔬 REAL DATA TERM-LEVEL DECAY DETECTION RESULTS")
        logger.info("=" * 70)
        
        overall = summary['overall_statistics']
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"    📈 Total terms analyzed: {overall['total_terms_analyzed']}")
        logger.info(f"    🔍 Total decaying terms found: {overall['total_decaying_terms']}")
        logger.info(f"    📊 Overall decay rate: {overall['overall_decay_rate']:.2f}%")
        
        for category, insights in summary['detailed_category_insights'].items():
            logger.info(f"\n📂 {category.upper()} CATEGORY:")
            logger.info(f"    📈 Terms analyzed: {insights['total_terms_analyzed']}")
            logger.info(f"    🔍 Decaying terms: {insights['decaying_terms_count']} ({insights['decay_percentage']:.1f}%)")
            logger.info(f"    🚀 Accelerating terms: {insights['accelerating_terms_count']}")
            
            # Show trend state distribution
            dist = insights['trend_state_distribution']
            logger.info(f"    📊 States: Decay({dist['Decaying']}) | Accel({dist['Accelerating']}) | Grow({dist['Growing']}) | Stable({dist['Stable']}) | Decline({dist['Declining']})")
            
            # Show top decaying terms
            if insights['top_decaying_by_confidence']:
                logger.info(f"    🔍 Top decaying terms:")
                for i, term_info in enumerate(insights['top_decaying_by_confidence'][:3], 1):
                    logger.info(f"        {i}. '{term_info['term']}' (confidence: {term_info['decay_confidence']:.3f}, max TF-IDF: {term_info['max_tfidf']:.4f})")
            
            # Show top accelerating terms  
            if insights['top_accelerating_terms']:
                logger.info(f"    🚀 Top accelerating terms:")
                for i, term_info in enumerate(insights['top_accelerating_terms'][:3], 1):
                    logger.info(f"        {i}. '{term_info['term']}' (max TF-IDF: {term_info['max_tfidf']:.4f}, growth: {term_info['avg_growth_rate']:.6f})")
    
    def run_real_analysis(self, input_path, output_path=None):
        """Run the complete real data term decay analysis"""
        if output_path is None:
            output_path = "data/interim/real_term_decay_analysis_results.json"
        
        logger.info("🔬 STARTING REAL DATA TERM-LEVEL DECAY DETECTION")
        logger.info("=" * 70)
        
        # Load data
        enhanced_results = self.load_enhanced_results(input_path)
        temporal_trends = enhanced_results.get('temporal_trends', {})
        
        # Extract real term time series
        term_timeseries = self.extract_real_term_timeseries(temporal_trends)
        
        # Analyze decay patterns in real data
        decay_results = self.analyze_real_term_decay(term_timeseries)
        
        # Generate detailed insights
        summary = self.generate_detailed_insights(decay_results)
        
        # Save results
        self.save_real_analysis_results(decay_results, summary, output_path)
        
        # Print insights
        self.print_real_data_insights(summary)
        
        logger.info(f"\n✅ Real data analysis complete! Results saved to: {output_path}")
        
        return decay_results, summary

def main():
    """Main execution function"""
    analyzer = RealDataTermDecayAnalyzer()
    
    input_path = "data/interim/enhanced_modeling_results.json"
    output_path = "data/interim/real_term_decay_analysis_results.json"
    
    try:
        decay_results, summary = analyzer.run_real_analysis(input_path, output_path)
        
        logger.info("\n🎯 REAL DATA ANALYSIS SUMMARY:")
        overall = summary['overall_statistics']
        logger.info(f"    📊 Analyzed actual temporal TF-IDF patterns")
        logger.info(f"    🔍 Found {overall['total_decaying_terms']} terms with real decay patterns")
        logger.info(f"    📋 Applied exact steps.md rule to {overall['total_terms_analyzed']} terms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real data analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
