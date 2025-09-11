#!/usr/bin/env python3
"""
Term-Level Decay Detection Analysis
Analyzes decay patterns for individual terms within each beauty category
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path

# No longer need the original DecayDetector import since we have our own implementation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TermDecayAnalyzer:
    """Analyzes decay patterns for individual terms within categories"""
    
    def __init__(self):
        self.results = {}
        
    def detect_term_decay(self, time_series_values, period_T=3):
        """
        Simplified decay detection for individual term time series.
        
        Args:
            time_series_values: List of TF-IDF values over time
            period_T: Period length for decay detection
            
        Returns:
            Dict with decay detection results
        """
        if len(time_series_values) < period_T + 1:
            return {
                'trend_state': 'Insufficient_Data',
                'decay_confidence': 0.0,
                'avg_growth_rate': 0.0,
                'avg_acceleration': 0.0
            }
        
        # Convert to numpy array for easier calculations
        values = np.array(time_series_values)
        
        # Calculate first derivative (growth rate)
        growth_rates = np.diff(values)
        
        # Calculate second derivative (acceleration)
        accelerations = np.diff(growth_rates)
        
        # Get the last period_T values for analysis
        recent_growth = growth_rates[-period_T:] if len(growth_rates) >= period_T else growth_rates
        recent_acceleration = accelerations[-period_T:] if len(accelerations) >= period_T else accelerations
        
        avg_growth_rate = np.mean(recent_growth) if len(recent_growth) > 0 else 0.0
        avg_acceleration = np.mean(recent_acceleration) if len(recent_acceleration) > 0 else 0.0
        
        # Apply decay detection rule: if growth_rate > 0 and acceleration < 0
        is_decaying = avg_growth_rate > 0 and avg_acceleration < 0
        
        # Calculate decay confidence based on how many recent periods show decay pattern
        decay_periods = 0
        for i in range(min(period_T, len(growth_rates)-1)):
            if i < len(growth_rates) and i < len(accelerations):
                growth_idx = -(i+1)
                accel_idx = -(i+1)
                if growth_rates[growth_idx] > 0 and accelerations[accel_idx] < 0:
                    decay_periods += 1
        
        decay_confidence = decay_periods / min(period_T, len(accelerations)) if len(accelerations) > 0 else 0.0
        
        # Determine trend state
        if is_decaying:
            trend_state = 'Decaying'
        elif avg_growth_rate > 0 and avg_acceleration > 0:
            trend_state = 'Accelerating'
        elif avg_growth_rate > 0 and abs(avg_acceleration) < 0.01:
            trend_state = 'Growing'
        elif abs(avg_growth_rate) < 0.01:
            trend_state = 'Stable'
        else:
            trend_state = 'Declining'
        
        return {
            'trend_state': trend_state,
            'decay_confidence': decay_confidence,
            'avg_growth_rate': float(avg_growth_rate),
            'avg_acceleration': float(avg_acceleration)
        }
        
    def load_enhanced_results(self, filepath):
        """Load the enhanced modeling results"""
        logger.info("📊 Loading enhanced modeling results...")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_term_timeseries(self, temporal_trends):
        """Extract time series data for each term across all categories"""
        logger.info("🔍 Extracting term time series data...")
        
        term_timeseries = defaultdict(lambda: defaultdict(list))
        
        for category, data in temporal_trends.items():
            trending_terms_by_window = data.get('trending_terms_by_window', {})
            
            # Sort time windows chronologically
            sorted_windows = sorted(trending_terms_by_window.keys())
            
            for window in sorted_windows:
                terms = trending_terms_by_window[window]
                
                # Create a dictionary of terms for this window
                window_terms = {}
                for term_data in terms:
                    if isinstance(term_data, dict):
                        term = term_data.get('term', '')
                        tfidf_score = term_data.get('tfidf_score', 0.0)
                        if term:
                            window_terms[term] = tfidf_score
                
                # For each term in this window, add the score
                # For missing terms, add 0.0
                all_terms = set()
                for window_data in trending_terms_by_window.values():
                    for term_data in window_data:
                        if isinstance(term_data, dict) and term_data.get('term'):
                            all_terms.add(term_data['term'])
                
                for term in all_terms:
                    score = window_terms.get(term, 0.0)
                    term_timeseries[category][term].append({
                        'window': window,
                        'tfidf_score': score
                    })
        
        return term_timeseries
    
    def analyze_term_decay(self, term_timeseries, min_appearances=5):
        """Analyze decay patterns for terms with sufficient data"""
        logger.info(f"🔬 Analyzing decay patterns for terms with ≥{min_appearances} appearances...")
        
        decay_results = {}
        
        for category, terms in term_timeseries.items():
            logger.info(f"\n📊 Processing {category} category...")
            category_results = {}
            
            decaying_count = 0
            total_analyzed = 0
            
            for term, timeseries in terms.items():
                # Filter terms with sufficient appearances
                non_zero_scores = [point['tfidf_score'] for point in timeseries if point['tfidf_score'] > 0]
                
                if len(non_zero_scores) >= min_appearances:
                    total_analyzed += 1
                    
                    # Extract TF-IDF scores as engagement metric
                    scores = [point['tfidf_score'] for point in timeseries]
                    
                    # Apply decay detection
                    try:
                        # Extract TF-IDF scores as engagement metric
                        decay_info = self.detect_term_decay(scores)
                        
                        category_results[term] = {
                            'trend_state': decay_info['trend_state'],
                            'decay_confidence': decay_info['decay_confidence'],
                            'avg_growth_rate': decay_info['avg_growth_rate'],
                            'avg_acceleration': decay_info['avg_acceleration'],
                            'appearances': len(non_zero_scores),
                            'total_windows': len(timeseries),
                            'max_tfidf': max(scores),
                            'min_tfidf': min(scores),
                            'latest_tfidf': scores[-1] if scores else 0.0
                        }
                        
                        if decay_info['trend_state'] == 'Decaying':
                            decaying_count += 1
                            logger.info(f"    🔍 {term}: DECAYING (confidence: {decay_info['decay_confidence']:.3f})")
                    
                    except Exception as e:
                        logger.warning(f"    ⚠️ Error analyzing {term}: {e}")
            
            decay_results[category] = category_results
            logger.info(f"    📈 Found {decaying_count} decaying terms out of {total_analyzed} analyzed")
        
        return decay_results
    
    def generate_summary_insights(self, decay_results):
        """Generate summary insights from term decay analysis"""
        logger.info("\n📋 Generating summary insights...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'decay_rule_applied': "if growth_rate > 0 and acceleration < 0 for period T: then trend_state = 'Decaying'",
            'categories_analyzed': len(decay_results),
            'category_insights': {}
        }
        
        for category, terms in decay_results.items():
            if not terms:
                continue
                
            trend_states = [term_data['trend_state'] for term_data in terms.values()]
            decaying_terms = [term for term, data in terms.items() if data['trend_state'] == 'Decaying']
            
            # Top decaying terms by confidence
            decaying_by_confidence = sorted(
                [(term, data) for term, data in terms.items() if data['trend_state'] == 'Decaying'],
                key=lambda x: x[1]['decay_confidence'],
                reverse=True
            )[:10]
            
            # Terms with highest peak TF-IDF that are now decaying
            high_impact_decaying = sorted(
                [(term, data) for term, data in terms.items() if data['trend_state'] == 'Decaying'],
                key=lambda x: x[1]['max_tfidf'],
                reverse=True
            )[:5]
            
            summary['category_insights'][category] = {
                'total_terms_analyzed': len(terms),
                'decaying_terms_count': len(decaying_terms),
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
                        'appearances': data['appearances']
                    }
                    for term, data in decaying_by_confidence
                ],
                'high_impact_decaying_terms': [
                    {
                        'term': term,
                        'max_tfidf': data['max_tfidf'],
                        'decay_confidence': data['decay_confidence'],
                        'latest_tfidf': data['latest_tfidf']
                    }
                    for term, data in high_impact_decaying
                ]
            }
        
        return summary
    
    def save_results(self, decay_results, summary, output_path):
        """Save the complete term decay analysis results"""
        logger.info(f"💾 Saving results to {output_path}...")
        
        complete_results = {
            'term_decay_analysis': decay_results,
            'summary_insights': summary,
            'methodology': {
                'decay_detection_rule': "if growth_rate > 0 and acceleration < 0 for period T: then trend_state = 'Decaying'",
                'period_T': 3,
                'minimum_appearances': 5,
                'engagement_metric': 'tfidf_score',
                'analysis_scope': 'individual_terms_within_categories'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    def print_insights(self, summary):
        """Print key insights to console"""
        logger.info("\n🔬 TERM-LEVEL DECAY DETECTION INSIGHTS")
        logger.info("=" * 60)
        
        for category, insights in summary['category_insights'].items():
            logger.info(f"\n📊 {category.upper()} CATEGORY:")
            logger.info(f"    📈 Total terms analyzed: {insights['total_terms_analyzed']}")
            logger.info(f"    🔍 Decaying terms: {insights['decaying_terms_count']} ({insights['decay_percentage']:.1f}%)")
            
            # Show trend state distribution
            dist = insights['trend_state_distribution']
            logger.info(f"    📊 Trend States: Decaying({dist['Decaying']}) | Accelerating({dist['Accelerating']}) | Growing({dist['Growing']}) | Stable({dist['Stable']}) | Declining({dist['Declining']})")
            
            # Show top decaying terms
            if insights['top_decaying_by_confidence']:
                logger.info(f"    🔍 Top decaying terms:")
                for i, term_info in enumerate(insights['top_decaying_by_confidence'][:3], 1):
                    logger.info(f"        {i}. '{term_info['term']}' (confidence: {term_info['decay_confidence']:.3f}, max TF-IDF: {term_info['max_tfidf']:.3f})")
    
    def run_analysis(self, input_path, output_path=None):
        """Run the complete term decay analysis"""
        if output_path is None:
            output_path = "data/interim/term_decay_analysis_results.json"
        
        logger.info("🔬 STARTING TERM-LEVEL DECAY DETECTION ANALYSIS")
        logger.info("=" * 60)
        
        # Load data
        enhanced_results = self.load_enhanced_results(input_path)
        temporal_trends = enhanced_results.get('temporal_trends', {})
        
        # Extract term time series
        term_timeseries = self.extract_term_timeseries(temporal_trends)
        
        # Analyze decay patterns
        decay_results = self.analyze_term_decay(term_timeseries)
        
        # Generate insights
        summary = self.generate_summary_insights(decay_results)
        
        # Save results
        self.save_results(decay_results, summary, output_path)
        
        # Print insights
        self.print_insights(summary)
        
        logger.info(f"\n✅ Analysis complete! Results saved to: {output_path}")
        
        return decay_results, summary

def main():
    """Main execution function"""
    analyzer = TermDecayAnalyzer()
    
    input_path = "data/interim/enhanced_modeling_results.json"
    output_path = "data/interim/term_decay_analysis_results.json"
    
    try:
        decay_results, summary = analyzer.run_analysis(input_path, output_path)
        
        logger.info("\n🎯 KEY FINDINGS:")
        total_categories = len(summary['category_insights'])
        total_decaying_terms = sum(
            insights['decaying_terms_count'] 
            for insights in summary['category_insights'].values()
        )
        
        logger.info(f"    📊 Analyzed {total_categories} beauty categories")
        logger.info(f"    🔍 Found {total_decaying_terms} terms showing decay patterns")
        logger.info(f"    📋 Applied exact steps.md rule: 'if growth_rate > 0 and acceleration < 0'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
