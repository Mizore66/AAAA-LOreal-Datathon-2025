#!/usr/bin/env python3
"""
🔬 L'Oréal Datathon 2025 - Decay Detection Pipeline
===================================================

Pipeline that runs enhanced modeling and then applies decay detection at both
category and term levels using the results.
"""

import sys
sys.path.append('src')

import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import subprocess
from modeling_optimized import DecayDetector
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecayDetectionPipeline:
    """Pipeline that runs enhanced modeling then applies decay detection."""
    
    def __init__(self):
        self.decay_detector = DecayDetector()
        self.enhanced_results = None
        self.decay_results = {}
        
        logger.info("✅ Decay detection pipeline initialized")
    
    def run_enhanced_modeling(self):
        """Run the enhanced modeling pipeline first."""
        
        logger.info("🚀 Step 1: Running Enhanced Modeling Pipeline...")
        logger.info("=" * 60)
        
        try:
            # Run the enhanced modeling script
            result = subprocess.run([
                sys.executable, 'run_enhanced_modeling.py'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("✅ Enhanced modeling completed successfully")
                
                # Load the results
                results_file = Path("data/interim/enhanced_modeling_results.json")
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.enhanced_results = json.load(f)
                    logger.info(f"📊 Loaded enhanced modeling results")
                    return True
                else:
                    logger.error("❌ Enhanced modeling results file not found")
                    return False
            else:
                logger.error(f"❌ Enhanced modeling failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running enhanced modeling: {e}")
            return False
    
    def analyze_category_level_decay(self):
        """Analyze decay at the category level using temporal trends."""
        
        logger.info("\n🔍 Step 2: Category-Level Decay Detection...")
        logger.info("=" * 60)
        
        if not self.enhanced_results:
            logger.error("❌ No enhanced results available")
            return {}
        
        temporal_trends = self.enhanced_results.get('temporal_trends', {})
        category_decay_results = {}
        
        for category, trend_data in temporal_trends.items():
            logger.info(f"   🏷️ Analyzing {category}...")
            
            # Extract temporal data from trending terms by window
            window_trends = trend_data.get('trending_terms_by_window', {})
            
            if len(window_trends) < 5:
                logger.warning(f"      ⚠️ Insufficient data for {category} ({len(window_trends)} windows)")
                continue
            
            # Create time series data for decay detection
            decay_analysis_data = []
            
            for window_str, terms in window_trends.items():
                try:
                    # Parse window string to get timestamp
                    window_timestamp = pd.to_datetime(window_str)
                    
                    # Calculate aggregate metrics for this window
                    total_tfidf = sum(term.get('tfidf_score', 0) for term in terms)
                    term_count = len(terms)
                    avg_tfidf = total_tfidf / max(term_count, 1)
                    
                    decay_analysis_data.append({
                        'feature': category,
                        'timestamp': window_timestamp,
                        'time_bin': window_timestamp,
                        'count': term_count,
                        'total_tfidf': total_tfidf,
                        'avg_tfidf': avg_tfidf
                    })
                    
                except Exception as e:
                    logger.warning(f"      ⚠️ Could not parse window {window_str}: {e}")
                    continue
            
            if len(decay_analysis_data) >= 5:
                # Run decay detection on category-level data
                decay_df = pd.DataFrame(decay_analysis_data)
                decay_df = decay_df.sort_values('timestamp')
                
                # Use avg_tfidf as the main metric for decay detection
                decay_df['count'] = decay_df['avg_tfidf']  # Use TF-IDF as the metric
                
                decay_results = self.decay_detector.detect_decay(decay_df, period_T=3)
                
                if not decay_results.empty:
                    decay_info = decay_results.iloc[0]
                    category_decay_results[category] = {
                        'trend_state': decay_info.get('trend_state', 'Unknown'),
                        'decay_confidence': float(decay_info.get('decay_confidence', 0)),
                        'avg_growth_rate': float(decay_info.get('avg_growth_rate', 0)),
                        'avg_acceleration': float(decay_info.get('avg_acceleration', 0)),
                        'periods_analyzed': int(decay_info.get('periods_analyzed', 0)),
                        'data_points': len(decay_analysis_data),
                        'analysis_type': 'category_level_tfidf'
                    }
                    
                    state = category_decay_results[category]['trend_state']
                    confidence = category_decay_results[category]['decay_confidence']
                    logger.info(f"      📊 Result: {state} (confidence: {confidence:.3f})")
                else:
                    logger.warning(f"      ⚠️ No decay results for {category}")
            else:
                logger.warning(f"      ⚠️ Insufficient valid data points for {category}")
        
        return category_decay_results
    
    def analyze_term_level_decay(self):
        """Analyze decay at the term level using real data."""
        
        logger.info("\n🔬 Step 3: Term-Level Decay Detection...")
        logger.info("=" * 60)
        
        try:
            # Run the real term decay analysis
            result = subprocess.run([
                sys.executable, 'real_term_decay_analysis.py'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("✅ Term-level decay analysis completed")
                
                # Load the term-level results
                term_results_file = Path("data/interim/real_term_decay_analysis_results.json")
                if term_results_file.exists():
                    with open(term_results_file, 'r') as f:
                        term_results = json.load(f)
                    
                    # Extract summary statistics
                    term_decay_summary = {}
                    categories_analysis = term_results.get('detailed_analysis', {}).get('categories', {})
                    
                    for category, category_data in categories_analysis.items():
                        term_decay_summary[category] = {
                            'total_terms_analyzed': category_data.get('total_terms_analyzed', 0),
                            'decaying_terms_count': category_data.get('decaying_terms_count', 0),
                            'accelerating_terms_count': category_data.get('accelerating_terms_count', 0),
                            'decay_percentage': category_data.get('decay_percentage', 0.0),
                            'trend_distribution': category_data.get('trend_state_distribution', {}),
                            'top_decaying_terms': category_data.get('top_decaying_by_confidence', []),
                            'top_accelerating_terms': category_data.get('top_accelerating_terms', []),
                            'analysis_type': 'term_level_tfidf'
                        }
                        
                        terms_count = term_decay_summary[category]['total_terms_analyzed']
                        decaying_count = term_decay_summary[category]['decaying_terms_count']
                        accelerating_count = term_decay_summary[category]['accelerating_terms_count']
                        
                        logger.info(f"   📝 {category}: {terms_count} terms analyzed")
                        logger.info(f"      🔍 Decaying: {decaying_count}, Accelerating: {accelerating_count}")
                    
                    return term_decay_summary
                else:
                    logger.error("❌ Term-level results file not found")
                    return {}
            else:
                logger.error(f"❌ Term-level decay analysis failed: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error running term-level analysis: {e}")
            return {}
    
    def generate_comprehensive_report(self, category_results, term_results):
        """Generate a comprehensive decay detection report."""
        
        logger.info("\n📋 Step 4: Generating Comprehensive Report...")
        logger.info("=" * 60)
        
        # Combine results
        comprehensive_results = {
            "pipeline_info": {
                "pipeline_name": "L'Oréal Datathon 2025 - Decay Detection Pipeline",
                "generated_at": datetime.now().isoformat(),
                "analysis_levels": ["category_level", "term_level"]
            },
            "category_level_decay": category_results,
            "term_level_decay": term_results,
            "cross_analysis": self.compare_category_vs_term_results(category_results, term_results),
            "overall_summary": self.generate_overall_summary(category_results, term_results)
        }
        
        # Save comprehensive results
        output_file = Path("data/interim/comprehensive_decay_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive report saved to: {output_file}")
        return comprehensive_results, str(output_file)
    
    def compare_category_vs_term_results(self, category_results, term_results):
        """Compare category-level vs term-level decay detection results."""
        
        comparison = {}
        
        # Find categories analyzed in both levels
        common_categories = set(category_results.keys()) & set(term_results.keys())
        
        for category in common_categories:
            cat_state = category_results[category].get('trend_state', 'Unknown')
            term_decaying = term_results[category].get('decaying_terms_count', 0)
            term_accelerating = term_results[category].get('accelerating_terms_count', 0)
            term_total = term_results[category].get('total_terms_analyzed', 0)
            
            comparison[category] = {
                'category_trend_state': cat_state,
                'category_confidence': category_results[category].get('decay_confidence', 0),
                'terms_decaying_count': term_decaying,
                'terms_accelerating_count': term_accelerating,
                'terms_total': term_total,
                'terms_decay_percentage': (term_decaying / max(term_total, 1)) * 100,
                'agreement': self.assess_agreement(cat_state, term_decaying, term_accelerating, term_total)
            }
        
        return comparison
    
    def assess_agreement(self, category_state, term_decaying, term_accelerating, term_total):
        """Assess agreement between category and term level analysis."""
        
        if term_total == 0:
            return "insufficient_data"
        
        term_decay_rate = term_decaying / term_total
        term_accel_rate = term_accelerating / term_total
        
        if category_state == 'Decaying' and term_decay_rate > 0.1:
            return "strong_agreement"
        elif category_state == 'Accelerating' and term_accel_rate > 0.1:
            return "strong_agreement"
        elif category_state in ['Stable', 'Growing'] and term_decay_rate < 0.05 and term_accel_rate < 0.05:
            return "agreement"
        elif abs(term_decay_rate - term_accel_rate) < 0.02:
            return "neutral_agreement"
        else:
            return "disagreement"
    
    def generate_overall_summary(self, category_results, term_results):
        """Generate overall summary statistics."""
        
        # Category-level summary
        category_states = {}
        for category, data in category_results.items():
            state = data.get('trend_state', 'Unknown')
            if state not in category_states:
                category_states[state] = 0
            category_states[state] += 1
        
        # Term-level summary
        total_terms = sum(data.get('total_terms_analyzed', 0) for data in term_results.values())
        total_decaying_terms = sum(data.get('decaying_terms_count', 0) for data in term_results.values())
        total_accelerating_terms = sum(data.get('accelerating_terms_count', 0) for data in term_results.values())
        
        return {
            "categories_analyzed": len(category_results),
            "category_trend_distribution": category_states,
            "total_terms_analyzed": total_terms,
            "total_decaying_terms": total_decaying_terms,
            "total_accelerating_terms": total_accelerating_terms,
            "overall_term_decay_rate": (total_decaying_terms / max(total_terms, 1)) * 100,
            "overall_term_acceleration_rate": (total_accelerating_terms / max(total_terms, 1)) * 100
        }
    
    def run_complete_pipeline(self):
        """Run the complete decay detection pipeline."""
        
        logger.info("🎭 L'Oréal Datathon 2025 - Decay Detection Pipeline")
        logger.info("=" * 70)
        
        # Step 1: Run enhanced modeling
        if not self.run_enhanced_modeling():
            logger.error("❌ Pipeline failed at enhanced modeling stage")
            return None
        
        # Step 2: Category-level decay detection
        category_results = self.analyze_category_level_decay()
        
        # Step 3: Term-level decay detection
        term_results = self.analyze_term_level_decay()
        
        # Step 4: Generate comprehensive report
        comprehensive_results, report_path = self.generate_comprehensive_report(
            category_results, term_results
        )
        
        # Step 5: Summary insights
        self.print_summary_insights(comprehensive_results)
        
        logger.info(f"\n✅ Complete decay detection pipeline finished!")
        logger.info(f"📊 Comprehensive report: {report_path}")
        
        return comprehensive_results
    
    def print_summary_insights(self, results):
        """Print key insights from the analysis."""
        
        logger.info("\n💡 KEY DECAY DETECTION INSIGHTS")
        logger.info("=" * 70)
        
        overall = results.get('overall_summary', {})
        cross_analysis = results.get('cross_analysis', {})
        
        # Overall statistics
        logger.info(f"📊 Overall Statistics:")
        logger.info(f"   • Categories analyzed: {overall.get('categories_analyzed', 0)}")
        logger.info(f"   • Total terms analyzed: {overall.get('total_terms_analyzed', 0):,}")
        logger.info(f"   • Terms showing decay: {overall.get('total_decaying_terms', 0)} ({overall.get('overall_term_decay_rate', 0):.1f}%)")
        logger.info(f"   • Terms showing acceleration: {overall.get('total_accelerating_terms', 0)} ({overall.get('overall_term_acceleration_rate', 0):.1f}%)")
        
        # Category trend distribution
        category_dist = overall.get('category_trend_distribution', {})
        if category_dist:
            logger.info(f"\n🏷️ Category-Level Trends:")
            for state, count in category_dist.items():
                logger.info(f"   • {state}: {count} categories")
        
        # Agreement analysis
        agreements = {}
        for category, comp in cross_analysis.items():
            agreement = comp.get('agreement', 'unknown')
            if agreement not in agreements:
                agreements[agreement] = []
            agreements[agreement].append(category)
        
        if agreements:
            logger.info(f"\n🔍 Category vs Term Analysis Agreement:")
            for agreement_type, categories in agreements.items():
                logger.info(f"   • {agreement_type}: {', '.join(categories)}")

def main():
    """Main execution function."""
    
    try:
        pipeline = DecayDetectionPipeline()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print("\n🎉 Decay detection pipeline completed successfully!")
            print("📊 Check comprehensive_decay_analysis.json for detailed results")
        else:
            print("❌ Decay detection pipeline failed")
            
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise

if __name__ == "__main__":
    main()
