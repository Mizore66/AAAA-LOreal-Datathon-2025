#!/usr/bin/env python3
"""
Demo script showcasing Phase 3 enhanced trend detection capabilities.
Demonstrates keyword-independent trend detection and statistical anomaly detection.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def demo_phase3_capabilities():
    """Demonstrate the enhanced trend detection capabilities."""
    
    print("=" * 60)
    print("L'ORÉAL DATATHON 2025: PHASE 3 TREND DETECTION DEMO")
    print("=" * 60)
    
    print("\n🚀 KEY INNOVATION: Keyword-Independent Trend Detection")
    print("Traditional approach: Limited to predefined keywords")
    print("Our approach: Discovers ANY emerging trends, even novel ones")
    
    print("\n📊 ENHANCED CAPABILITIES:")
    print("✅ Statistical Anomaly Detection - Finds unusual spikes in term frequency")
    print("✅ N-gram Analysis - Discovers emerging multi-word phrases")
    print("✅ Growth Rate Analysis - Identifies rapidly growing trends")
    print("✅ Trend Clustering - Groups related emerging concepts")
    print("✅ Velocity Tracking - Measures how quickly trends spread")
    print("✅ Caching System - Optimizes repeated analysis runs")
    
    # Load and display results
    data_dir = Path("data/processed")
    
    if (data_dir / "features_emerging_terms_6h.parquet").exists():
        print("\n📈 ANALYSIS RESULTS:")
        
        # Load emerging terms
        emerging_df = pd.read_parquet(data_dir / "features_emerging_terms_6h.parquet")
        emerging_only = emerging_df[emerging_df['is_emerging'] == True]
        
        print(f"💡 Total terms analyzed: {len(emerging_df['feature'].unique())}")
        print(f"🔥 Emerging trends detected: {len(emerging_only)}")
        print(f"📊 High-growth terms (>5x): {len(emerging_df[emerging_df['growth_rate'] > 5])}")
        
        # Show top emerging trends
        if not emerging_only.empty:
            print("\n🏆 TOP EMERGING TRENDS (Keyword-Independent):")
            latest_bin = emerging_only['bin'].max()
            top_trends = emerging_only[emerging_only['bin'] == latest_bin].sort_values('velocity', ascending=False).head(10)
            
            for i, (_, trend) in enumerate(top_trends.iterrows(), 1):
                growth_icon = "🚀" if trend['growth_rate'] > 3 else "📈"
                print(f"{i:2d}. {growth_icon} '{trend['feature']}' - Growth: {trend['growth_rate']:.1f}x, Velocity: {trend['velocity']:.2f}")
        
        # Load clusters
        if (data_dir / "trend_clusters.parquet").exists():
            clusters_df = pd.read_parquet(data_dir / "trend_clusters.parquet")
            print(f"\n🔗 Trend clusters identified: {len(clusters_df)}")
            
            # Show top cluster
            if not clusters_df.empty:
                top_cluster = clusters_df.sort_values('avg_velocity', ascending=False).iloc[0]
                print(f"🏅 Top cluster: {len(top_cluster['terms'])} related terms with velocity {top_cluster['avg_velocity']:.2f}")
                print(f"   Example terms: {', '.join(top_cluster['terms'][:5])}")
    
    else:
        print("\n⚠️  No analysis results found. Run 'python src/data_processing.py' first.")
    
    print("\n🎯 BUSINESS VALUE:")
    print("• Detect truly emerging trends before competitors")
    print("• Identify novel beauty concepts and terminology")
    print("• Track trend velocity and growth patterns")
    print("• Cluster related trending concepts")
    print("• Reduce dependency on predefined keyword lists")
    
    print("\n📋 NEXT STEPS:")
    print("1. Run with real social media datasets")
    print("2. Integrate with real-time data feeds")
    print("3. Add semantic similarity analysis")
    print("4. Build Streamlit dashboard for visualization")
    
    print("\n" + "=" * 60)
    print("Demo completed! Check data/interim/ for detailed reports.")
    print("=" * 60)

if __name__ == '__main__':
    demo_phase3_capabilities()