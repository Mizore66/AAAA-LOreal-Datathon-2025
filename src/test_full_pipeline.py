#!/usr/bin/env python3
"""
Test script for the Full Pipeline with progress bars
Demonstrates all pipeline components working together with comprehensive progress tracking.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the full pipeline
from full_pipeline import FullPipeline

def create_sample_data():
    """Create sample data for testing the pipeline"""
    
    # Create sample directory structure
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Sample comments data (simulating YouTube comments structure)
    comments_data = pd.DataFrame({
        'textOriginal': [
            'This mascara is amazing! #beauty #makeup',
            'Best foundation ever, covers everything',
            'Love this skincare routine! My skin is glowing ✨',
            'This lipstick shade is perfect for summer',
            'skincare routine that changed my life',
            'foundation matching tutorial',
            'This new perfume smells incredible',
            'Hair care tips for damaged hair',
            'Fashion week looks inspiration',
            'Beauty haul from Sephora'
        ],
        'likeCount': [100, 50, 75, 30, 200, 80, 45, 90, 120, 60],
        'authorId': [f'user_{i}' for i in range(10)],
        'videoId': [f'video_{i%3}' for i in range(10)],
        'parentCommentId': [None] * 8 + ['parent_1', 'parent_2'],  # 2 replies
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1D')
    })
    
    # Sample videos data (simulating YouTube videos structure)
    videos_data = pd.DataFrame({
        'title': [
            'Best Makeup Tutorial 2024',
            'Skincare Routine for Glowing Skin',
            'Fashion Haul Spring Collection'
        ],
        'description': [
            'Complete makeup tutorial featuring latest beauty trends and products',
            'My daily skincare routine that transformed my skin in 30 days',
            'Spring fashion haul with latest trends and styling tips'
        ],
        'tags': [
            'makeup,beauty,tutorial,cosmetics',
            'skincare,beauty,routine,glowing',
            'fashion,style,haul,spring'
        ],
        'viewCount': [50000, 75000, 30000],
        'likeCount': [2500, 3000, 1200],
        'commentCount': [150, 200, 80],
        'favouriteCount': [500, 800, 300],
        'contentDuration': ['PT10M30S', 'PT15M45S', 'PT8M20S'],
        'channelId': ['channel_1', 'channel_2', 'channel_3'],
        'topicCategories': [
            'Beauty,Fashion',
            'Beauty,Skincare',
            'Fashion,Lifestyle'
        ],
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='7D')
    })
    
    # Save sample data
    comments_path = data_dir / 'sample_comments.parquet'
    videos_path = data_dir / 'sample_videos.parquet'
    
    comments_data.to_parquet(comments_path, index=False)
    videos_data.to_parquet(videos_path, index=False)
    
    print(f"✅ Created sample data:")
    print(f"   📂 Comments: {comments_path} ({len(comments_data)} rows)")
    print(f"   📂 Videos: {videos_path} ({len(videos_data)} rows)")
    
    return {
        'comments': str(comments_path),
        'videos': str(videos_path)
    }

def main():
    """Test the full pipeline with sample data"""
    print("🧪 Testing Full Pipeline with Progress Bars")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample data...")
    data_sources = create_sample_data()
    
    # Initialize pipeline
    print("\n🚀 Initializing Full Pipeline...")
    config = {
        'data_sources': data_sources,
        'processing': {
            'chunk_size': 1000,  # Small chunks for demo
            'enable_audio': False,  # Skip audio for demo
        },
        'feature_processing': {
            'enable_spell_check': True,
            'enable_translation': True,
        },
        'modeling': {
            'enable_semantic_validation': True,
            'enable_sentiment_analysis': True,
            'enable_decay_detection': True,
        }
    }
    
    pipeline = FullPipeline(config=config)
    
    # Run the full pipeline
    print("\n🎯 Running Full Pipeline...")
    try:
        results = pipeline.run_full_pipeline(
            data_sources=data_sources,
            save_intermediate=True
        )
        
        print("\n🎉 Pipeline completed successfully!")
        print("\n📊 Results Summary:")
        
        # Data processing results
        if 'data_processing' in results:
            data_stats = results['data_processing'].get('statistics', {})
            print(f"   📊 Data Processing:")
            print(f"      - Datasets processed: {data_stats.get('total_datasets', 0)}")
            print(f"      - Total rows: {data_stats.get('total_rows', 0)}")
        
        # Feature processing results
        if 'feature_processing' in results:
            feature_meta = results['feature_processing'].get('metadata', {})
            print(f"   🔤 Feature Processing:")
            print(f"      - Files processed: {feature_meta.get('total_files_processed', 0)}")
            print(f"      - Corrections made: {feature_meta.get('total_corrections', 0)}")
            print(f"      - Translations: {feature_meta.get('total_translations', 0)}")
        
        # Modeling results
        if 'modeling' in results:
            model_metrics = results['modeling'].get('performance_metrics', {})
            print(f"   🤖 Modeling:")
            print(f"      - Models trained: {model_metrics.get('models_trained', 0)}")
            print(f"      - Analyses completed: {model_metrics.get('analysis_steps_completed', 0)}")
        
        print(f"\n📁 Results saved in: {pipeline.dirs['reports']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)