#!/usr/bin/env python3
"""
Demo script showing how to use the new --data-dir functionality
to process multiple comment and video files from a directory
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path

def create_sample_dataset_directory():
    """Create a sample dataset directory with multiple comment and video files"""
    
    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp()) / 'demo_dataset'
    temp_dir.mkdir(parents=True)
    
    print(f"📁 Creating sample dataset directory: {temp_dir}")
    
    # Create sample comment files
    comment_data_set1 = pd.DataFrame({
        'textOriginal': [
            'Great mascara! Love the volume',
            'This foundation is perfect for my skin tone',
            'Amazing lipstick color, lasts all day'
        ],
        'authorId': ['user1', 'user2', 'user3'],
        'videoId': ['video1', 'video1', 'video2'],
        'likeCount': [15, 8, 12],
        'parentCommentId': [None, None, None]
    })
    
    comment_data_set2 = pd.DataFrame({
        'textOriginal': [
            'Perfect for sensitive skin',
            'Love this new perfume collection',
            'Best skincare routine ever!'
        ],
        'authorId': ['user4', 'user5', 'user6'],
        'videoId': ['video2', 'video3', 'video3'],
        'likeCount': [22, 5, 18],
        'parentCommentId': [None, None, None]
    })
    
    # Create sample video files
    video_data_set1 = pd.DataFrame({
        'title': ['Ultimate Mascara Review 2025', 'Perfect Foundation Match Guide'],
        'description': [
            'Testing the latest mascara releases for volume and length',
            'How to find your perfect foundation shade and undertone'
        ],
        'viewCount': [15000, 25000],
        'likeCount': [350, 480],
        'commentCount': [45, 67],
        'contentDuration': ['00:08:30', '00:12:45'],
        'channelId': ['beauty_guru_1', 'makeup_expert'],
        'topicCategories': ['Beauty|Makeup', 'Beauty|Foundation']
    })
    
    video_data_set2 = pd.DataFrame({
        'title': ['Skincare Routine for Beginners', 'Perfume Collection Tour'],
        'description': [
            'Simple 5-step skincare routine for healthy glowing skin',
            'My favorite perfumes for different occasions and seasons'
        ],
        'viewCount': [18000, 12000],
        'likeCount': [420, 290],
        'commentCount': [38, 52],
        'contentDuration': ['00:10:15', '00:15:20'],
        'channelId': ['skincare_pro', 'fragrance_lover'],
        'topicCategories': ['Beauty|Skincare', 'Beauty|Fragrance']
    })
    
    # Save files with different naming patterns to test discovery
    comment_data_set1.to_parquet(temp_dir / 'beauty_comments_batch1.parquet')
    comment_data_set2.to_parquet(temp_dir / 'user_comments_batch2.parquet')
    video_data_set1.to_parquet(temp_dir / 'beauty_videos_batch1.parquet')
    video_data_set2.to_parquet(temp_dir / 'content_videos_batch2.parquet')
    
    print(f"✅ Created sample dataset with 4 files:")
    print(f"  📝 Comment files: beauty_comments_batch1.parquet, user_comments_batch2.parquet")
    print(f"  🎥 Video files: beauty_videos_batch1.parquet, content_videos_batch2.parquet")
    
    return str(temp_dir)

def show_usage_examples(dataset_dir):
    """Show different ways to use the pipeline with the directory"""
    
    print("\n" + "="*80)
    print("🚀 PIPELINE USAGE EXAMPLES")
    print("="*80)
    
    print(f"\n1️⃣  Process all files in directory:")
    print(f"   python src/full_pipeline.py --data-dir {dataset_dir}")
    
    print(f"\n2️⃣  Process directory with custom output:")
    print(f"   python src/full_pipeline.py --data-dir {dataset_dir} --output-dir results/")
    
    print(f"\n3️⃣  Process directory with disabled features:")
    print(f"   python src/full_pipeline.py --data-dir {dataset_dir} --disable-translation")
    
    print(f"\n4️⃣  Process specific files (alternative):")
    files = list(Path(dataset_dir).glob('*.parquet'))
    comment_files = [f for f in files if 'comment' in f.name.lower()]
    video_files = [f for f in files if 'video' in f.name.lower()]
    
    if comment_files and video_files:
        print(f"   python src/full_pipeline.py --comments {comment_files[0]} --videos {video_files[0]}")

def main():
    """Main demo function"""
    print("🎭 L'Oréal Datathon 2025 - Directory Processing Demo")
    print("="*60)
    
    # Create sample dataset
    dataset_dir = create_sample_dataset_directory()
    
    # Show file discovery
    print(f"\n📂 Dataset directory contents:")
    for file in Path(dataset_dir).glob('*.parquet'):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")
    
    # Show usage examples
    show_usage_examples(dataset_dir)
    
    print(f"\n🔍 To test the pipeline with this sample data, run:")
    print(f"   python src/full_pipeline.py --data-dir {dataset_dir}")
    
    print(f"\n📁 Sample dataset location: {dataset_dir}")
    print(f"   (This temporary directory will be cleaned up when you restart your session)")
    
    return dataset_dir

if __name__ == '__main__':
    demo_dir = main()