# Directory Processing Guide

## Overview

The L'Oréal Datathon 2025 pipeline now supports processing multiple comment and video files from a directory with the new `--data-dir` option. This feature automatically discovers and processes all parquet files in a directory, categorizing them as comment or video files based on naming patterns and schema detection.

## Usage

### Basic Directory Processing

To process all comment and video files in a directory:

```bash
python src/full_pipeline.py --data-dir data/processed/dataset
```

### Directory Processing with Options

```bash
# With custom output directory
python src/full_pipeline.py --data-dir data/processed/dataset --output-dir results/

# With disabled features
python src/full_pipeline.py --data-dir data/processed/dataset --disable-translation --disable-audio

# Using configuration file
python src/full_pipeline.py --data-dir data/processed/dataset --config pipeline_config.json
```

## File Discovery

The pipeline automatically discovers and categorizes parquet files using:

### 1. Naming Pattern Detection
- **Comment files**: Files containing `comment`, `comments`, `reply`, `replies` in filename
- **Video files**: Files containing `video`, `videos`, `content`, `post`, `posts` in filename

### 2. Schema Detection (Fallback)
- **Comment files**: Files with `textOriginal` or `parentCommentId` columns
- **Video files**: Files with `title`, `description`, `viewCount`, or `contentDuration` columns

## Supported File Structures

### Comment Files Expected Schema
```
textOriginal      (str)  - Main comment text
authorId          (str)  - Comment author ID
videoId           (str)  - Associated video ID  
likeCount         (int)  - Number of likes
parentCommentId   (str)  - Reply parent (null for top-level)
```

### Video Files Expected Schema
```
title             (str)  - Video title
description       (str)  - Video description
viewCount         (int)  - Number of views
likeCount         (int)  - Number of likes
commentCount      (int)  - Number of comments
contentDuration   (str)  - Video duration
channelId         (str)  - Channel identifier
topicCategories   (str)  - Content categories
```

## Examples

### Example Directory Structure
```
data/processed/dataset/
├── beauty_comments_batch1.parquet
├── beauty_comments_batch2.parquet
├── user_comments_set1.parquet
├── beauty_videos_batch1.parquet
├── content_videos_batch2.parquet
└── product_videos_set1.parquet
```

### Processing Results
The pipeline will automatically:
1. **Discover** all 6 parquet files
2. **Categorize** 3 as comment files, 3 as video files
3. **Process** each file with appropriate schema handling
4. **Track progress** across all files with comprehensive progress bars

### Command Examples

```bash
# Basic processing
python src/full_pipeline.py --data-dir data/processed/dataset

# Expected output:
# 📂 Found 6 files to process across 2 data types
# ✅ Discovered 3 comment files and 3 video files
# Processing all data files: 100%|████| 6/6 [00:05<00:00, 1.2file/s]
```

## Progress Tracking

The directory processing provides multi-level progress tracking:

### Main Progress Bar
- Overall file processing progress
- Files per second processing rate
- ETA for completion

### Per-File Progress 
- Individual file processing status
- File name and size information
- Success/failure indicators

### Data Processing Progress
- Chunk-level processing within each file
- Row counts and processing statistics
- Memory usage and optimization metrics

## Comparison: Directory vs Individual Files

| Feature | Directory Processing | Individual Files |
|---------|---------------------|------------------|
| **Command** | `--data-dir path/` | `--comments file1 --videos file2` |
| **File Discovery** | Automatic | Manual specification |
| **Scalability** | Handles multiple files | Limited to single files |
| **Progress Tracking** | Multi-file progress | Single-file progress |
| **Schema Detection** | Automatic categorization | Manual categorization |
| **Error Handling** | Continues on file errors | Stops on errors |

## Performance

### Processing Speed
- **Small files** (<1MB): ~10-20 files/second
- **Medium files** (1-100MB): ~2-5 files/second  
- **Large files** (>100MB): ~0.5-2 files/second

### Memory Management
- Chunked processing for large files (50K rows/chunk)
- Memory limit: 2GB per chunk
- Automatic garbage collection between files

### Optimization Tips
1. **Group similar files**: Keep comments and videos in same directory
2. **Use descriptive names**: Helps with automatic categorization
3. **Consistent schemas**: Ensure all comment files have same structure
4. **Monitor progress**: Use tqdm progress bars to track processing

## Error Handling

### File-Level Errors
- **Missing files**: Logged as warnings, processing continues
- **Corrupt files**: Skipped with error message, processing continues
- **Schema mismatches**: Automatic fallback to generic processing

### Directory-Level Errors
- **Missing directory**: Clear error message with suggestions
- **Empty directory**: Warning message, switches to sample data mode
- **Permission errors**: Error message with suggested fixes

### Recovery Options
```bash
# If processing fails, retry with simpler options
python src/full_pipeline.py --data-dir data/processed/dataset --disable-audio --disable-translation

# Check logs for specific file errors
tail -f pipeline_*.log
```

## Configuration

### Directory Processing Config
```json
{
  "data_sources": {
    "data_dir": "data/processed/dataset"
  },
  "processing": {
    "chunk_size": 50000,
    "max_memory_gb": 2,
    "parallel_processing": true
  },
  "discovery": {
    "include_subdirectories": true,
    "file_patterns": ["*.parquet"],
    "schema_detection": true
  }
}
```

### Usage with Config
```bash
python src/full_pipeline.py --config directory_config.json
```

This new directory processing capability makes it easy to handle large datasets with multiple files while maintaining the same comprehensive progress tracking and optimization features of the original pipeline.