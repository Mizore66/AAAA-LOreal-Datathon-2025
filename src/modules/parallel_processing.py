#!/usr/bin/env python3
"""
Parallel processing utilities for L'Oréal Datathon 2025
Handles concurrent processing of data sources and chunked operations.
"""

import pandas as pd
import logging
from typing import List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from .config import CONFIG, MAX_WORKERS

logger = logging.getLogger(__name__)

def process_source_chunk(args: Tuple) -> Optional[pd.DataFrame]:
    """Process a single data source chunk for parallel processing."""
    name, df, processing_func = args
    try:
        return processing_func(name, df)
    except Exception as e:
        logger.error(f"Error processing source {name}: {e}")
        return None

def parallel_process_sources(sources: List[Tuple[str, pd.DataFrame]], 
                           processing_func: Callable,
                           max_workers: Optional[int] = None) -> List[pd.DataFrame]:
    """Process sources in parallel when beneficial."""
    if not CONFIG["enable_parallel"] or len(sources) < 2:
        # Sequential processing
        results = []
        for name, df in sources:
            result = processing_func(name, df)
            if result is not None:
                results.append(result)
        return results
    
    max_workers = max_workers or min(MAX_WORKERS, len(sources))
    results = []
    
    logger.info(f"Processing {len(sources)} sources with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for parallel processing
        args_list = [(name, df, processing_func) for name, df in sources]
        
        # Execute in parallel
        future_results = executor.map(process_source_chunk, args_list)
        
        # Collect results
        for result in future_results:
            if result is not None:
                results.append(result)
    
    return results

class ChunkedDataProcessor:
    """
    Advanced chunked processing system for handling large datasets efficiently.
    Splits datasets into manageable chunks and processes them independently.
    """
    
    def __init__(self, chunk_size: int = 50_000, memory_limit_mb: int = 500):
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        
    def calculate_optimal_chunk_size(self, df: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on dataset characteristics."""
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if memory_mb > self.memory_limit_mb:
            # Reduce chunk size for large datasets
            factor = memory_mb / self.memory_limit_mb
            optimal_size = max(1000, int(self.chunk_size / factor))
            logger.info(f"Adjusting chunk size from {self.chunk_size} to {optimal_size} due to memory constraints")
            return optimal_size
        
        return self.chunk_size
    
    def create_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into manageable chunks."""
        if df.empty:
            return []
        
        chunk_size = self.calculate_optimal_chunk_size(df)
        chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks of ~{chunk_size:,} rows each")
        return chunks
    
    def process_chunk_hashtags(self, chunk: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
        """Process a single chunk for hashtag aggregation."""
        from .aggregation import aggregate_hashtags_optimized
        return aggregate_hashtags_optimized("chunk", chunk, label)
    
    def process_chunk_keywords(self, chunk: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
        """Process a single chunk for keyword aggregation."""
        from .aggregation import aggregate_keywords_optimized
        return aggregate_keywords_optimized("chunk", chunk, label)
    
    def process_chunk_emerging_terms(self, chunk: pd.DataFrame, label: str = '6h') -> Optional[pd.DataFrame]:
        """Process a single chunk for emerging terms."""
        from .aggregation import process_emerging_source
        return process_emerging_source("chunk", chunk, label)
    
    def process_dataset_chunked(self, df: pd.DataFrame, processing_func: Callable, 
                              label: str = '6h', feature_type: str = 'hashtags') -> Optional[pd.DataFrame]:
        """
        Process a large dataset using chunked approach.
        
        Args:
            df: DataFrame to process
            processing_func: Function to apply to each chunk
            label: Time label for processing
            feature_type: Type of features being processed
        
        Returns:
            Combined results from all chunks
        """
        if df.empty:
            return None
        
        # Create chunks
        chunks = self.create_chunks(df)
        
        if len(chunks) == 1:
            # No need for chunking
            return processing_func(chunks[0], label)
        
        # Process chunks
        logger.info(f"Processing {len(chunks)} chunks for {feature_type}")
        chunk_results = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            result = processing_func(chunk, label)
            if result is not None:
                chunk_results.append(result)
        
        if not chunk_results:
            logger.warning(f"No results from chunked processing of {feature_type}")
            return None
        
        # Combine results
        combined = pd.concat(chunk_results, ignore_index=True)
        
        # Re-aggregate if necessary (for overlapping features across chunks)
        if 'bin' in combined.columns and 'feature' in combined.columns and 'count' in combined.columns:
            if 'source_type' in combined.columns:
                combined = combined.groupby(['bin', 'feature', 'source_type'], as_index=False)['count'].sum()
            else:
                combined = combined.groupby(['bin', 'feature'], as_index=False)['count'].sum()
            
            # Re-add category if missing
            if 'category' not in combined.columns:
                from .text_processing import categorize_feature
                combined['category'] = combined['feature'].apply(categorize_feature)
        
        logger.info(f"Chunked processing complete: {len(combined)} final records")
        return combined

# Global chunked processor instance
chunked_processor = ChunkedDataProcessor(
    chunk_size=CONFIG.get("chunk_size", 50_000),
    memory_limit_mb=CONFIG.get("chunk_memory_limit_mb", 300)
)

def process_dataset_with_chunking(name: str, df: pd.DataFrame, 
                                processing_type: str = 'hashtags', 
                                label: str = '6h') -> Optional[pd.DataFrame]:
    """
    Process a dataset using chunked approach if it's large enough.
    
    Args:
        name: Dataset name for logging
        df: DataFrame to process
        processing_type: Type of processing ('hashtags', 'keywords', 'emerging_terms')
        label: Time label for aggregation
    
    Returns:
        Processed results
    """
    if df.empty:
        return None
    
    # Determine if chunking is beneficial
    from .data_loading import get_dataset_size_mb
    dataset_size_mb = get_dataset_size_mb(df)
    threshold_rows = CONFIG.get("chunking_threshold_rows", 100_000)
    should_chunk = (len(df) > threshold_rows) or (dataset_size_mb > 200)
    
    if not should_chunk:
        # Use regular processing
        if processing_type == 'hashtags':
            from .aggregation import aggregate_hashtags_optimized
            return aggregate_hashtags_optimized(name, df, label)
        elif processing_type == 'keywords':
            from .aggregation import aggregate_keywords_optimized
            return aggregate_keywords_optimized(name, df, label)
        elif processing_type == 'emerging_terms':
            from .aggregation import process_emerging_source
            return process_emerging_source(name, df, label)
    
    # Use chunked processing for large datasets
    logger.info(f"Using chunked processing for {name}: {len(df):,} rows, {dataset_size_mb:.1f}MB")
    
    if processing_type == 'hashtags':
        return chunked_processor.process_dataset_chunked(
            df, chunked_processor.process_chunk_hashtags, label, 'hashtags'
        )
    elif processing_type == 'keywords':
        return chunked_processor.process_dataset_chunked(
            df, chunked_processor.process_chunk_keywords, label, 'keywords'
        )
    elif processing_type == 'emerging_terms':
        return chunked_processor.process_dataset_chunked(
            df, chunked_processor.process_chunk_emerging_terms, label, 'emerging_terms'
        )
    else:
        logger.error(f"Unknown processing type: {processing_type}")
        return None
