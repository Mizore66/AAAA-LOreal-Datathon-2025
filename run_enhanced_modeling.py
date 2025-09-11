#!/usr/bin/env python3
"""
🎭 L'Oréal Datathon 2025 - Enhanced Modeling Pipeline with Model Persistence and Term Trends
========================================================================================

Enhanced pipeline that saves trained models and tracks trending terms within categories.
"""

import sys
sys.path.append('src')

import pandas as pd
import json
import logging
import pickle
import joblib
from pathlib import Path
from modeling_optimized import (
    ModelingPipeline, 
    CategoryClassifier, 
    SentimentAnalyzer,
    SemanticValidator
)
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTrendAnalyzer:
    """Enhanced trend analyzer with term-level trend detection and model persistence."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pipeline = ModelingPipeline()
        self.category_classifier = CategoryClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.semantic_validator = SemanticValidator()
        
        # TF-IDF for term trend analysis
        self.tfidf_vectorizers = {}
        self.term_clusters = {}
        
        logger.info("✅ Enhanced trend analyzer initialized")
    
    def extract_trending_terms(self, texts, category, top_n=20):
        """Extract trending terms for a specific category using TF-IDF."""
        
        if not texts:
            return []
        
        # Clean and prepare texts
        cleaned_texts = []
        for text in texts:
            if isinstance(text, str) and len(text.strip()) > 10:
                # Remove URLs, mentions, hashtags for cleaner analysis
                clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text.lower())
                clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
                cleaned_texts.append(clean_text)
        
        if len(cleaned_texts) < 5:
            return []
        
        try:
            # Create TF-IDF vectorizer for this category
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),  # Include bigrams
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Get top terms
            top_indices = mean_scores.argsort()[-top_n:][::-1]
            trending_terms = [
                {
                    'term': feature_names[i],
                    'tfidf_score': float(mean_scores[i]),
                    'category': category
                }
                for i in top_indices if mean_scores[i] > 0
            ]
            
            # Save the vectorizer for this category
            self.tfidf_vectorizers[category] = vectorizer
            
            return trending_terms
            
        except Exception as e:
            logger.warning(f"Term extraction failed for {category}: {e}")
            return []
    
    def analyze_temporal_trends(self, df, category_col='category', text_col='cleaned_text', time_col='timestamp'):
        """Analyze how terms trend over time within categories."""
        
        temporal_trends = {}
        
        # Group by category
        for category in df[category_col].unique():
            category_data = df[df[category_col] == category].copy()
            
            if len(category_data) < 50:  # Minimum data for trend analysis
                continue
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(category_data[time_col]):
                category_data[time_col] = pd.to_datetime(category_data[time_col])
            
            # Sort by time
            category_data = category_data.sort_values(time_col)
            
            # Split into time windows (e.g., weekly)
            category_data['time_window'] = category_data[time_col].dt.to_period('W')
            
            # Analyze terms per time window
            window_trends = {}
            for window, window_data in category_data.groupby('time_window'):
                texts = window_data[text_col].dropna().tolist()
                window_terms = self.extract_trending_terms(texts, category, top_n=10)
                window_trends[str(window)] = window_terms
            
            temporal_trends[category] = {
                'total_posts': len(category_data),
                'time_windows': len(window_trends),
                'trending_terms_by_window': window_trends
            }
            
            logger.info(f"   📈 {category}: {len(window_trends)} time windows analyzed")
        
        return temporal_trends
    
    def save_models(self, suffix=""):
        """Save all trained models and components."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_prefix = f"loreal_datathon_{timestamp}{suffix}"
        
        saved_models = {}
        
        try:
            # Save TF-IDF vectorizers
            if self.tfidf_vectorizers:
                tfidf_path = self.models_dir / f"{model_prefix}_tfidf_vectorizers.pkl"
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizers, f)
                saved_models['tfidf_vectorizers'] = str(tfidf_path)
                logger.info(f"   💾 Saved TF-IDF vectorizers to {tfidf_path}")
            
            # Save category classifier (if it has a trained model)
            if hasattr(self.category_classifier, 'classifier') and self.category_classifier.classifier:
                try:
                    classifier_path = self.models_dir / f"{model_prefix}_category_classifier.pkl"
                    with open(classifier_path, 'wb') as f:
                        pickle.dump(self.category_classifier.classifier, f)
                    saved_models['category_classifier'] = str(classifier_path)
                    logger.info(f"   💾 Saved category classifier to {classifier_path}")
                except Exception as e:
                    logger.warning(f"Could not save category classifier: {e}")
            
            # Save sentiment analyzer model info
            if hasattr(self.sentiment_analyzer, 'sentiment_pipeline'):
                sentiment_info = {
                    'model_name': getattr(self.sentiment_analyzer, 'model_name', 'unknown'),
                    'pipeline_type': 'sentiment-analysis'
                }
                sentiment_path = self.models_dir / f"{model_prefix}_sentiment_info.json"
                with open(sentiment_path, 'w') as f:
                    json.dump(sentiment_info, f, indent=2)
                saved_models['sentiment_info'] = str(sentiment_path)
                logger.info(f"   💾 Saved sentiment model info to {sentiment_path}")
            
            # Save semantic validator model info
            if hasattr(self.semantic_validator, 'model'):
                semantic_info = {
                    'model_type': 'sentence-transformer',
                    'model_name': 'all-MiniLM-L6-v2'
                }
                semantic_path = self.models_dir / f"{model_prefix}_semantic_info.json"
                with open(semantic_path, 'w') as f:
                    json.dump(semantic_info, f, indent=2)
                saved_models['semantic_info'] = str(semantic_path)
                logger.info(f"   💾 Saved semantic model info to {semantic_path}")
            
            # Save model registry
            registry = {
                'created_at': datetime.now().isoformat(),
                'model_prefix': model_prefix,
                'saved_models': saved_models,
                'model_types': list(saved_models.keys())
            }
            
            registry_path = self.models_dir / f"{model_prefix}_model_registry.json"
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"✅ Model registry saved to {registry_path}")
            return saved_models, str(registry_path)
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return {}, None
    
    def load_models(self, model_registry_path):
        """Load previously saved models."""
        
        try:
            with open(model_registry_path, 'r') as f:
                registry = json.load(f)
            
            saved_models = registry['saved_models']
            
            # Load TF-IDF vectorizers
            if 'tfidf_vectorizers' in saved_models:
                with open(saved_models['tfidf_vectorizers'], 'rb') as f:
                    self.tfidf_vectorizers = pickle.load(f)
                logger.info("✅ Loaded TF-IDF vectorizers")
            
            # Load category classifier
            if 'category_classifier' in saved_models:
                with open(saved_models['category_classifier'], 'rb') as f:
                    classifier = pickle.load(f)
                    self.category_classifier.classifier = classifier
                logger.info("✅ Loaded category classifier")
            
            logger.info(f"✅ Successfully loaded models from {model_registry_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

def run_enhanced_modeling():
    """Run the enhanced modeling pipeline with term trends and model persistence."""
    
    logger.info("🎭 L'Oréal Datathon 2025 - Enhanced Modeling Pipeline")
    logger.info("=" * 70)
    
    # Load processed data files
    results_file = Path("data/interim/data_processing_results.json")
    if not results_file.exists():
        logger.error("❌ No processing results found. Run simple_pipeline.py first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    processed_files = results['processed_datasets']
    logger.info(f"📊 Found {len(processed_files)} processed datasets")
    
    # Initialize enhanced analyzer
    logger.info("🚀 Initializing Enhanced Modeling Pipeline...")
    analyzer = EnhancedTrendAnalyzer()
    
    # Collect all data for comprehensive analysis
    all_data = []
    dataset_summaries = {}
    
    for i, file_path in enumerate(processed_files, 1):
        if not Path(file_path).exists():
            logger.warning(f"⚠️ File not found: {file_path}")
            continue
        
        dataset_name = Path(file_path).stem.replace('_processed', '')
        logger.info(f"\n📈 Processing dataset {i}/{len(processed_files)}: {dataset_name}")
        
        # Load the dataset
        df = pd.read_parquet(file_path)
        logger.info(f"   • Loaded {len(df):,} rows")
        
        # Add dataset identifier
        df['dataset_source'] = dataset_name
        all_data.append(df)
        
        # Quick sentiment analysis sample
        sample_texts = df['cleaned_text'].dropna().sample(min(50, len(df))).tolist()
        sentiment_results = [analyzer.sentiment_analyzer.analyze_sentiment(text) for text in sample_texts]
        
        # Category-wise term extraction
        category_terms = {}
        for category in df['category'].unique():
            category_texts = df[df['category'] == category]['cleaned_text'].dropna().tolist()
            if len(category_texts) >= 10:
                trending_terms = analyzer.extract_trending_terms(category_texts, category, top_n=15)
                category_terms[category] = trending_terms
                logger.info(f"   📝 {category}: {len(trending_terms)} trending terms extracted")
        
        dataset_summaries[dataset_name] = {
            'total_rows': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'sentiment_sample': len(sentiment_results),
            'trending_terms_by_category': category_terms
        }
    
    # Combine all data for comprehensive analysis
    logger.info("\n🔬 COMPREHENSIVE CROSS-DATASET ANALYSIS")
    logger.info("=" * 70)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"📊 Combined dataset: {len(combined_df):,} total rows")
        
        # Temporal trend analysis
        logger.info("⏰ Analyzing temporal trends...")
        temporal_trends = analyzer.analyze_temporal_trends(combined_df)
        
        # Overall term trends by category
        logger.info("🏷️ Analyzing overall term trends by category...")
        overall_category_terms = {}
        for category in combined_df['category'].unique():
            category_texts = combined_df[combined_df['category'] == category]['cleaned_text'].dropna().tolist()
            if len(category_texts) >= 50:  # Higher threshold for overall analysis
                trending_terms = analyzer.extract_trending_terms(category_texts, category, top_n=25)
                overall_category_terms[category] = trending_terms
                logger.info(f"   📊 {category}: {len(trending_terms)} terms extracted from {len(category_texts)} posts")
        
        # Save models
        logger.info("\n💾 SAVING TRAINED MODELS")
        logger.info("=" * 70)
        saved_models, registry_path = analyzer.save_models("_enhanced")
        
        # Generate comprehensive results
        comprehensive_results = {
            "model_info": {
                "saved_models": saved_models,
                "model_registry": registry_path,
                "models_directory": str(analyzer.models_dir)
            },
            "dataset_summaries": dataset_summaries,
            "temporal_trends": temporal_trends,
            "overall_category_terms": overall_category_terms,
            "analysis_summary": {
                "total_datasets": len(processed_files),
                "total_rows_analyzed": len(combined_df),
                "categories_analyzed": len(overall_category_terms),
                "temporal_windows_created": sum(t.get('time_windows', 0) for t in temporal_trends.values()),
                "models_saved": len(saved_models)
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Save comprehensive results
        output_file = Path("data/interim/enhanced_modeling_results.json")
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Generate insights report
        logger.info("\n💡 KEY ENHANCED MODELING INSIGHTS")
        logger.info("=" * 70)
        
        for category, terms in overall_category_terms.items():
            logger.info(f"\n🔥 Top Trending Terms in {category.title()}:")
            for i, term_data in enumerate(terms[:10], 1):
                score = term_data['tfidf_score']
                logger.info(f"   {i:2d}. {term_data['term']} (score: {score:.3f})")
        
        # Temporal insights
        logger.info(f"\n⏰ Temporal Analysis Summary:")
        for category, trend_data in temporal_trends.items():
            windows = trend_data['time_windows']
            posts = trend_data['total_posts']
            logger.info(f"   • {category}: {posts:,} posts across {windows} time windows")
        
        logger.info(f"\n💾 Models Saved:")
        for model_type, path in saved_models.items():
            logger.info(f"   • {model_type}: {Path(path).name}")
        
        logger.info(f"\n📊 Enhanced results saved to: {output_file}")
        logger.info(f"📦 Model registry: {registry_path}")
        logger.info("\n✅ Enhanced modeling pipeline completed successfully!")
        
        return comprehensive_results
    
    else:
        logger.error("❌ No data could be loaded for analysis")
        return None

if __name__ == "__main__":
    try:
        results = run_enhanced_modeling()
        if results:
            print("\n🎉 Enhanced modeling completed successfully!")
            print("📊 Check enhanced_modeling_results.json for detailed term trends")
            print("💾 Check models/ directory for saved model files")
        else:
            print("❌ Enhanced modeling failed")
        
    except Exception as e:
        logger.error(f"❌ Error during enhanced modeling: {e}")
        raise
