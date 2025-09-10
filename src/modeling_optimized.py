#!/usr/bin/env python3
"""
Modeling Pipeline for L'Oréal Datathon 2025
Implementation focusing on semantic validation, sentiment analysis, and decay detection.

Includes:
1. Semantic Validation using sentence transformers
2. Segment & Sentiment Analysis with spaCy and transformers
3. Decay Detection with time-series analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
import json
import re
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
HAS_SENTENCE_TRANSFORMERS = True
HAS_TRANSFORMERS = True
HAS_SPACY = True
HAS_SKLEARN = True

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    HAS_SKLEARN = False
    logger.warning("sentence-transformers or sklearn not available, semantic validation disabled")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, sentiment analysis disabled")

try:
    import spacy
except ImportError:
    HAS_SPACY = False
    logger.warning("spaCy not available, NER features disabled")

# Define paths
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed" / "dataset"
INTERIM_DIR = ROOT / "data" / "interim"
MODELS_DIR = ROOT / "models"

# Create directories if they don't exist
for dir_path in [INTERIM_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class SemanticValidator:
    """Semantic validation using sentence transformers for embeddings and clustering."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic validator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                self.model = None
        else:
            logger.warning("Sentence transformers not available, semantic validation disabled")
    
    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings or None if model not available
        """
        if not self.model or not texts:
            return None
        
        try:
            # Check cache first
            cache_key = hash(tuple(texts))
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Cache results
            self.embeddings_cache[cache_key] = embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def cluster_semantic_features(self, features: List[str], hashtags: List[str] = None, 
                                n_clusters: int = 5) -> Dict[str, any]:
        """
        Cluster semantically related features and hashtags.
        
        Args:
            features: List of feature names/hashtags
            hashtags: Optional list of hashtag titles
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary containing clustering results
        """
        if not self.model or not features:
            return {}
        
        logger.info(f"Clustering {len(features)} features into {n_clusters} semantic groups")
        
        # Combine features and hashtags for comprehensive clustering
        all_texts = features.copy()
        if hashtags:
            all_texts.extend(hashtags)
        
        # Remove duplicates while preserving order
        all_texts = list(dict.fromkeys(all_texts))
        
        # Generate embeddings
        embeddings = self.generate_embeddings(all_texts)
        if embeddings is None:
            return {}
        
        try:
            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(all_texts)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Organize results
            clusters = {}
            for i, text in enumerate(all_texts):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        'features': [],
                        'center_embedding': kmeans.cluster_centers_[cluster_id],
                        'coherence_score': 0.0
                    }
                clusters[cluster_id]['features'].append(text)
            
            # Calculate coherence scores for each cluster
            for cluster_id, cluster_data in clusters.items():
                cluster_embeddings = embeddings[[i for i, text in enumerate(all_texts) 
                                               if text in cluster_data['features']]]
                if len(cluster_embeddings) > 1:
                    # Calculate average pairwise cosine similarity
                    similarities = cosine_similarity(cluster_embeddings)
                    # Get upper triangle (excluding diagonal)
                    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                    cluster_data['coherence_score'] = np.mean(upper_triangle)
                else:
                    cluster_data['coherence_score'] = 1.0
            
            result = {
                'clusters': clusters,
                'n_clusters': len(clusters),
                'total_features': len(all_texts),
                'average_coherence': np.mean([c['coherence_score'] for c in clusters.values()])
            }
            
            logger.info(f"Created {len(clusters)} semantic clusters with average coherence: {result['average_coherence']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {e}")
            return {}
    
    def find_semantic_trends(self, trend_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Find groups of semantically related trends that are rising together.
        
        Args:
            trend_candidates: DataFrame with trend candidates
            
        Returns:
            Enhanced DataFrame with semantic groupings
        """
        if not self.model or trend_candidates.empty:
            return trend_candidates
        
        logger.info("Analyzing semantic relationships in trend candidates")
        
        # Get features with positive growth
        rising_trends = trend_candidates[
            trend_candidates.get('rate_of_change', 0) > 0
        ].copy()
        
        if rising_trends.empty:
            return trend_candidates
        
        # Cluster rising trends
        features = rising_trends['feature'].unique().tolist()
        clustering_result = self.cluster_semantic_features(features, n_clusters=min(8, len(features)))
        
        if not clustering_result:
            return trend_candidates
        
        # Add cluster information to trends
        feature_to_cluster = {}
        for cluster_id, cluster_data in clustering_result['clusters'].items():
            for feature in cluster_data['features']:
                feature_to_cluster[feature] = {
                    'cluster_id': cluster_id,
                    'coherence_score': cluster_data['coherence_score'],
                    'cluster_size': len(cluster_data['features'])
                }
        
        # Add semantic information to original DataFrame
        trend_candidates['semantic_cluster'] = trend_candidates['feature'].map(
            lambda x: feature_to_cluster.get(x, {}).get('cluster_id', -1)
        )
        trend_candidates['semantic_coherence'] = trend_candidates['feature'].map(
            lambda x: feature_to_cluster.get(x, {}).get('coherence_score', 0.0)
        )
        trend_candidates['cluster_size'] = trend_candidates['feature'].map(
            lambda x: feature_to_cluster.get(x, {}).get('cluster_size', 1)
        )
        
        # Calculate cluster-level trend strength
        cluster_stats = rising_trends.groupby(
            rising_trends['feature'].map(lambda x: feature_to_cluster.get(x, {}).get('cluster_id', -1))
        ).agg({
            'rate_of_change': ['mean', 'count'],
            'count': 'sum'
        }).round(3)
        
        if not cluster_stats.empty:
            logger.info(f"Found {len(cluster_stats)} semantic clusters with rising trends")
        
        return trend_candidates

class DemographicsAnalyzer:
    """Analyze demographics using spaCy NER for age indicators."""
    
    def __init__(self):
        """Initialize demographics analyzer."""
        self.nlp = None
        self.age_patterns = [
            # Direct age mentions
            re.compile(r'\b(\d{1,2})\s*(years?\s*old|yo|y\.o\.)\b', re.IGNORECASE),
            re.compile(r'\bage\s*(\d{1,2})\b', re.IGNORECASE),
            # Generation keywords
            re.compile(r'\b(gen\s*z|generation\s*z|zoomer)\b', re.IGNORECASE),
            re.compile(r'\b(millennial|gen\s*y|generation\s*y)\b', re.IGNORECASE),
            re.compile(r'\b(gen\s*x|generation\s*x)\b', re.IGNORECASE),
            re.compile(r'\b(boomer|baby\s*boomer)\b', re.IGNORECASE),
            # Life stage indicators
            re.compile(r'\b(teen|teenager|teenage)\b', re.IGNORECASE),
            re.compile(r'\b(college|university|student)\b', re.IGNORECASE),
            re.compile(r'\b(young\s*adult|twenty\s*something)\b', re.IGNORECASE),
            re.compile(r'\b(middle\s*aged|mid\s*life)\b', re.IGNORECASE),
        ]
        
        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("Loaded spaCy English model for NER")
            except OSError:
                try:
                    # Try alternative model name
                    self.nlp = spacy.load('en')
                    logger.info("Loaded spaCy English model")
                except OSError:
                    logger.warning("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
                    self.nlp = None
        else:
            logger.warning("spaCy not available, demographics analysis disabled")
    
    def extract_age_indicators(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Extract age indicators from user bios/profile descriptions.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dictionaries with age information
        """
        results = []
        
        for text in texts:
            if not isinstance(text, str):
                results.append({'age_group': 'unknown', 'confidence': 0.0, 'indicators': []})
                continue
            
            indicators = []
            age_group = 'unknown'
            confidence = 0.0
            
            # Check for direct age patterns
            for pattern in self.age_patterns:
                matches = pattern.findall(text.lower())
                if matches:
                    indicators.extend(matches)
            
            # Determine age group based on indicators
            text_lower = text.lower()
            
            if any(term in text_lower for term in ['gen z', 'generation z', 'zoomer']):
                age_group = 'gen_z'
                confidence = 0.8
            elif any(term in text_lower for term in ['teen', 'teenager']):
                age_group = 'teen'
                confidence = 0.9
            elif any(term in text_lower for term in ['college', 'university', 'student']):
                age_group = 'young_adult'
                confidence = 0.7
            elif any(term in text_lower for term in ['millennial', 'gen y']):
                age_group = 'millennial'
                confidence = 0.8
            elif any(term in text_lower for term in ['gen x', 'generation x']):
                age_group = 'gen_x'
                confidence = 0.8
            elif any(term in text_lower for term in ['boomer', 'baby boomer']):
                age_group = 'boomer'
                confidence = 0.8
            
            # Extract numeric ages
            numeric_ages = []
            for pattern in self.age_patterns[:2]:  # Only numeric patterns
                matches = pattern.findall(text)
                for match in matches:
                    try:
                        age = int(match[0] if isinstance(match, tuple) else match)
                        numeric_ages.append(age)
                    except (ValueError, IndexError):
                        continue
            
            # Override age group with numeric age if available
            if numeric_ages:
                avg_age = np.mean(numeric_ages)
                if avg_age <= 17:
                    age_group = 'teen'
                    confidence = 0.9
                elif avg_age <= 26:
                    age_group = 'gen_z'
                    confidence = 0.9
                elif avg_age <= 42:
                    age_group = 'millennial'
                    confidence = 0.9
                elif avg_age <= 57:
                    age_group = 'gen_x'
                    confidence = 0.9
                else:
                    age_group = 'boomer'
                    confidence = 0.9
            
            results.append({
                'age_group': age_group,
                'confidence': confidence,
                'indicators': indicators,
                'numeric_ages': numeric_ages
            })
        
        return results

class CategoryClassifier:
    """Fine-tune transformer models for category classification."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize category classifier.
        
        Args:
            model_name: Name of the transformer model to use
        """
        self.model_name = model_name
        self.classifier = None
        self.categories = ['skincare', 'makeup', 'hair', 'fashion', 'lifestyle']
        
        if HAS_TRANSFORMERS:
            try:
                # Use a pre-trained classification pipeline
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info(f"Loaded zero-shot classifier for category classification")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                self.classifier = None
        else:
            logger.warning("Transformers not available, category classification disabled")
    
    def classify_posts(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Classify posts into categories.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of classification results
        """
        if not self.classifier:
            # Fallback to rule-based classification
            return self._rule_based_classification(texts)
        
        results = []
        
        logger.info(f"Classifying {len(texts)} posts into categories")
        
        for text in tqdm(texts, desc="Classifying posts"):
            if not isinstance(text, str) or not text.strip():
                results.append({
                    'predicted_category': 'lifestyle',
                    'confidence': 0.0,
                    'scores': {cat: 0.0 for cat in self.categories}
                })
                continue
            
            try:
                result = self.classifier(text, self.categories)
                
                predicted_category = result['labels'][0]
                confidence = result['scores'][0]
                
                # Create scores dictionary
                scores = {}
                for label, score in zip(result['labels'], result['scores']):
                    scores[label] = score
                
                results.append({
                    'predicted_category': predicted_category,
                    'confidence': confidence,
                    'scores': scores
                })
                
            except Exception as e:
                logger.warning(f"Classification failed for text: {e}")
                results.append({
                    'predicted_category': 'lifestyle',
                    'confidence': 0.0,
                    'scores': {cat: 0.0 for cat in self.categories}
                })
        
        return results
    
    def _rule_based_classification(self, texts: List[str]) -> List[Dict[str, any]]:
        """Fallback rule-based classification."""
        
        category_keywords = {
            'skincare': ['skin', 'skincare', 'serum', 'moisturizer', 'cleanser', 'spf', 'sunscreen', 
                        'retinol', 'niacinamide', 'hyaluronic', 'vitamin c', 'acne', 'wrinkle'],
            'makeup': ['makeup', 'foundation', 'concealer', 'mascara', 'lipstick', 'eyeshadow', 
                      'blush', 'highlighter', 'contour', 'bronzer', 'primer'],
            'hair': ['hair', 'shampoo', 'conditioner', 'styling', 'scalp', 'hair care', 'keratin'],
            'fashion': ['fashion', 'style', 'outfit', 'clothing', 'dress', 'shoes', 'accessories'],
            'lifestyle': ['lifestyle', 'life', 'daily', 'routine', 'wellness', 'health']
        }
        
        results = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            text_lower = text.lower()
            scores = {}
            
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[category] = score / len(keywords)  # Normalize
            
            if max(scores.values()) > 0:
                predicted_category = max(scores, key=scores.get)
                confidence = scores[predicted_category]
            else:
                predicted_category = 'lifestyle'
                confidence = 0.1
            
            results.append({
                'predicted_category': predicted_category,
                'confidence': confidence,
                'scores': scores
            })
        
        return results

class SentimentAnalyzer:
    """Sentiment analysis using pre-trained models."""
    
    def __init__(self, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment'):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: Name of the sentiment analysis model
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        
        if HAS_TRANSFORMERS:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
                logger.info(f"Loaded sentiment analysis model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentiment model {model_name}, using default")
                try:
                    self.sentiment_pipeline = pipeline("sentiment-analysis")
                    logger.info("Loaded default sentiment analysis model")
                except Exception as e2:
                    logger.error(f"Failed to load any sentiment model: {e2}")
                    self.sentiment_pipeline = None
        else:
            logger.warning("Transformers not available, sentiment analysis disabled")
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Analyze sentiment of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment analysis results
        """
        if not self.sentiment_pipeline:
            return self._rule_based_sentiment(texts)
        
        results = []
        
        logger.info(f"Analyzing sentiment for {len(texts)} texts")
        
        for text in tqdm(texts, desc="Analyzing sentiment"):
            if not isinstance(text, str) or not text.strip():
                results.append({
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'score': 0.0
                })
                continue
            
            try:
                # Truncate text if too long
                text_truncated = text[:500]
                
                result = self.sentiment_pipeline(text_truncated)
                
                if isinstance(result, list):
                    result = result[0]
                
                sentiment = result['label'].lower()
                confidence = result['score']
                
                # Convert to standardized sentiment labels
                if sentiment in ['positive', 'pos']:
                    sentiment = 'positive'
                    score = confidence
                elif sentiment in ['negative', 'neg']:
                    sentiment = 'negative'
                    score = -confidence
                else:
                    sentiment = 'neutral'
                    score = 0.0
                
                results.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'score': score
                })
                
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                results.append({
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'score': 0.0
                })
        
        return results
    
    def _rule_based_sentiment(self, texts: List[str]) -> List[Dict[str, any]]:
        """Fallback rule-based sentiment analysis."""
        
        positive_words = {
            'love', 'amazing', 'great', 'awesome', 'fantastic', 'excellent', 'perfect',
            'beautiful', 'gorgeous', 'stunning', 'wonderful', 'incredible', 'best',
            'favorite', 'obsessed', 'recommend', 'holy grail'
        }
        
        negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'worst', 'bad', 'disappointed',
            'annoying', 'irritating', 'useless', 'waste', 'regret', 'broke out'
        }
        
        results = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'positive'
                confidence = min(positive_count / 10, 1.0)
                score = confidence
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = min(negative_count / 10, 1.0)
                score = -confidence
            else:
                sentiment = 'neutral'
                confidence = 0.5
                score = 0.0
            
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'score': score
            })
        
        return results

class DecayDetector:
    """Detect trend decay using time-series analysis."""
    
    def __init__(self, period_threshold: int = 3):
        """
        Initialize decay detector.
        
        Args:
            period_threshold: Number of periods to analyze for decay
        """
        self.period_threshold = period_threshold
    
    def calculate_derivatives(self, time_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate first and second derivatives of time series.
        
        Args:
            time_series: Time series data
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # First derivative (growth rate)
        first_derivative = time_series.diff()
        
        # Second derivative (acceleration)
        second_derivative = first_derivative.diff()
        
        return first_derivative, second_derivative
    
    def detect_decay(self, trend_data: pd.DataFrame, period_T: int = 3) -> pd.DataFrame:
        """
        Detect decay in confirmed trends.
        
        Args:
            trend_data: DataFrame with trend data including time series
            period_T: Period length for decay detection
            
        Returns:
            DataFrame with decay detection results
        """
        if trend_data.empty:
            return trend_data
        
        logger.info(f"Detecting decay in {len(trend_data)} trend records")
        
        # Ensure data is sorted by time
        if 'timestamp' in trend_data.columns:
            trend_data = trend_data.sort_values('timestamp')
        elif 'time_bin' in trend_data.columns:
            trend_data = trend_data.sort_values('time_bin')
        
        decay_results = []
        
        # Group by feature to analyze each trend separately
        for feature, group in trend_data.groupby('feature'):
            if len(group) < period_T + 1:
                # Not enough data points
                continue
            
            # Get engagement time series
            if 'count' in group.columns:
                engagement_series = group['count']
            else:
                continue
            
            # Calculate derivatives
            growth_rate, acceleration = self.calculate_derivatives(engagement_series)
            
            # Check decay condition for the last period_T points
            recent_growth = growth_rate.tail(period_T)
            recent_acceleration = acceleration.tail(period_T)
            
            # Rule: if growth_rate > 0 and acceleration < 0 for period T
            decay_condition = (
                (recent_growth > 0).all() and 
                (recent_acceleration < 0).all() and
                len(recent_acceleration.dropna()) >= period_T - 1
            )
            
            if decay_condition:
                trend_state = "Decaying"
                decay_confidence = min(1.0, abs(recent_acceleration.mean()) / max(recent_growth.mean(), 0.001))
            else:
                # Additional checks for other states
                if (recent_growth > 0).all():
                    if (recent_acceleration > 0).any():
                        trend_state = "Accelerating"
                    else:
                        trend_state = "Growing"
                elif (recent_growth < 0).all():
                    trend_state = "Declining"
                else:
                    trend_state = "Stable"
                
                decay_confidence = 0.0
            
            # Calculate additional metrics
            total_change = engagement_series.iloc[-1] - engagement_series.iloc[0] if len(engagement_series) > 1 else 0
            avg_growth_rate = growth_rate.mean() if not growth_rate.empty else 0
            avg_acceleration = acceleration.mean() if not acceleration.empty else 0
            
            decay_results.append({
                'feature': feature,
                'trend_state': trend_state,
                'decay_confidence': decay_confidence,
                'avg_growth_rate': avg_growth_rate,
                'avg_acceleration': avg_acceleration,
                'total_change': total_change,
                'periods_analyzed': len(recent_growth),
                'latest_growth': recent_growth.iloc[-1] if not recent_growth.empty else 0,
                'latest_acceleration': recent_acceleration.iloc[-1] if not recent_acceleration.empty else 0
            })
        
        # Create results DataFrame
        decay_df = pd.DataFrame(decay_results)
        
        if not decay_df.empty:
            # Merge with original data
            trend_data_enhanced = trend_data.merge(
                decay_df[['feature', 'trend_state', 'decay_confidence', 'avg_growth_rate', 'avg_acceleration']], 
                on='feature', 
                how='left'
            )
            
            # Fill missing values
            trend_data_enhanced['trend_state'] = trend_data_enhanced['trend_state'].fillna('Unknown')
            trend_data_enhanced['decay_confidence'] = trend_data_enhanced['decay_confidence'].fillna(0.0)
            
            logger.info(f"Decay detection completed. Found {len(decay_df[decay_df['trend_state'] == 'Decaying'])} decaying trends")
            
            return trend_data_enhanced
        else:
            # Add empty columns if no results
            trend_data['trend_state'] = 'Unknown'
            trend_data['decay_confidence'] = 0.0
            trend_data['avg_growth_rate'] = 0.0
            trend_data['avg_acceleration'] = 0.0
            
            return trend_data

class ModelingPipeline:
    """Main modeling pipeline combining all components."""
    
    def __init__(self):
        """Initialize modeling pipeline."""
        self.semantic_validator = SemanticValidator()
        self.demographics_analyzer = DemographicsAnalyzer()
        self.category_classifier = CategoryClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.decay_detector = DecayDetector()
    
    def run_semantic_validation(self, trend_data: pd.DataFrame) -> pd.DataFrame:
        """Run semantic validation on trend data."""
        logger.info("Running semantic validation")
        return self.semantic_validator.find_semantic_trends(trend_data)
    
    def run_demographic_analysis(self, user_bios: List[str]) -> pd.DataFrame:
        """Run demographic analysis on user bios."""
        logger.info("Running demographic analysis")
        
        demographics_results = self.demographics_analyzer.extract_age_indicators(user_bios)
        
        return pd.DataFrame(demographics_results)
    
    def run_category_classification(self, posts: List[str]) -> pd.DataFrame:
        """Run category classification on posts."""
        logger.info("Running category classification")
        
        classification_results = self.category_classifier.classify_posts(posts)
        
        return pd.DataFrame(classification_results)
    
    def run_sentiment_analysis(self, texts: List[str]) -> pd.DataFrame:
        """Run sentiment analysis on texts."""
        logger.info("Running sentiment analysis")
        
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(texts)
        
        return pd.DataFrame(sentiment_results)
    
    def run_decay_detection(self, trend_data: pd.DataFrame) -> pd.DataFrame:
        """Run decay detection on trend data."""
        logger.info("Running decay detection")
        return self.decay_detector.detect_decay(trend_data)
    
    def run_full_pipeline(self, data_sources: Dict[str, any]) -> Dict[str, any]:
        """
        Run the complete modeling pipeline.
        
        Args:
            data_sources: Dictionary containing various data sources
            
        Returns:
            Dictionary containing all modeling results
        """
        logger.info("Starting full modeling pipeline")
        
        results = {
            'semantic_validation': None,
            'demographics': None,
            'category_classification': None,
            'sentiment_analysis': None,
            'decay_detection': None
        }
        
        # Semantic validation on trend candidates
        if 'trend_candidates' in data_sources and data_sources['trend_candidates'] is not None:
            trend_data = data_sources['trend_candidates']
            if not trend_data.empty:
                results['semantic_validation'] = self.run_semantic_validation(trend_data)
                results['decay_detection'] = self.run_decay_detection(trend_data)
        
        # Demographic analysis on user bios
        if 'user_bios' in data_sources:
            user_bios = data_sources['user_bios']
            if user_bios:
                results['demographics'] = self.run_demographic_analysis(user_bios)
        
        # Category classification on posts
        if 'posts' in data_sources:
            posts = data_sources['posts']
            if posts:
                results['category_classification'] = self.run_category_classification(posts)
        
        # Sentiment analysis on text data
        if 'texts' in data_sources:
            texts = data_sources['texts']
            if texts:
                results['sentiment_analysis'] = self.run_sentiment_analysis(texts)
        
        # Save results
        self.save_results(results)
        
        logger.info("Modeling pipeline completed successfully")
        return results
    
    def save_results(self, results: Dict) -> None:
        """Save modeling results to files."""
        
        for result_name, df in results.items():
            if df is not None and not df.empty:
                # Convert any list columns to strings for parquet compatibility
                df_copy = df.copy()
                for col in df_copy.columns:
                    if df_copy[col].dtype == 'object':
                        # Check if column contains lists
                        sample_val = df_copy[col].dropna().iloc[0] if not df_copy[col].dropna().empty else None
                        if isinstance(sample_val, list):
                            df_copy[col] = df_copy[col].apply(lambda x: str(x) if x is not None else None)
                
                filepath = INTERIM_DIR / f"modeling_{result_name}.parquet"
                df_copy.to_parquet(filepath, index=False)
                logger.info(f"Saved {result_name} results to {filepath}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results_generated': [k for k, v in results.items() if v is not None and not v.empty],
            'total_records': {k: len(v) if v is not None else 0 for k, v in results.items()}
        }
        
        with open(INTERIM_DIR / 'modeling_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Main function to run the modeling pipeline."""
    
    logger.info("L'Oréal Datathon 2025 - Modeling Pipeline")
    
    # Example usage with sample data
    sample_data = {
        'trend_candidates': pd.DataFrame({
            'feature': ['skincare', 'makeup', 'niacinamide', 'retinol', 'contour'],
            'count': [100, 80, 120, 90, 60],
            'rate_of_change': [0.15, 0.10, 0.25, 0.08, -0.05],
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='6H')
        }),
        'user_bios': [
            "22yo beauty enthusiast and Gen Z makeup lover",
            "College student passionate about skincare",
            "Millennial mom sharing beauty tips",
            "Teen interested in fashion trends"
        ],
        'posts': [
            "Love this new skincare routine with niacinamide!",
            "Best makeup tutorial for beginners",
            "Hair care tips for damaged hair",
            "Fashion haul from my favorite brands"
        ],
        'texts': [
            "This product is amazing! Highly recommend",
            "Not impressed with this purchase, waste of money",
            "Good value for the price, will buy again",
            "Absolutely love the results!"
        ]
    }
    
    # Initialize modeling pipeline
    pipeline = ModelingPipeline()
    
    # Run pipeline
    results = pipeline.run_full_pipeline(sample_data)
    
    # Print summary
    print("\n" + "="*60)
    print("MODELING PIPELINE SUMMARY")
    print("="*60)
    
    for result_name, df in results.items():
        if df is not None and not df.empty:
            print(f"{result_name.upper()}: {len(df)} records")
            if result_name == 'semantic_validation' and 'semantic_cluster' in df.columns:
                clusters = df['semantic_cluster'].nunique()
                print(f"  Semantic clusters: {clusters}")
            elif result_name == 'demographics' and 'age_group' in df.columns:
                age_groups = df['age_group'].value_counts()
                print(f"  Age groups: {dict(age_groups)}")
            elif result_name == 'sentiment_analysis' and 'sentiment' in df.columns:
                sentiments = df['sentiment'].value_counts()
                print(f"  Sentiments: {dict(sentiments)}")
            elif result_name == 'decay_detection' and 'trend_state' in df.columns:
                states = df['trend_state'].value_counts()
                print(f"  Trend states: {dict(states)}")
        else:
            print(f"{result_name.upper()}: No data")
    
    print("="*60)

if __name__ == "__main__":
    main()