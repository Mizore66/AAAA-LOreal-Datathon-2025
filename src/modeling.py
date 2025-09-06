# Placeholder for modeling utilities (Phase 3)

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Set, Any
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import logging
import warnings

# Global warning suppression as requested (mute all warnings)
warnings.filterwarnings("ignore")
for noisy_logger in [
    'prophet', 'cmdstanpy', 'urllib3', 'matplotlib', 'fbprophet', 'transformers'
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Optional / heavyweight dependencies are guarded so the script can run in a
# constrained environment (current workspace has only core libs installed).
# Flags below let later code fall back gracefully if a library is missing.
# ---------------------------------------------------------------------------
HAS_SKLEARN = True
HAS_STATSMODELS = True
HAS_PROPHET = True
HAS_TRANSFORMERS = True
HAS_TORCH = True
HAS_SPACY = True

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        classification_report, confusion_matrix, mean_absolute_error, 
        mean_squared_error, r2_score, f1_score, precision_score, recall_score
    )
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SKLEARN = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.seasonal import STL
except Exception:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet  # fbprophet / prophet
except Exception:
    HAS_PROPHET = False

try:
    from transformers import pipeline
except Exception:
    HAS_TRANSFORMERS = False

try:
    import torch
except Exception:
    HAS_TORCH = False

try:
    import spacy
except Exception:
    HAS_SPACY = False

try:
    import scipy.stats as stats
except Exception:
    # Only used lightly; provide tiny shim
    class _StatsShim:
        def zscore(self, a, nan_policy='omit'):
            a = np.asarray(a, dtype=float)
            m = np.nanmean(a)
            s = np.nanstd(a) or 1.0
            return (a - m) / s
    stats = _StatsShim()

# Setup paths

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
INTERIM_DIR = ROOT / "data" / "interim"

# Supported aggregation frequencies we will attempt to load
AGG_FREQUENCIES = [
    "1h", "3h", "6h", "1d", "3d", "7d", "14d", "1m", "3m", "6m"
]

# Create directories
for dir_path in [MODELS_DIR, INTERIM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    feature: str
    timestamp: pd.Timestamp
    score: float
    platform: Optional[str] = None
    category: Optional[str] = None
    anomaly_type: Optional[str] = None
    residual_value: Optional[float] = None
    trend_component: Optional[float] = None
    seasonal_component: Optional[float] = None
    cross_platform_correlation: Optional[float] = None
    semantic_cluster: Optional[int] = None  # MiniLM / primary model
    distilbert_cluster: Optional[int] = None  # Secondary DistilBERT semantic cluster


@dataclass
class TrendState:
    """Data class for trend lifecycle state."""
    feature: str
    state: str  # "Emerging", "Growing", "Peak", "Decaying", "Dormant"
    growth_rate: float
    acceleration: float
    confidence: float
    timestamp: pd.Timestamp


@dataclass
class SemanticCluster:
    """Data class for semantic clustering results."""
    cluster_id: int
    features: List[str]
    centroid_embedding: np.ndarray
    cluster_size: int
    avg_similarity: float


@dataclass
class ModelMetrics:
    """Data class for model performance metrics."""
    model_name: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    cross_val_mean: Optional[float] = None
    cross_val_std: Optional[float] = None


@dataclass
class ForecastResult:
    """Data class for forecasting results."""
    feature: str
    predictions: pd.DataFrame
    model_metrics: ModelMetrics
    forecast_horizon: int
    confidence_intervals: Optional[pd.DataFrame] = None


class STLAnomalyDetector:
    """STL-based anomaly detection; falls back to simple z-score if statsmodels absent."""

    def __init__(self, period: int = 24, seasonal: int = 7, threshold_std: float = 3.0):
        self.period = period
        self.seasonal = seasonal
        self.threshold_std = threshold_std
        self.decomposition_results = {}

    def decompose_time_series(self, ts_data: pd.Series, feature_name: str) -> Optional[Dict[str, pd.Series]]:
        if not HAS_STATSMODELS:
            return None  # Will trigger fallback path
        try:
            if len(ts_data) < self.period * 2:
                logger.warning(f"Time series too short for STL decomposition: {feature_name}")
                return None
            stl = STL(ts_data, seasonal=self.seasonal, period=self.period)
            decomposition = stl.fit()
            result = {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'resid': decomposition.resid
            }
            self.decomposition_results[feature_name] = result
            return result
        except Exception as e:
            logger.error(f"STL decomposition failed for {feature_name}: {e}")
            return None

    def detect_anomalies(self, ts_data: pd.Series, feature_name: str, platform: Optional[str] = None) -> List[Anomaly]:
        decomposition = self.decompose_time_series(ts_data, feature_name)
        anomalies: List[Anomaly] = []
        if decomposition is None:
            # Fallback: simple z-score based anomaly detection
            series = ts_data.astype(float)
            if len(series) < 5:
                return []
            z = stats.zscore(series.values)
            if isinstance(z, np.ndarray):
                threshold = 3.0
                for idx, zval in zip(series.index, z):
                    if abs(zval) > threshold:
                        anomalies.append(Anomaly(
                            feature=feature_name,
                            timestamp=idx,
                            score=float(abs(zval)),
                            platform=platform,
                            residual_value=float(zval),
                            anomaly_type="ZScore"
                        ))
            if anomalies:
                logger.info(f"Detected {len(anomalies)} anomalies for {feature_name} (fallback z-score)")
            return anomalies

        residuals = decomposition['resid']
        trend = decomposition['trend']
        seasonal = decomposition['seasonal']
        residual_std = residuals.std() or 1.0
        threshold = self.threshold_std * residual_std
        anomaly_mask = np.abs(residuals) > threshold
        anomaly_indices = ts_data.index[anomaly_mask]
        for idx in anomaly_indices:
            # Safe extraction (duplicate indices may return Series)
            resid_val = residuals.loc[idx]
            if isinstance(resid_val, pd.Series):
                resid_val = resid_val.iloc[0]
            trend_val = trend.loc[idx]
            if isinstance(trend_val, pd.Series):
                trend_val = trend_val.iloc[0]
            seasonal_val = seasonal.loc[idx]
            if isinstance(seasonal_val, pd.Series):
                seasonal_val = seasonal_val.iloc[0]
            score_val = abs(resid_val) / residual_std if residual_std else 0.0
            anomalies.append(Anomaly(
                feature=feature_name,
                timestamp=idx,
                score=float(score_val),
                platform=platform,
                residual_value=float(resid_val),
                trend_component=float(trend_val) if trend_val is not None and not pd.isna(trend_val) else None,
                seasonal_component=float(seasonal_val) if seasonal_val is not None and not pd.isna(seasonal_val) else None,
                anomaly_type="STL_Residual"
            ))
        if anomalies:
            logger.info(f"Detected {len(anomalies)} anomalies for {feature_name}")
        return anomalies


class CrossPlatformValidator:
    """Cross-platform validation for trend credibility."""
    
    def __init__(self, correlation_threshold: float = 0.2):
        self.correlation_threshold = correlation_threshold
        
    def calculate_cross_platform_correlation(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate correlations between features across platforms.
        
        Args:
            data: DataFrame with columns ['feature', 'platform', 'bin', 'count']
            
        Returns:
            Dictionary mapping feature pairs to correlation coefficients
        """
        correlations = {}
        
        # Group by feature to find cross-platform correlations
        features = data['feature'].unique()
        platforms = data['platform'].unique() if 'platform' in data.columns else ['unknown']
        
        for feature in features:
            feature_data = data[data['feature'] == feature]
            
            if len(platforms) > 1:
                # Pivot to get platform columns
                pivot_data = feature_data.pivot_table(
                    values='count', 
                    index='bin', 
                    columns='platform', 
                    fill_value=0
                )
                
                # Calculate correlations between platforms for this feature
                if len(pivot_data.columns) > 1:
                    corr_matrix = pivot_data.corr()
                    for i, platform1 in enumerate(pivot_data.columns):
                        for j, platform2 in enumerate(pivot_data.columns):
                            if i < j:  # Avoid duplicates
                                key = f"{feature}_{platform1}_{platform2}"
                                correlations[key] = corr_matrix.loc[platform1, platform2]
        
        return correlations
    
    def validate_anomalies(self, anomalies: List[Anomaly], correlations: Dict[str, float]) -> List[Anomaly]:
        """
        Validate anomalies using cross-platform correlations.
        
        Args:
            anomalies: List of detected anomalies
            correlations: Cross-platform correlation dictionary
            
        Returns:
            List of validated anomalies with correlation scores
        """
        validated_anomalies = []
        
        for anomaly in anomalies:
            # Find relevant correlation for this anomaly
            max_correlation = 0.0
            for corr_key, corr_value in correlations.items():
                if anomaly.feature in corr_key and (anomaly.platform is None or anomaly.platform in corr_key):
                    max_correlation = max(max_correlation, abs(corr_value))
            
            # Update anomaly with correlation info
            anomaly.cross_platform_correlation = max_correlation
            
            # If no cross-platform data available, still keep the anomaly but with lower confidence
            # Only filter out if we have strong negative evidence
            if max_correlation >= self.correlation_threshold or len(correlations) == 0:
                validated_anomalies.append(anomaly)
            elif max_correlation == 0.0:  # No correlation data found - keep it
                validated_anomalies.append(anomaly)
        
        logger.info(f"Validated {len(validated_anomalies)} out of {len(anomalies)} anomalies")
        return validated_anomalies


class SemanticAnalyzer:
    """Semantic validation using sentence transformers (primary + optional DistilBERT)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", secondary_model: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.secondary_model_name = secondary_model
        self.embedding_pipeline = None
        self.secondary_pipeline = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.secondary_cache: Dict[str, np.ndarray] = {}

    def _init_pipeline(self, name: str):
        try:
            return pipeline(
                "feature-extraction",
                model=name,
                device=0 if HAS_TORCH and 'cuda' in str(getattr(torch, 'device', '')) and torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Failed to load transformer model {name}: {e}")
            return None

    def initialize_models(self):
        if self.embedding_pipeline is None:
            self.embedding_pipeline = self._init_pipeline(self.model_name)
            if self.embedding_pipeline:
                logger.info(f"Initialized semantic model: {self.model_name}")
        if self.secondary_pipeline is None:
            self.secondary_pipeline = self._init_pipeline(self.secondary_model_name)
            if self.secondary_pipeline:
                logger.info(f"Initialized DistilBERT secondary model: {self.secondary_model_name}")

    def _embed_list(self, texts: List[str], pipeline_obj, cache: Dict[str, np.ndarray], expected_dim: int = 384) -> np.ndarray:
        if pipeline_obj is None:
            # Fallback TF-IDF
            vectorizer = TfidfVectorizer(max_features=expected_dim, stop_words='english')
            return vectorizer.fit_transform(texts).toarray()
        vectors = []
        for t in texts:
            if t in cache:
                vectors.append(cache[t])
                continue
            try:
                emb_tokens = pipeline_obj(t)[0]
                vec = np.mean(emb_tokens, axis=0)
            except Exception:
                vec = np.zeros(expected_dim)
            cache[t] = vec
            vectors.append(vec)
        return np.vstack(vectors)

    def get_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.initialize_models()
        primary = self._embed_list(texts, self.embedding_pipeline, self.embeddings_cache, expected_dim=384)
        secondary = self._embed_list(texts, self.secondary_pipeline, self.secondary_cache, expected_dim=768)
        return primary, secondary

    def cluster_features(self, features: List[str], n_clusters: int = 5) -> Tuple[List[SemanticCluster], List[SemanticCluster]]:
        if len(features) < n_clusters:
            n_clusters = max(1, len(features)//2 or 1)
        prim_emb, sec_emb = self.get_embeddings(features)
        clusters_primary: List[SemanticCluster] = []
        clusters_secondary: List[SemanticCluster] = []
        # Primary clustering
        try:
            k1 = KMeans(n_clusters=n_clusters, random_state=42)
            labels1 = k1.fit_predict(prim_emb)
            for i in range(n_clusters):
                feats = [features[j] for j in range(len(features)) if labels1[j] == i]
                if not feats:
                    continue
                emb_grp = prim_emb[labels1 == i]
                centroid = emb_grp.mean(axis=0)
                sims = cosine_similarity(emb_grp)
                avg_sim = np.mean(sims[np.triu_indices_from(sims, k=1)]) if emb_grp.shape[0] > 1 else 1.0
                clusters_primary.append(SemanticCluster(i, feats, centroid, len(feats), float(avg_sim)))
        except Exception as e:
            logger.error(f"Primary clustering failed: {e}")
        # Secondary clustering (DistilBERT)
        try:
            k2 = KMeans(n_clusters=n_clusters, random_state=42)
            labels2 = k2.fit_predict(sec_emb)
            for i in range(n_clusters):
                feats = [features[j] for j in range(len(features)) if labels2[j] == i]
                if not feats:
                    continue
                emb_grp = sec_emb[labels2 == i]
                centroid = emb_grp.mean(axis=0)
                sims = cosine_similarity(emb_grp)
                avg_sim = np.mean(sims[np.triu_indices_from(sims, k=1)]) if emb_grp.shape[0] > 1 else 1.0
                clusters_secondary.append(SemanticCluster(i, feats, centroid, len(feats), float(avg_sim)))
        except Exception as e:
            logger.error(f"Secondary (DistilBERT) clustering failed: {e}")
        return clusters_primary, clusters_secondary

    def annotate_anomalies(self, anomalies: List[Anomaly], clusters_primary: List[SemanticCluster], clusters_secondary: List[SemanticCluster]) -> List[Anomaly]:
        fmap_primary = {}
        fmap_secondary = {}
        for c in clusters_primary:
            for f in c.features:
                fmap_primary[f] = c.cluster_id
        for c in clusters_secondary:
            for f in c.features:
                fmap_secondary[f] = c.cluster_id
        for an in anomalies:
            if an.feature in fmap_primary:
                an.semantic_cluster = fmap_primary[an.feature]
            if an.feature in fmap_secondary:
                an.distilbert_cluster = fmap_secondary[an.feature]
        return anomalies


class DemographicAnalyzer:
    """Demographic and category analysis using rule-based approaches."""
    
    def __init__(self):
        self.age_indicators = {
            'gen_z': ['gen z', 'genz', 'zoomer', 'tiktok', 'university', 'college', '18', '19', '20', '21', '22'],
            'millennial': ['millennial', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'],
            'gen_x': ['gen x', 'genx', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45'],
            'boomer': ['boomer', 'baby boomer', '50', '55', '60', '65']
        }
        
        self.category_keywords = {
            'skincare': ['skincare', 'skin care', 'serum', 'moisturizer', 'cleanser', 'toner', 'niacinamide', 'retinol', 'hyaluronic'],
            'makeup': ['makeup', 'foundation', 'concealer', 'mascara', 'lipstick', 'eyeshadow', 'blush', 'bronzer'],
            'hair': ['hair', 'shampoo', 'conditioner', 'hair mask', 'hair oil', 'styling', 'hair color'],
            'lifestyle': ['lifestyle', 'wellness', 'self care', 'routine', 'morning routine', 'night routine']
        }
    
    def analyze_demographics(self, text_data: List[str]) -> Dict[str, int]:
        """
        Analyze demographic indicators in text data.
        
        Args:
            text_data: List of text content (bios, descriptions, etc.)
            
        Returns:
            Dictionary with demographic counts
        """
        demographics = {age_group: 0 for age_group in self.age_indicators.keys()}
        
        for text in text_data:
            text_lower = text.lower()
            for age_group, indicators in self.age_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    demographics[age_group] += 1
        
        return demographics
    
    def classify_category(self, feature_text: str) -> str:
        """
        Classify a feature into beauty categories.
        
        Args:
            feature_text: Text to classify
            
        Returns:
            Category name
        """
        text_lower = feature_text.lower()
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'other'


class SentimentAnalyzer:
    """Sentiment analysis for trend-related content (CardiffNLP)."""

    def __init__(self):
        self.sentiment_pipeline = None
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    def initialize_sentiment_model(self):
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if HAS_TORCH and torch.cuda.is_available() else -1
            )
            logger.info(f"Initialized sentiment model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize CardiffNLP sentiment model: {e}")
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                logger.info("Using generic fallback sentiment model")
            except Exception:
                self.sentiment_pipeline = None

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        if self.sentiment_pipeline is None:
            self.initialize_sentiment_model()
        if self.sentiment_pipeline is None:
            return [{'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} for _ in texts]
        outputs = []
        for t in texts:
            try:
                res = self.sentiment_pipeline(t)
                sent = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                for item in res:
                    lab = item['label'].lower()
                    sc = float(item['score'])
                    if 'positive' in lab or 'pos' in lab:
                        sent['positive'] = sc
                    elif 'negative' in lab or 'neg' in lab:
                        sent['negative'] = sc
                    elif 'neutral' in lab:
                        sent['neutral'] = sc
                # Normalize if sums to >1 due to independent scoring variants
                ssum = sum(sent.values()) or 1.0
                for k in sent:
                    sent[k] /= ssum
                outputs.append(sent)
            except Exception:
                outputs.append({'positive': 0.33, 'negative': 0.33, 'neutral': 0.34})
        return outputs


class TrendDecayDetector:
    """Detect trend decay using growth rate and acceleration analysis."""
    
    def __init__(self, decay_period: int = 7):
        self.decay_period = decay_period
    
    def calculate_derivatives(self, ts_data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate first and second derivatives of time series.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # First derivative (growth rate)
        first_derivative = ts_data.diff()
        
        # Second derivative (acceleration)
        second_derivative = first_derivative.diff()
        
        return first_derivative, second_derivative
    
    def detect_trend_state(self, ts_data: pd.Series, feature_name: str) -> List[TrendState]:
        """
        Detect trend lifecycle states.
        
        Args:
            ts_data: Time series data
            feature_name: Name of the feature
            
        Returns:
            List of trend states over time
        """
        first_deriv, second_deriv = self.calculate_derivatives(ts_data)
        
        trend_states = []
        
        for i in range(len(ts_data)):
            if i < self.decay_period:
                continue
                
            # Look at recent period
            recent_growth = first_deriv.iloc[i-self.decay_period:i+1]
            recent_accel = second_deriv.iloc[i-self.decay_period:i+1]
            
            growth_rate = recent_growth.mean() if not recent_growth.isna().all() else 0
            acceleration = recent_accel.mean() if not recent_accel.isna().all() else 0
            
            # Determine trend state
            if growth_rate > 0 and acceleration > 0:
                state = "Growing"
                confidence = min(0.9, abs(growth_rate) * abs(acceleration) / 100)
            elif growth_rate > 0 and acceleration < 0:
                state = "Decaying"
                confidence = min(0.9, abs(growth_rate) * abs(acceleration) / 100)
            elif growth_rate < 0:
                state = "Dormant"
                confidence = min(0.8, abs(growth_rate) / 10)
            elif abs(growth_rate) < 0.1:
                state = "Peak"
                confidence = 0.6
            else:
                state = "Emerging"
                confidence = 0.5
            
            trend_state = TrendState(
                feature=feature_name,
                state=state,
                growth_rate=growth_rate,
                acceleration=acceleration,
                confidence=confidence,
                timestamp=ts_data.index[i]
            )
            trend_states.append(trend_state)
        
        return trend_states


class TrendDetectionModel:
    """
    Comprehensive trend detection model implementing STL-based anomaly detection
    with cross-platform validation, semantic analysis, and decay detection.
    """
    
    def __init__(self, contamination: float = 0.1, period: int = 24, threshold_std: float = 3.0):
        self.contamination = contamination
        self.period = period
        self.threshold_std = threshold_std
        
        # Initialize components
        self.stl_detector = STLAnomalyDetector(period=period, threshold_std=threshold_std)
        self.cross_platform_validator = CrossPlatformValidator()
        # Semantic analyzers (MiniLM + DistilBERT hybrid)
        self.semantic_analyzer = SemanticAnalyzer(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            secondary_model="distilbert-base-uncased"
        )
        self.demographic_analyzer = DemographicAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.decay_detector = TrendDecayDetector()
        
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit the trend detection model."""
        self.is_fitted = True
        logger.info("STL-based trend detection model fitted successfully")
    
    def predict_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using the comprehensive STL-based approach.
        
        Args:
            df: DataFrame with trend data
            
        Returns:
            DataFrame with detected anomalies and additional analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        all_anomalies = []
        all_trend_states = []
        
        # Process each feature individually
        features = df['feature'].unique() if 'feature' in df.columns else [df.columns[0]]
        
        for feature in features:
            feature_data = df[df['feature'] == feature] if 'feature' in df.columns else df
            
            # Create time series
            if 'bin' in feature_data.columns:
                ts_data = feature_data.set_index('bin')['count']
            else:
                ts_data = feature_data.iloc[:, -1]  # Assume last column is the value
            
            ts_data.index = pd.to_datetime(ts_data.index)
            ts_data = ts_data.sort_index()
            
            platform = feature_data['platform'].iloc[0] if 'platform' in feature_data.columns else None
            
            # 1. STL-based anomaly detection
            feature_anomalies = self.stl_detector.detect_anomalies(ts_data, feature, platform)
            all_anomalies.extend(feature_anomalies)
            
            # 2. Decay detection
            trend_states = self.decay_detector.detect_trend_state(ts_data, feature)
            all_trend_states.extend(trend_states)
        
        # 3. Cross-platform validation
        if len(all_anomalies) > 0:
            correlations = self.cross_platform_validator.calculate_cross_platform_correlation(df)
            all_anomalies = self.cross_platform_validator.validate_anomalies(all_anomalies, correlations)
        
        # 4. Semantic validation
        feature_names = [anomaly.feature for anomaly in all_anomalies]
        if feature_names:
            unique_features = list(set(feature_names))
            clusters_primary, clusters_secondary = self.semantic_analyzer.cluster_features(unique_features)
            all_anomalies = self.semantic_analyzer.annotate_anomalies(all_anomalies, clusters_primary, clusters_secondary)
        
        # Convert results to DataFrame
        if all_anomalies:
            anomaly_records = []
            for anomaly in all_anomalies:
                record = asdict(anomaly)
                record['timestamp'] = record['timestamp'].isoformat() if record['timestamp'] else None
                anomaly_records.append(record)
            
            results_df = pd.DataFrame(anomaly_records)
        else:
            results_df = pd.DataFrame()
        
        logger.info(f"Detected {len(all_anomalies)} anomalies using STL-based approach")
        return results_df


class TimeSeriesForecaster:
    """Time series forecasting with multiple models (Prophet optional)."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prepare_ts_data(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Prepare time series data for a specific feature."""
        feature_data = df[df['feature'] == feature].copy()
        feature_data = feature_data.sort_values('bin').reset_index(drop=True)
        feature_data['ds'] = pd.to_datetime(feature_data['bin'])
        # Prophet cannot handle timezone-aware datetimes; strip any tz info
        try:
            if feature_data['ds'].dt.tz is not None:
                feature_data['ds'] = feature_data['ds'].dt.tz_localize(None)
        except Exception:
            # Fallback: force naive datetimes
            feature_data['ds'] = feature_data['ds'].astype('datetime64[ns]')
        feature_data['y'] = feature_data['count']
        return feature_data[['ds', 'y']]
    
    def fit_prophet(self, ts_data: pd.DataFrame, feature: str):  # type: ignore
        """Fit Prophet model if available; return None otherwise."""
        if not HAS_PROPHET:
            return None
        try:
            # Down-sample very long histories to speed up Stan compile / fit
            if len(ts_data) > 4000:
                ts_data = ts_data.iloc[::2].reset_index(drop=True)
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            model.fit(ts_data)
            self.models[f"{feature}_prophet"] = model
            logger.info(f"Prophet model fitted for feature: {feature}")
            return model
        except Exception as e:
            logger.error(f"Failed to fit Prophet model for {feature}: {e}")
            return None
    
    def fit_arima(self, ts_data: pd.DataFrame, feature: str, order: Tuple[int, int, int] = (1, 1, 1)):
        """Fit ARIMA if statsmodels present; otherwise None."""
        if not HAS_STATSMODELS:
            return None
        try:
            model = ARIMA(ts_data['y'], order=order)
            fitted_model = model.fit()
            self.models[f"{feature}_arima"] = fitted_model
            logger.info(f"ARIMA model fitted for feature: {feature}")
            return fitted_model
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model for {feature}: {e}")
            return None
    
    def fit_ets(self, ts_data: pd.DataFrame, feature: str):
        """Fit ETS model if statsmodels available."""
        if not HAS_STATSMODELS:
            return None
        try:
            model = ETSModel(ts_data['y'], trend='add', seasonal=None)
            fitted_model = model.fit()
            self.models[f"{feature}_ets"] = fitted_model
            logger.info(f"ETS model fitted for feature: {feature}")
            return fitted_model
        except Exception as e:
            logger.error(f"Failed to fit ETS model for {feature}: {e}")
            return None
    
    def forecast_feature(self, df: pd.DataFrame, feature: str, periods: int = 24) -> Dict[str, ForecastResult]:
        """Generate forecasts for a specific feature using multiple models."""
        ts_data = self.prepare_ts_data(df, feature)
        
        if len(ts_data) < 10:  # Need sufficient data
            logger.warning(f"Insufficient data for feature: {feature}")
            return {}
        
        results = {}
        
        # Prophet forecast
        prophet_model = self.fit_prophet(ts_data, feature)
        if prophet_model is not None:
            future = prophet_model.make_future_dataframe(periods=periods, freq='6H')
            forecast = prophet_model.predict(future)
            
            # Calculate metrics on training data
            train_pred = forecast[:-periods]['yhat']
            train_actual = ts_data['y']
            min_len = min(len(train_pred), len(train_actual))
            
            metrics = ModelMetrics(
                model_name="Prophet",
                mae=mean_absolute_error(train_actual[:min_len], train_pred[:min_len]),
                rmse=np.sqrt(mean_squared_error(train_actual[:min_len], train_pred[:min_len])),
                r2=r2_score(train_actual[:min_len], train_pred[:min_len])
            )
            
            results['prophet'] = ForecastResult(
                feature=feature,
                predictions=forecast[['ds', 'yhat']].rename(columns={'yhat': 'prediction'}),
                model_metrics=metrics,
                forecast_horizon=periods,
                confidence_intervals=forecast[['ds', 'yhat_lower', 'yhat_upper']]
            )
        
        # ARIMA forecast
        arima_model = self.fit_arima(ts_data, feature)
        if arima_model is not None:
            forecast = arima_model.forecast(steps=periods)
            forecast_index = pd.date_range(
                start=ts_data['ds'].max() + pd.Timedelta(hours=6),
                periods=periods,
                freq='6H'
            )
            
            arima_forecast = pd.DataFrame({
                'ds': forecast_index,
                'prediction': forecast
            })
            
            # Calculate metrics
            residuals = arima_model.resid
            metrics = ModelMetrics(
                model_name="ARIMA",
                mae=np.mean(np.abs(residuals)),
                rmse=np.sqrt(np.mean(residuals**2))
            )
            
            results['arima'] = ForecastResult(
                feature=feature,
                predictions=arima_forecast,
                model_metrics=metrics,
                forecast_horizon=periods
            )
        
        return results


class TrendCategoryClassifier:
    """Classification model for trend categorization."""
    
    def __init__(self):
        # Slightly more robust configurations + class balancing
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def prepare_classification_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for classification."""
        # Feature engineering for classification
        feature_df = df.copy()
        
        # Ensure time ordering per feature for lag calculations
        if 'bin' in feature_df.columns:
            feature_df['bin'] = pd.to_datetime(feature_df['bin'])
            feature_df = feature_df.sort_values(['feature', 'bin'])

        # Group-wise temporal features
        def _add_group_features(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values('bin')
            g['lag1'] = g['count'].shift(1)
            g['lag2'] = g['count'].shift(2)
            g['count_diff1'] = g['count'].diff()
            # 24h window == 4 bins (6h)
            g['rolling_std_24h'] = g['count'].rolling(window=4, min_periods=2).std()
            if 'rolling_mean_24h' in g.columns:
                g['count_over_mean'] = g['count'] / (g['rolling_mean_24h'] + 1e-6)
            else:
                g['count_over_mean'] = 1.0
            return g

        if 'feature' in feature_df.columns and 'bin' in feature_df.columns:
            feature_df = feature_df.groupby('feature', group_keys=False).apply(_add_group_features)
        else:
            # Fallback if structure unexpected
            feature_df['lag1'] = feature_df['count'].shift(1)
            feature_df['lag2'] = feature_df['count'].shift(2)
            feature_df['count_diff1'] = feature_df['count'].diff()
            feature_df['rolling_std_24h'] = feature_df['count'].rolling(window=4, min_periods=2).std()
            feature_df['count_over_mean'] = 1.0

        # Calculate additional features
        feature_df['count_log'] = np.log1p(feature_df['count'])
        feature_df['velocity_abs'] = np.abs(feature_df.get('velocity', 0))
        feature_df['growth_rate_log'] = np.log1p(feature_df.get('growth_rate', 1))
        feature_df['delta_vs_mean_abs'] = np.abs(feature_df['delta_vs_mean'])
        
        # Time-based features
        feature_df['hour'] = pd.to_datetime(feature_df['bin']).dt.hour
        feature_df['day_of_week'] = pd.to_datetime(feature_df['bin']).dt.dayofweek
        
        # Select features for classification
        feature_cols = [
            'count_log', 'rolling_mean_24h', 'delta_vs_mean_abs',
            'velocity_abs', 'growth_rate_log', 'hour', 'day_of_week',
            'lag1', 'lag2', 'count_diff1', 'rolling_std_24h', 'count_over_mean'
        ]
        
        available_cols = [col for col in feature_cols if col in feature_df.columns]
        X = feature_df[available_cols].fillna(0).values
        y = feature_df['category'].values
        
        return X, y
    
    def fit(self, df: pd.DataFrame):
        """Fit classification models."""
        X, y = self.prepare_classification_features(df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Fit models
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        
        # Evaluate models
        rf_score = self.rf_model.score(X_test, y_test)
        gb_score = self.gb_model.score(X_test, y_test)
        
        logger.info(f"Random Forest accuracy: {rf_score:.3f}")
        logger.info(f"Gradient Boosting accuracy: {gb_score:.3f}")
        
        self.is_fitted = True
    
    def predict_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict trend categories."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _ = self.prepare_classification_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        
        # Get probabilities
        rf_proba = self.rf_model.predict_proba(X_scaled)
        gb_proba = self.gb_model.predict_proba(X_scaled)
        
        # Decode predictions
        rf_categories = self.label_encoder.inverse_transform(rf_pred)
        gb_categories = self.label_encoder.inverse_transform(gb_pred)

        # Simple ensemble (majority vote with probability tie-breaker)
        ensemble_pred = []
        for i in range(len(rf_pred)):
            if rf_pred[i] == gb_pred[i]:
                ensemble_pred.append(rf_pred[i])
            else:
                # pick higher max probability model
                if rf_proba[i].max() >= gb_proba[i].max():
                    ensemble_pred.append(rf_pred[i])
                else:
                    ensemble_pred.append(gb_pred[i])
        ensemble_categories = self.label_encoder.inverse_transform(np.array(ensemble_pred))
        
        results = df.copy()
        results['rf_predicted_category'] = rf_categories
        results['gb_predicted_category'] = gb_categories
        results['rf_confidence'] = np.max(rf_proba, axis=1)
        results['gb_confidence'] = np.max(gb_proba, axis=1)
        results['ensemble_predicted_category'] = ensemble_categories
        # Ensemble confidence as max of chosen model's probability
        chosen_conf = []
        for i in range(len(ensemble_pred)):
            if ensemble_pred[i] == rf_pred[i]:
                chosen_conf.append(rf_proba[i].max())
            else:
                chosen_conf.append(gb_proba[i].max())
        results['ensemble_confidence'] = chosen_conf
        
        return results


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    @staticmethod
    def evaluate_classification_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                                    model_name: str) -> ModelMetrics:
        """Evaluate classification model performance."""
        y_pred = model.predict(X_test)
        
        metrics = ModelMetrics(
            model_name=model_name,
            f1_score=f1_score(y_test, y_pred, average='weighted'),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            accuracy=model.score(X_test, y_test)
        )
        
        return metrics
    
    @staticmethod
    def evaluate_time_series_model(actual: np.ndarray, predicted: np.ndarray, 
                                 model_name: str) -> ModelMetrics:
        """Evaluate time series model performance."""
        min_len = min(len(actual), len(predicted))
        actual_trim = actual[:min_len]
        predicted_trim = predicted[:min_len]
        
        metrics = ModelMetrics(
            model_name=model_name,
            mae=mean_absolute_error(actual_trim, predicted_trim),
            rmse=np.sqrt(mean_squared_error(actual_trim, predicted_trim)),
            r2=r2_score(actual_trim, predicted_trim)
        )
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
        """Perform cross-validation on model."""
        scores = cross_val_score(model, X, y, cv=cv)
        return scores.mean(), scores.std()


class ModelPersistence:
    """Model saving and loading utilities."""
    
    @staticmethod
    def save_model(model: Any, filepath: Path, metadata: Dict = None):
        """Save model with metadata."""
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: Path) -> Tuple[Any, Dict]:
        """Load model with metadata."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model'], model_data.get('metadata', {})
    
    @staticmethod
    def save_metrics(metrics: List[ModelMetrics], filepath: Path):
        """Save model metrics to JSON."""
        metrics_data = [asdict(metric) for metric in metrics]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")


def load_processed_features() -> Dict[str, pd.DataFrame]:
    """Load processed feature datasets."""
    datasets: Dict[str, pd.DataFrame] = {}

    base_feature_roots = [
        "features_hashtags_", "features_keywords_", "features_emerging_terms_", "features_audio_"
    ]

    for freq in AGG_FREQUENCIES:
        for root_name in base_feature_roots:
            file = f"{root_name}{freq}.parquet"
            filepath = PROC_DIR / file
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    if df.empty:
                        continue
                    # Attach frequency column for downstream logic
                    df['frequency'] = freq
                    key = file.replace('.parquet', '')
                    datasets[key] = df
                    logger.info(f"Loaded {file}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")

    if not datasets:
        logger.warning("No processed feature parquet files found across supported frequencies.")
    else:
        logger.info(f"Loaded feature datasets for frequencies: {sorted({df['frequency'].iloc[0] for df in datasets.values()})}")
    return datasets


def _compute_stl_period(freq: str) -> int:
    """Heuristic period selection per aggregation frequency for STL.

    For hourly aggregations we approximate daily seasonality.
    For daily / multi-day we approximate weekly or monthly cycles.
    """
    mapping = {
        '1h': 24,   # 24 hours ~ daily
        '3h': 8,    # 8 * 3h = 24h
        '6h': 4,    # 4 * 6h = 24h
        '1d': 7,    # weekly
        '3d': 7,    # still try weekly-ish (approx)
        '7d': 8,    # ~ two months of weekly points (arbitrary > seasonal req)
        '14d': 6,   # ~ quarter-year segmentation
        '1m': 12,   # yearly seasonality in months
        '3m': 8,    # multi-quarter pattern
        '6m': 4     # yearly split into halves
    }
    return mapping.get(freq, 7)


def run_comprehensive_modeling(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Run modeling pipeline across all available aggregation frequencies.

    For each frequency we run anomaly detection. Forecasting & classification
    are performed only on a chosen base frequency (prefers 6h, then 1h, else first available).
    """
    results: Dict[str, Any] = {
        'trend_detection': {},   # per-frequency anomalies
        'forecasting': {},       # base frequency forecasts
        'classification': {},    # base frequency classification
        'metrics': [],
        'base_frequency': None
    }

    # Organize datasets by frequency (expect column 'frequency')
    freq_groups: Dict[str, List[pd.DataFrame]] = {}
    for name, df in datasets.items():
        if 'frequency' not in df.columns:
            continue
        freq = df['frequency'].iloc[0]
        freq_groups.setdefault(freq, []).append(df)

    if not freq_groups:
        logger.warning("No datasets with 'frequency' column available for modeling.")
        return results

    # Determine base frequency for forecasting & classification
    preferred_order = ['6h', '1h', '3h', '1d']
    available_freqs = set(freq_groups.keys())
    base_freq = next((f for f in preferred_order if f in available_freqs), None)
    if base_freq is None:
        base_freq = sorted(available_freqs)[0]
    results['base_frequency'] = base_freq
    logger.info(f"Selected base frequency for forecasting/classification: {base_freq}")

    all_anomaly_frames = []

    for freq, frames in freq_groups.items():
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty or 'category' not in combined.columns:
            logger.warning(f"Skipping frequency {freq}: empty or missing category column")
            continue
        logger.info(f"Processing frequency {freq}: {len(combined)} rows across {len(frames)} tables")

        period = _compute_stl_period(freq)
        trend_detector = TrendDetectionModel(contamination=0.1, period=period)
        trend_detector.fit(combined)
        anomalies_df = trend_detector.predict_anomalies(combined)
        if not anomalies_df.empty:
            anomalies_df['frequency'] = freq
            all_anomaly_frames.append(anomalies_df)
        results['trend_detection'][freq] = anomalies_df

        # Persist detector per frequency (lightweight)
        ModelPersistence.save_model(
            trend_detector,
            MODELS_DIR / f'trend_detection_model_{freq}.pkl',
            {'type': 'trend_detection', 'contamination': 0.1, 'frequency': freq}
        )

        # Only run forecasting/classification for base frequency
        if freq == base_freq:
            # Forecasting
            logger.info(f"Training forecasting models for base frequency {base_freq}...")
            forecaster = TimeSeriesForecaster()
            top_features = (combined.groupby('feature')['count']
                            .sum()
                            .sort_values(ascending=False)
                            .head(10)
                            .index.tolist())
            for feature in top_features:
                feature_forecasts = forecaster.forecast_feature(combined, feature, periods=24)
                if feature_forecasts:
                    results['forecasting'][feature] = feature_forecasts
            ModelPersistence.save_model(
                forecaster,
                MODELS_DIR / f'forecasting_models_{base_freq}.pkl',
                {'type': 'time_series_forecasting', 'top_features': top_features, 'frequency': base_freq}
            )

            # Classification
            logger.info(f"Training classification models for base frequency {base_freq}...")
            classifier = TrendCategoryClassifier()
            category_counts = combined['category'].value_counts()
            valid_categories = category_counts[category_counts >= 50].index
            classification_df = combined[combined['category'].isin(valid_categories)]
            if len(classification_df) > 100:
                classifier.fit(classification_df)
                predictions = classifier.predict_categories(classification_df)
                results['classification']['predictions'] = predictions
                ModelPersistence.save_model(
                    classifier,
                    MODELS_DIR / f'classification_model_{base_freq}.pkl',
                    {'type': 'category_classification', 'valid_categories': valid_categories.tolist(), 'frequency': base_freq}
                )

    # Concatenate anomalies across frequencies for backward compatibility key
    if all_anomaly_frames:
        results['trend_detection']['anomalies'] = pd.concat(all_anomaly_frames, ignore_index=True)
    else:
        results['trend_detection']['anomalies'] = pd.DataFrame()

    return results


def generate_model_report(results: Dict[str, Any]):
    """Generate comprehensive model performance report."""
    report_path = INTERIM_DIR / 'phase3_model_performance_report.md'
    
    lines = ["# Phase 3: Model Performance Report", ""]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Trend Detection Results
    lines.append("## Trend Detection Models")
    anomalies_all = results['trend_detection'].get('anomalies', pd.DataFrame())
    if not anomalies_all.empty:
        lines.append(f"- Total anomalies detected (all frequencies): {len(anomalies_all)}")
        freqs_present = sorted(anomalies_all['frequency'].unique()) if 'frequency' in anomalies_all.columns else []
        if freqs_present:
            lines.append(f"- Frequencies covered: {', '.join(freqs_present)}")
        lines.append("- Models: Per-frequency STL (dynamic period) + Cross-Platform & Semantic Validation")
        # Show per-frequency counts
        if 'frequency' in anomalies_all.columns:
            freq_counts = anomalies_all.groupby('frequency').size().sort_values(ascending=False)
            lines.append("\n### Anomalies by Frequency")
            lines.append(freq_counts.to_frame('count').to_markdown())
        # Top anomalies global
        top_anomalies = anomalies_all.nlargest(10, 'score')
        lines.append("\n### Top Anomalies (Global)")
        try:
            cols = ['feature', 'timestamp', 'frequency', 'score', 'anomaly_type', 'platform', 'semantic_cluster']
            cols = [c for c in cols if c in top_anomalies.columns]
            lines.append(top_anomalies[cols].to_markdown(index=False))
        except Exception as e:
            lines.append(f"Could not render anomalies table: {e}")
    else:
        lines.append("No anomalies detected")
    
    lines.append("")
    
    # Forecasting Results
    lines.append("## Time Series Forecasting Models")
    forecasting_results = results['forecasting']
    if forecasting_results:
        base_freq = results.get('base_frequency')
        lines.append(f"- Base frequency: {base_freq}")
        lines.append(f"- Features forecasted: {len(forecasting_results)}")
        lines.append("- Models used: Prophet, ARIMA (ETS reserved)")
        
        lines.append("\n### Forecasting Performance")
        for feature, models in forecasting_results.items():
            lines.append(f"\n#### {feature}")
            for model_name, forecast_result in models.items():
                metrics = forecast_result.model_metrics
                mae_str = f"{metrics.mae:.3f}" if metrics.mae is not None else "N/A"
                rmse_str = f"{metrics.rmse:.3f}" if metrics.rmse is not None else "N/A"
                r2_str = f"{metrics.r2:.3f}" if metrics.r2 is not None else "N/A"
                lines.append(f"**{model_name.upper()}**: MAE={mae_str}, RMSE={rmse_str}, R²={r2_str}")
    else:
        lines.append("No forecasting results available")
    
    lines.append("")
    
    # Classification Results  
    lines.append("## Category Classification Models")
    classification_results = results['classification']
    if classification_results:
        lines.append("- Models used: Random Forest + Gradient Boosting")
        predictions = classification_results.get('predictions', pd.DataFrame())
        if not predictions.empty:
            lines.append(f"- Predictions generated: {len(predictions)}")
            
            # Accuracy comparison
            rf_accuracy = (predictions['category'] == predictions['rf_predicted_category']).mean()
            gb_accuracy = (predictions['category'] == predictions['gb_predicted_category']).mean()
            
            lines.append(f"\n### Model Accuracy")
            lines.append(f"- Random Forest: {rf_accuracy:.3f}")
            lines.append(f"- Gradient Boosting: {gb_accuracy:.3f}")
    else:
        lines.append("No classification results available")
    
    lines.append("")
    
    # Model Files
    lines.append("## Saved Models")
    lines.append("The following models have been saved to the `models/` directory:")
    model_files = list(MODELS_DIR.glob('*.pkl'))
    for model_file in model_files:
        lines.append(f"- `{model_file.name}`")
    
    lines.append("")
    lines.append("## Next Steps")
    lines.append("1. Validate model performance on held-out test data")
    lines.append("2. Implement real-time prediction pipeline")
    lines.append("3. Set up model monitoring and drift detection")
    lines.append("4. Integrate models with Streamlit dashboard")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    logger.info(f"Model report generated: {report_path}")


def main():
    """Main Phase 3 modeling pipeline."""
    logger.info("Starting Phase 3: Model Building...")
    
    # Load processed features
    datasets = load_processed_features()
    
    if not datasets:
        logger.error("No processed feature datasets found. Run Phase 2 first.")
        return
    
    # Run comprehensive modeling
    results = run_comprehensive_modeling(datasets)
    
    # Generate performance report
    generate_model_report(results)
    
    # Save results summary
    summary_path = MODELS_DIR / 'modeling_results_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'datasets_processed': list(datasets.keys()),
        'anomalies_detected': len(results['trend_detection'].get('anomalies', [])),
        'features_forecasted': len(results['forecasting']),
        'classification_accuracy': 'computed' if results['classification'] else 'not_available'
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Phase 3 modeling completed successfully!")
    logger.info(f"Summary: {summary}")


if __name__ == '__main__':
    main()
