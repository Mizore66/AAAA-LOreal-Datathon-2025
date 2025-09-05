# Phase 3: Model Building Framework for Trend Detection and Forecasting

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import logging

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, mean_absolute_error, 
    mean_squared_error, r2_score, f1_score, precision_score, recall_score
)
from sklearn.svm import OneClassSVM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
import scipy.stats as stats

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
INTERIM_DIR = ROOT / "data" / "interim"

# Create directories
for dir_path in [MODELS_DIR, INTERIM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Data class for anomaly detection results."""
    feature: str
    timestamp: pd.Timestamp
    score: float
    platform: Optional[str] = None
    category: Optional[str] = None
    anomaly_type: Optional[str] = None


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


class TrendDetectionModel:
    """Advanced trend detection model with multiple algorithms."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.one_class_svm = OneClassSVM(nu=contamination)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        feature_cols = ['count', 'rolling_mean_24h', 'delta_vs_mean']
        if 'velocity' in df.columns:
            feature_cols.append('velocity')
        if 'growth_rate' in df.columns:
            feature_cols.append('growth_rate')
        
        features = df[feature_cols].fillna(0)
        return features.values
    
    def fit(self, df: pd.DataFrame):
        """Fit anomaly detection models."""
        features = self.prepare_features(df)
        features_scaled = self.scaler.fit_transform(features)
        
        self.isolation_forest.fit(features_scaled)
        self.one_class_svm.fit(features_scaled)
        self.is_fitted = True
        logger.info("Trend detection models fitted successfully")
    
    def predict_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from both models
        if_pred = self.isolation_forest.predict(features_scaled)
        if_scores = self.isolation_forest.score_samples(features_scaled)
        
        svm_pred = self.one_class_svm.predict(features_scaled)
        
        # Combine results
        results = df.copy()
        results['isolation_forest_anomaly'] = if_pred == -1
        results['isolation_forest_score'] = if_scores
        results['svm_anomaly'] = svm_pred == -1
        results['ensemble_anomaly'] = (if_pred == -1) | (svm_pred == -1)
        
        return results[results['ensemble_anomaly']]


class TimeSeriesForecaster:
    """Time series forecasting with multiple models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prepare_ts_data(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Prepare time series data for a specific feature."""
        feature_data = df[df['feature'] == feature].copy()
        feature_data = feature_data.sort_values('bin').reset_index(drop=True)
        feature_data['ds'] = pd.to_datetime(feature_data['bin'])
        feature_data['y'] = feature_data['count']
        return feature_data[['ds', 'y']]
    
    def fit_prophet(self, ts_data: pd.DataFrame, feature: str) -> Prophet:
        """Fit Prophet model for time series forecasting."""
        try:
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
    
    def fit_arima(self, ts_data: pd.DataFrame, feature: str, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMA:
        """Fit ARIMA model for time series forecasting."""
        try:
            model = ARIMA(ts_data['y'], order=order)
            fitted_model = model.fit()
            self.models[f"{feature}_arima"] = fitted_model
            logger.info(f"ARIMA model fitted for feature: {feature}")
            return fitted_model
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model for {feature}: {e}")
            return None
    
    def fit_ets(self, ts_data: pd.DataFrame, feature: str) -> ETSModel:
        """Fit ETS (Exponential Smoothing) model."""
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
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def prepare_classification_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for classification."""
        # Feature engineering for classification
        feature_df = df.copy()
        
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
            'velocity_abs', 'growth_rate_log', 'hour', 'day_of_week'
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
        
        results = df.copy()
        results['rf_predicted_category'] = rf_categories
        results['gb_predicted_category'] = gb_categories
        results['rf_confidence'] = np.max(rf_proba, axis=1)
        results['gb_confidence'] = np.max(gb_proba, axis=1)
        
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
    datasets = {}
    
    feature_files = [
        'features_hashtags_6h.parquet',
        'features_keywords_6h.parquet', 
        'features_emerging_terms_6h.parquet',
        'features_audio_6h.parquet'
    ]
    
    for file in feature_files:
        filepath = PROC_DIR / file
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                datasets[file.replace('.parquet', '')] = df
                logger.info(f"Loaded {file}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
    
    return datasets


def run_comprehensive_modeling(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Run comprehensive modeling pipeline."""
    results = {
        'trend_detection': {},
        'forecasting': {},
        'classification': {},
        'metrics': []
    }
    
    # Combine all feature datasets
    all_features = []
    for name, df in datasets.items():
        if not df.empty and 'category' in df.columns:
            df_copy = df.copy()
            df_copy['source'] = name
            all_features.append(df_copy)
    
    if not all_features:
        logger.warning("No feature datasets available for modeling")
        return results
    
    combined_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} rows")
    
    # 1. Trend Detection Models
    logger.info("Training trend detection models...")
    trend_detector = TrendDetectionModel(contamination=0.1)
    trend_detector.fit(combined_df)
    
    anomalies = trend_detector.predict_anomalies(combined_df)
    results['trend_detection']['anomalies'] = anomalies
    
    # Save trend detection model
    ModelPersistence.save_model(
        trend_detector, 
        MODELS_DIR / 'trend_detection_model.pkl',
        {'type': 'trend_detection', 'contamination': 0.1}
    )
    
    # 2. Time Series Forecasting
    logger.info("Training forecasting models...")
    forecaster = TimeSeriesForecaster()
    
    # Get top features by activity for forecasting
    top_features = (combined_df.groupby('feature')['count']
                   .sum()
                   .sort_values(ascending=False)
                   .head(10)
                   .index.tolist())
    
    for feature in top_features:
        feature_forecasts = forecaster.forecast_feature(combined_df, feature, periods=24)
        if feature_forecasts:
            results['forecasting'][feature] = feature_forecasts
    
    # Save forecasting models
    ModelPersistence.save_model(
        forecaster,
        MODELS_DIR / 'forecasting_models.pkl',
        {'type': 'time_series_forecasting', 'top_features': top_features}
    )
    
    # 3. Category Classification
    logger.info("Training classification models...")
    classifier = TrendCategoryClassifier()
    
    # Filter data with sufficient samples per category
    category_counts = combined_df['category'].value_counts()
    valid_categories = category_counts[category_counts >= 50].index
    classification_df = combined_df[combined_df['category'].isin(valid_categories)]
    
    if len(classification_df) > 100:
        classifier.fit(classification_df)
        
        # Get predictions
        predictions = classifier.predict_categories(classification_df)
        results['classification']['predictions'] = predictions
        
        # Save classification model
        ModelPersistence.save_model(
            classifier,
            MODELS_DIR / 'classification_model.pkl',
            {'type': 'category_classification', 'valid_categories': valid_categories.tolist()}
        )
    
    return results


def generate_model_report(results: Dict[str, Any]):
    """Generate comprehensive model performance report."""
    report_path = INTERIM_DIR / 'phase3_model_performance_report.md'
    
    lines = ["# Phase 3: Model Performance Report", ""]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Trend Detection Results
    lines.append("## Trend Detection Models")
    anomalies = results['trend_detection'].get('anomalies', pd.DataFrame())
    if not anomalies.empty:
        lines.append(f"- Anomalies detected: {len(anomalies)}")
        lines.append("- Models used: Isolation Forest + One-Class SVM ensemble")
        
        # Top anomalies
        top_anomalies = anomalies.nlargest(10, 'isolation_forest_score')
        lines.append("\n### Top Anomalies Detected")
        try:
            cols = ['feature', 'bin', 'count', 'isolation_forest_score', 'category']
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
        lines.append(f"- Features forecasted: {len(forecasting_results)}")
        lines.append("- Models used: Prophet, ARIMA, ETS")
        
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
