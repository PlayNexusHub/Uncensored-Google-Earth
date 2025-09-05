"""
Machine Learning Integration Module for PlayNexus Satellite Toolkit
Provides automated feature detection, classification, and predictive modeling capabilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import logging
from datetime import datetime
import joblib
import pickle

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from .error_handling import PlayNexusLogger, ValidationError, ProcessingError
from .config import ConfigManager
from .progress_tracker import ProgressTracker, track_progress

logger = PlayNexusLogger(__name__)

class SatelliteMLPipeline:
    """Machine learning pipeline for satellite imagery analysis."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the ML pipeline."""
        self.config = config or ConfigManager()
        self.logger = PlayNexusLogger(__name__)
        self.progress_tracker = ProgressTracker()
        self._setup_ml_config()
        self._setup_models()
    
    def _setup_ml_config(self):
        """Setup machine learning configuration."""
        self.default_random_state = 42
        self.default_test_size = 0.2
        self.default_cv_folds = 5
        self.model_save_dir = Path.home() / '.playnexus' / 'models'
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_models(self):
        """Setup available machine learning models."""
        self.classification_models = {
            'random_forest': RandomForestClassifier(random_state=self.default_random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.default_random_state),
            'svm': SVC(random_state=self.default_random_state, probability=True),
            'logistic_regression': LogisticRegression(random_state=self.default_random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.default_random_state),
            'mlp': MLPClassifier(random_state=self.default_random_state, max_iter=1000)
        }
        
        if xgb:
            self.classification_models['xgboost'] = xgb.XGBClassifier(random_state=self.default_random_state)
        
        if lgb:
            self.classification_models['lightgbm'] = lgb.LGBMClassifier(random_state=self.default_random_state)
        
        self.clustering_models = {
            'kmeans': KMeans(random_state=self.default_random_state),
            'dbscan': DBSCAN(),
            'agglomerative': AgglomerativeClustering()
        }
        
        self.anomaly_detection_models = {
            'isolation_forest': IsolationForest(random_state=self.default_random_state),
            'one_class_svm': OneClassSVM()
        }
        
        self.dimensionality_reduction_models = {
            'pca': PCA(),
            'nmf': NMF(random_state=self.default_random_state),
            'fast_ica': FastICA(random_state=self.default_random_state)
        }
    
    @track_progress("Feature extraction")
    def extract_features(
        self,
        image_data: np.ndarray,
        feature_types: List[str] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Extract various features from satellite imagery."""
        if feature_types is None:
            feature_types = ['statistical', 'textural', 'spectral', 'morphological']
        
        features = {}
        
        for feature_type in feature_types:
            if feature_type == 'statistical':
                features.update(self._extract_statistical_features(image_data))
            elif feature_type == 'textural':
                features.update(self._extract_textural_features(image_data, **kwargs))
            elif feature_type == 'spectral':
                features.update(self._extract_spectral_features(image_data))
            elif feature_type == 'morphological':
                features.update(self._extract_morphological_features(image_data, **kwargs))
            else:
                self.logger.warning(f"Unknown feature type: {feature_type}")
        
        return features
    
    def _extract_statistical_features(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract statistical features from image data."""
        features = {}
        
        # Basic statistics
        features['mean'] = np.nanmean(image_data, axis=0)
        features['std'] = np.nanstd(image_data, axis=0)
        features['min'] = np.nanmin(image_data, axis=0)
        features['max'] = np.nanmax(image_data, axis=0)
        features['median'] = np.nanmedian(image_data, axis=0)
        features['range'] = features['max'] - features['min']
        features['variance'] = np.nanvar(image_data, axis=0)
        features['skewness'] = self._calculate_skewness(image_data)
        features['kurtosis'] = self._calculate_kurtosis(image_data)
        
        # Percentiles
        features['p25'] = np.nanpercentile(image_data, 25, axis=0)
        features['p75'] = np.nanpercentile(image_data, 75, axis=0)
        features['iqr'] = features['p75'] - features['p25']
        
        return features
    
    def _extract_textural_features(self, image_data: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
        """Extract textural features using GLCM-like approach."""
        features = {}
        
        # Local variance
        features['local_variance'] = self._local_variance(image_data, window_size)
        
        # Local entropy
        features['local_entropy'] = self._local_entropy(image_data, window_size)
        
        # Local contrast
        features['local_contrast'] = self._local_contrast(image_data, window_size)
        
        # Local homogeneity
        features['local_homogeneity'] = self._local_homogeneity(image_data, window_size)
        
        return features
    
    def _extract_spectral_features(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features from multi-band imagery."""
        features = {}
        
        if image_data.ndim == 3:
            # Multi-band imagery
            n_bands = image_data.shape[0]
            
            # Band ratios
            for i in range(n_bands):
                for j in range(i + 1, n_bands):
                    ratio_name = f'band_ratio_{i}_{j}'
                    features[ratio_name] = np.divide(
                        image_data[i], 
                        image_data[j] + 1e-8
                    )
            
            # Normalized differences
            for i in range(n_bands):
                for j in range(i + 1, n_bands):
                    nd_name = f'nd_{i}_{j}'
                    features[nd_name] = (image_data[i] - image_data[j]) / (image_data[i] + image_data[j] + 1e-8)
            
            # Principal components (simplified)
            try:
                # Reshape for PCA
                reshaped_data = image_data.reshape(n_bands, -1).T
                valid_mask = ~np.any(np.isnan(reshaped_data), axis=1)
                
                if np.sum(valid_mask) > 0:
                    valid_data = reshaped_data[valid_mask]
                    pca = PCA(n_components=min(3, n_bands))
                    pca_result = pca.fit_transform(valid_data)
                    
                    # Map back to original shape
                    for comp in range(pca_result.shape[1]):
                        comp_name = f'pca_component_{comp}'
                        features[comp_name] = np.full(image_data.shape[1:], np.nan)
                        features[comp_name].flat[valid_mask] = pca_result[:, comp]
            except Exception as e:
                self.logger.warning(f"PCA failed: {e}")
        
        return features
    
    def _extract_morphological_features(self, image_data: np.ndarray, kernel_size: int = 3) -> Dict[str, np.ndarray]:
        """Extract morphological features from imagery."""
        features = {}
        
        try:
            from scipy import ndimage
            
            # Morphological operations
            features['erosion'] = ndimage.binary_erosion(image_data > np.nanmean(image_data), 
                                                       structure=np.ones((kernel_size, kernel_size)))
            features['dilation'] = ndimage.binary_dilation(image_data > np.nanmean(image_data), 
                                                         structure=np.ones((kernel_size, kernel_size)))
            features['opening'] = ndimage.binary_opening(image_data > np.nanmean(image_data), 
                                                       structure=np.ones((kernel_size, kernel_size)))
            features['closing'] = ndimage.binary_closing(image_data > np.nanmean(image_data), 
                                                       structure=np.ones((kernel_size, kernel_size)))
            
            # Distance transform
            features['distance_transform'] = ndimage.distance_transform_edt(image_data > np.nanmean(image_data))
            
        except ImportError:
            self.logger.warning("SciPy not available for morphological features")
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each pixel across time/bands."""
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        
        # Avoid division by zero
        std_safe = np.where(std > 1e-8, std, 1)
        
        skewness = np.nanmean(((data - mean[:, None, :]) / std_safe[:, None, :]) ** 3, axis=0)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each pixel across time/bands."""
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        
        # Avoid division by zero
        std_safe = np.where(std > 1e-8, std, 1)
        
        kurtosis = np.nanmean(((data - mean[:, None, :]) / std_safe[:, None, :]) ** 4, axis=0) - 3
        return kurtosis
    
    def _local_variance(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local variance using sliding window."""
        from scipy import ndimage
        
        # Use uniform filter to calculate local mean
        local_mean = ndimage.uniform_filter(data, size=window_size)
        local_mean_sq = ndimage.uniform_filter(data ** 2, size=window_size)
        
        # Local variance = E[X²] - (E[X])²
        local_variance = local_mean_sq - local_mean ** 2
        
        return local_variance
    
    def _local_entropy(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local entropy using sliding window."""
        from scipy import ndimage
        
        # Discretize data into bins for entropy calculation
        n_bins = 10
        data_binned = np.digitize(data, bins=np.linspace(np.nanmin(data), np.nanmax(data), n_bins))
        
        # Calculate local histogram
        local_entropy = np.zeros_like(data, dtype=float)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Extract local window
                i_start = max(0, i - window_size // 2)
                i_end = min(data.shape[0], i + window_size // 2 + 1)
                j_start = max(0, j - window_size // 2)
                j_end = min(data.shape[1], j + window_size // 2 + 1)
                
                local_window = data_binned[i_start:i_end, j_start:j_end]
                
                # Calculate entropy
                hist, _ = np.histogram(local_window, bins=range(n_bins + 1))
                hist = hist[hist > 0]  # Remove zero counts
                if len(hist) > 0:
                    p = hist / np.sum(hist)
                    local_entropy[i, j] = -np.sum(p * np.log2(p))
        
        return local_entropy
    
    def _local_contrast(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local contrast using sliding window."""
        from scipy import ndimage
        
        local_max = ndimage.maximum_filter(data, size=window_size)
        local_min = ndimage.minimum_filter(data, size=window_size)
        
        return local_max - local_min
    
    def _local_homogeneity(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local homogeneity using sliding window."""
        from scipy import ndimage
        
        # Local homogeneity is inversely related to local variance
        local_var = self._local_variance(data, window_size)
        
        # Normalize and invert
        max_var = np.nanmax(local_var)
        if max_var > 0:
            local_homogeneity = 1 - (local_var / max_var)
        else:
            local_homogeneity = np.ones_like(local_var)
        
        return local_homogeneity
    
    @track_progress("Model training")
    def train_classification_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_type: str = 'random_forest',
        test_size: float = None,
        random_state: int = None,
        hyperparameter_tuning: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a classification model on extracted features."""
        test_size = test_size or self.default_test_size
        random_state = random_state or self.default_random_state
        
        # Validate inputs
        if model_type not in self.classification_models:
            raise ValidationError(f"Unsupported model type: {model_type}")
        
        if features.shape[0] != labels.shape[0]:
            raise ValidationError("Number of samples must match number of labels")
        
        # Prepare data
        features_flat = features.reshape(features.shape[0], -1)
        
        # Remove NaN values
        valid_mask = ~np.any(np.isnan(features_flat), axis=1)
        features_clean = features_flat[valid_mask]
        labels_clean = labels[valid_mask]
        
        if len(features_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_clean, labels_clean, test_size=test_size, random_state=random_state, stratify=labels_clean
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get model
        model = self.classification_models[model_type]
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model, X_train_scaled, y_train, model_type)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.default_cv_folds)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            metrics['feature_importance'] = model.feature_importances_.tolist()
        
        results = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'test_data': (X_test_scaled, y_test),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model_type': model_type,
            'training_date': datetime.now()
        }
        
        return results
    
    def _tune_hyperparameters(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str
    ) -> Any:
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        param_grid = param_grids.get(model_type, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=self.default_cv_folds, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            return model
    
    @track_progress("Clustering analysis")
    def perform_clustering(
        self,
        features: np.ndarray,
        method: str = 'kmeans',
        n_clusters: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform clustering analysis on extracted features."""
        if method not in self.clustering_models:
            raise ValidationError(f"Unsupported clustering method: {method}")
        
        # Prepare data
        features_flat = features.reshape(features.shape[0], -1)
        
        # Remove NaN values
        valid_mask = ~np.any(np.isnan(features_flat), axis=1)
        features_clean = features_flat[valid_mask]
        
        if len(features_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Get model
        model = self.clustering_models[method]
        
        # Set number of clusters for applicable methods
        if hasattr(model, 'n_clusters'):
            model.n_clusters = n_clusters
        
        # Perform clustering
        cluster_labels = model.fit_predict(features_scaled)
        
        # Calculate metrics
        metrics = {}
        if len(np.unique(cluster_labels)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(features_scaled, cluster_labels)
            except:
                metrics['silhouette_score'] = None
        
        # Map cluster labels back to original shape
        cluster_map = np.full(features.shape[1:], -1, dtype=int)
        cluster_map.flat[valid_mask] = cluster_labels
        
        results = {
            'model': model,
            'scaler': scaler,
            'cluster_labels': cluster_map,
            'metrics': metrics,
            'method': method,
            'n_clusters': n_clusters,
            'analysis_date': datetime.now()
        }
        
        return results
    
    @track_progress("Anomaly detection")
    def detect_anomalies_ml(
        self,
        features: np.ndarray,
        method: str = 'isolation_forest',
        contamination: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Detect anomalies using machine learning methods."""
        if method not in self.anomaly_detection_models:
            raise ValidationError(f"Unsupported anomaly detection method: {method}")
        
        # Prepare data
        features_flat = features.reshape(features.shape[0], -1)
        
        # Remove NaN values
        valid_mask = ~np.any(np.isnan(features_flat), axis=1)
        features_clean = features_flat[valid_mask]
        
        if len(features_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Get model
        model = self.anomaly_detection_models[method]
        
        # Set contamination if applicable
        if hasattr(model, 'contamination'):
            model.contamination = contamination
        
        # Fit model and predict
        if method == 'isolation_forest':
            # Isolation Forest returns -1 for anomalies, 1 for normal
            anomaly_scores = model.fit_predict(features_scaled)
            is_anomaly = (anomaly_scores == -1)
        else:
            # One-Class SVM returns -1 for anomalies, 1 for normal
            anomaly_scores = model.fit_predict(features_scaled)
            is_anomaly = (anomaly_scores == -1)
        
        # Calculate anomaly scores (distance from decision boundary)
        if hasattr(model, 'decision_function'):
            anomaly_scores = model.decision_function(features_scaled)
        else:
            anomaly_scores = np.zeros(len(features_clean))
        
        # Map results back to original shape
        anomaly_map = np.full(features.shape[1:], False, dtype=bool)
        anomaly_map.flat[valid_mask] = is_anomaly
        
        score_map = np.full(features.shape[1:], np.nan, dtype=float)
        score_map.flat[valid_mask] = anomaly_scores
        
        results = {
            'model': model,
            'scaler': scaler,
            'anomaly_map': anomaly_map,
            'anomaly_scores': score_map,
            'method': method,
            'contamination': contamination,
            'n_anomalies': np.sum(is_anomaly),
            'analysis_date': datetime.now()
        }
        
        return results
    
    def save_model(self, model_results: Dict[str, Any], model_name: str) -> Path:
        """Save trained model and results."""
        model_path = self.model_save_dir / f"{model_name}.joblib"
        
        # Save model and scaler
        model_data = {
            'model': model_results['model'],
            'scaler': model_results.get('scaler'),
            'model_type': model_results.get('model_type'),
            'training_date': model_results.get('training_date'),
            'metrics': model_results.get('metrics')
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a previously saved model."""
        model_path = self.model_save_dir / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        model_data = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        
        return model_data
    
    def predict_on_new_data(
        self,
        model_results: Dict[str, Any],
        new_features: np.ndarray
    ) -> np.ndarray:
        """Make predictions on new data using trained model."""
        model = model_results['model']
        scaler = model_results.get('scaler')
        
        # Prepare data
        features_flat = new_features.reshape(new_features.shape[0], -1)
        
        # Scale if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features_flat)
        else:
            features_scaled = features_flat
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(features_scaled)
        else:
            raise ValidationError("Model does not support prediction")
        
        # Reshape predictions to match input dimensions
        if new_features.ndim > 1:
            predictions = predictions.reshape(new_features.shape[1:])
        
        return predictions

# Deep Learning Integration (if available)
if TENSORFLOW_AVAILABLE:
    class SatelliteDeepLearning:
        """Deep learning capabilities for satellite imagery analysis."""
        
        def __init__(self):
            self.logger = PlayNexusLogger(__name__)
        
        def create_cnn_model(
            self,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            **kwargs
        ) -> tf.keras.Model:
            """Create a CNN model for satellite image classification."""
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model

# Convenience functions
def extract_satellite_features(
    image_data: np.ndarray,
    feature_types: List[str] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Convenience function for feature extraction."""
    pipeline = SatelliteMLPipeline()
    return pipeline.extract_features(image_data, feature_types, **kwargs)

def train_satellite_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    model_type: str = 'random_forest',
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for training classification models."""
    pipeline = SatelliteMLPipeline()
    return pipeline.train_classification_model(features, labels, model_type, **kwargs)

def cluster_satellite_data(
    features: np.ndarray,
    method: str = 'kmeans',
    n_clusters: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for clustering analysis."""
    pipeline = SatelliteMLPipeline()
    return pipeline.perform_clustering(features, method, n_clusters, **kwargs)

def detect_satellite_anomalies_ml(
    features: np.ndarray,
    method: str = 'isolation_forest',
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for ML-based anomaly detection."""
    pipeline = SatelliteMLPipeline()
    return pipeline.detect_anomalies_ml(features, method, **kwargs)
