"""
Machine Learning Models for In-Vehicle Coupon Recommendation System

This module implements various ML models for predicting coupon acceptance.

Author: Mekala Jaswanth
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CouponRecommendationModel:
    """
    A class to train and evaluate models for coupon recommendation.
    
    Attributes:
        model: The trained machine learning model
        model_type: Type of model ('logistic', 'random_forest', 'gradient_boosting')
        best_params: Best hyperparameters from grid search
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        logger.info("Training completed")
        return self
    
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")
        
        param_grids = {
            'logistic': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"\nModel Evaluation Results for {self.model_type}:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for score in scoring:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=score)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            logger.info(f"{score.upper()}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.model_type in ['random_forest', 'gradient_boosting']:
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            logger.warning("Feature importance only available for tree-based models")
            return None
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self


class ModelComparison:
    """
    Compare multiple models for coupon recommendation.
    """
    
    def __init__(self):
        """Initialize the model comparison."""
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model):
        """
        Add a model to compare.
        
        Args:
            name (str): Model name
            model: Model instance
        """
        self.models[name] = model
    
    def train_all(self, X_train, y_train):
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            model.train(X_train, y_train)
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            pd.DataFrame: Comparison results
        """
        for name, model in self.models.items():
            logger.info(f"\nEvaluating {name}...")
            self.results[name] = model.evaluate(X_test, y_test)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            }
            for name, results in self.results.items()
        }).T
        
        logger.info("\n=== Model Comparison Results ===")
        logger.info(f"\n{comparison_df}")
        
        return comparison_df
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            tuple: (model_name, model_instance)
        """
        best_name = max(self.results.items(), 
                       key=lambda x: x[1][metric])[0]
        return best_name, self.models[best_name]


if __name__ == "__main__":
    # Example usage
    logger.info("Model module loaded successfully")
    # Uncomment to test with actual data
    # model = CouponRecommendationModel('random_forest')
    # model.train(X_train, y_train)
    # metrics = model.evaluate(X_test, y_test)
