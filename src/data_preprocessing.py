"""
Data Preprocessing Module for In-Vehicle Coupon Recommendation System

This module handles data loading, cleaning, and preprocessing for the coupon
recommendation system.

Author: Mekala Jaswanth
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CouponDataPreprocessor:
    """
    A class to preprocess the in-vehicle coupon recommendation dataset.
    
    Attributes:
        data (pd.DataFrame): The raw dataset
        processed_data (pd.DataFrame): The processed dataset
        label_encoders (dict): Dictionary of label encoders for each categorical column
        scaler (StandardScaler): Scaler for numerical features
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the CSV file containing the dataset
        """
        self.data = None
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load the dataset from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy to handle missing values ('drop', 'mode', 'median')
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        logger.info(f"Missing values before handling:\n{self.data.isnull().sum()}")
        
        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy == 'mode':
            for col in self.data.columns:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        elif strategy == 'median':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        
        logger.info(f"Missing values after handling:\n{self.data.isnull().sum()}")
        return self.data
    
    def encode_categorical_features(self, columns: list = None) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            columns (list): List of columns to encode. If None, all object columns are encoded.
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in self.data.columns and col != 'Y':  # Don't encode target variable
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded column: {col}")
        
        return self.data
    
    def create_target_variable(self, target_column: str = 'Y') -> tuple:
        """
        Separate features and target variable.
        
        Args:
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target_column] = le
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Features to scale
            
        Returns:
            pd.DataFrame: Scaled features
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        logger.info("Features scaled successfully")
        return X
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, data_path: str = None, 
                          missing_strategy: str = 'drop',
                          test_size: float = 0.2,
                          scale: bool = True) -> tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            data_path (str): Path to data file
            missing_strategy (str): Strategy for handling missing values
            test_size (float): Test set size
            scale (bool): Whether to scale features
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load data
        if data_path:
            self.load_data(data_path)
        
        # Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        
        # Encode categorical features
        self.encode_categorical_features()
        
        # Create target variable
        X, y = self.create_target_variable()
        
        # Scale features if requested
        if scale:
            X = self.scale_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        logger.info("Preprocessing pipeline completed successfully")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = CouponDataPreprocessor()
    # Uncomment the following line when you have the data file
    # X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline('data/in-vehicle-coupon-data.csv')
