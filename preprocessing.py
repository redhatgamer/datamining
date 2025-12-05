"""
Data Preprocessing Module
Handles missing values, outlier detection, and feature normalization
"""
import pandas as pd
import numpy as np
from scipy import stats

class DataPreprocessor:
    def __init__(self, df):
        """Initialize preprocessor with dataframe"""
        self.df = df.copy()
        self.preprocessed_df = None
        self.scaler_params = {}
        
    def analyze_missing_values(self):
        """Analyze and report missing values (optimized)"""
        print("\n" + "="*60)
        print("MISSING VALUE ANALYSIS")
        print("="*60)
        
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        # Filter only columns with missing values
        has_missing = missing > 0
        
        if not has_missing.any():
            print("No missing values found!")
        else:
            print(f"{'Column':<20} {'Missing Count':<15} {'Missing %':<10}")
            print("-" * 45)
            for col in self.df.columns[has_missing]:
                print(f"{col:<20} {missing[col]:<15} {missing_percent[col]:<10.2f}")
        
        return missing[has_missing]
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        
        # Handle categorical columns (product_name) - drop if missing
        if 'product_name' in self.df.columns:
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=['product_name'])
            dropped = initial_rows - len(self.df)
            if dropped > 0:
                print(f"Dropped {dropped} rows with missing product_name")
        
        # Handle numerical columns - impute with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"Imputed {col} with median: {median_val:.2f}")
        
        print(f"Final dataset shape: {self.df.shape}")
        return self.df
    
    def detect_outliers_iqr(self, column, iqr_multiplier=1.5):
        """Detect outliers using IQR method"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (iqr_multiplier * IQR)
        upper_bound = Q3 + (iqr_multiplier * IQR)
        
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, column, threshold=3):
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(self.df[column]))
        outliers = z_scores > threshold
        return outliers
    
    def handle_outliers(self, method='iqr', cap=True):
        """Detect and handle outliers"""
        print("\n" + "="*60)
        print("OUTLIER DETECTION AND TREATMENT")
        print("="*60)
        print(f"Method: {method.upper()}")
        print(f"Treatment: {'Cap at bounds' if cap else 'Remove records'}")
        
        numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'profit']
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]
        
        for col in numerical_cols:
            if method == 'iqr':
                outliers, lower, upper = self.detect_outliers_iqr(col)
            else:  # zscore
                outliers, lower, upper = self.detect_outliers_zscore(col), None, None
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"\n{col}: {outlier_count} outliers detected")
                
                if cap:
                    # Cap outliers at bounds
                    if method == 'iqr':
                        self.df[col] = self.df[col].clip(lower, upper)
                        print(f"  → Capped to range [{lower:.2f}, {upper:.2f}]")
                else:
                    # Remove outliers
                    self.df = self.df[~outliers]
                    print(f"  → Removed {outlier_count} records")
        
        print(f"\nFinal dataset shape: {self.df.shape}")
        return self.df
    
    def normalize_features(self, method='minmax'):
        """Normalize/standardize numerical features"""
        print("\n" + "="*60)
        print("FEATURE NORMALIZATION/STANDARDIZATION")
        print("="*60)
        print(f"Method: {method.upper()}")
        
        if method == 'minmax':
            print("\nMin-Max Normalization (scaling to 0-1):")
            print("Formula: (x - min) / (max - min)")
            print("Reason: Scales all features to same range, important for distance-based algorithms like K-means")
        else:
            print("\nZ-score Standardization (scaling to mean=0, std=1):")
            print("Formula: (x - mean) / std")
            print("Reason: Centers data around zero, useful when features have different distributions")
        
        self.preprocessed_df = self.df.copy()
        
        numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'profit']
        numerical_cols = [col for col in numerical_cols if col in self.preprocessed_df.columns]
        
        for col in numerical_cols:
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                self.preprocessed_df[col] = (self.df[col] - min_val) / (max_val - min_val)
                self.scaler_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                print(f"  {col}: min={min_val:.2f}, max={max_val:.2f}")
            else:  # zscore
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                self.preprocessed_df[col] = (self.df[col] - mean_val) / std_val
                self.scaler_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
                print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        return self.preprocessed_df
    
    def get_preprocessing_summary(self):
        """Generate preprocessing summary statistics"""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        
        print(f"\nDataset Shape: {self.preprocessed_df.shape}")
        print(f"Total Records: {len(self.preprocessed_df)}")
        print(f"Total Features: {len(self.preprocessed_df.columns)}")
        
        print("\nNumerical Features Statistics:")
        numerical_cols = self.preprocessed_df.select_dtypes(include=[np.number]).columns
        print(self.preprocessed_df[numerical_cols].describe().round(3))
        
        return self.preprocessed_df.describe()
    
    def preprocess(self, handle_outliers=True, outlier_method='iqr', 
                   normalize_method='minmax'):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        self.analyze_missing_values()
        self.handle_missing_values()
        
        if handle_outliers:
            self.handle_outliers(method=outlier_method, cap=True)
        
        self.normalize_features(method=normalize_method)
        self.get_preprocessing_summary()
        
        return self.preprocessed_df
