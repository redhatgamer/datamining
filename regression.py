"""
Regression Analysis Module
Implements Linear and Polynomial Regression for prediction
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionAnalyzer:
    """Compare multiple regression models"""
    
    def __init__(self, X, y, test_size=0.3, random_state=42):
        """
        Initialize regression analyzer
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray or pd.Series
            Target variable
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed
        """
        self.X = np.array(X) if isinstance(X, pd.DataFrame) else X
        self.y = np.array(y) if isinstance(y, pd.Series) else y
        self.test_size = test_size
        self.random_state = random_state
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        self.models = {}
        self.results = {}
        self.feature_names = None
    
    def train_linear_regression(self):
        """Train linear regression model"""
        print("\n" + "="*60)
        print("LINEAR REGRESSION MODEL")
        print("="*60)
        print("Training on data shape:", self.X_train.shape)
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        self.models['Linear'] = model
        self.results['Linear'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE:  {test_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE:  {test_mae:.4f}")
        print(f"Training R²:  {train_r2:.4f}")
        print(f"Testing R²:   {test_r2:.4f}")
        
        # Print coefficients
        if self.feature_names is not None:
            print("\nModel Coefficients:")
            for name, coef in zip(self.feature_names, model.coef_):
                print(f"  {name}: {coef:.6f}")
        print(f"Intercept: {model.intercept_:.6f}")
        
        return model
    
    def train_polynomial_regression(self, degree=2):
        """Train polynomial regression model"""
        print("\n" + "="*60)
        print(f"POLYNOMIAL REGRESSION MODEL (Degree={degree})")
        print("="*60)
        print("Training on data shape:", self.X_train.shape)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)
        
        print(f"Original features: {self.X_train.shape[1]}")
        print(f"Polynomial features: {X_train_poly.shape[1]}")
        
        model = LinearRegression()
        model.fit(X_train_poly, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        self.models[f'Polynomial_{degree}'] = (model, poly)
        self.results[f'Polynomial_{degree}'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE:  {test_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE:  {test_mae:.4f}")
        print(f"Training R²:  {train_r2:.4f}")
        print(f"Testing R²:   {test_r2:.4f}")
        
        return model, poly
    
    def get_comparison_dataframe(self):
        """Get model comparison as DataFrame"""
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train MSE': metrics['train_mse'],
                'Test MSE': metrics['test_mse'],
                'Train MAE': metrics['train_mae'],
                'Test MAE': metrics['test_mae'],
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Overfitting': metrics['test_mse'] - metrics['train_mse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_best_model(self):
        """Get best model based on test R² score"""
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        
        print("\n" + "="*60)
        print("BEST MODEL SELECTION")
        print("="*60)
        print(f"Best Model: {best_model[0]}")
        print(f"Test R² Score: {best_model[1]['test_r2']:.4f}")
        print(f"Test MSE: {best_model[1]['test_mse']:.4f}")
        print(f"Test MAE: {best_model[1]['test_mae']:.4f}")
        
        # Check for overfitting
        overfitting = best_model[1]['test_mse'] - best_model[1]['train_mse']
        if overfitting > best_model[1]['train_mse'] * 0.2:
            print("⚠ Warning: Signs of overfitting detected!")
        else:
            print("✓ Good generalization to test data")
        
        return best_model[0], best_model[1]
    
    def get_predictions(self):
        """Get predictions for all models"""
        predictions = {}
        
        for model_name, metrics in self.results.items():
            predictions[model_name] = {
                'y_test': self.y_test,
                'y_test_pred': metrics['y_test_pred'],
                'residuals': self.y_test - metrics['y_test_pred']
            }
        
        return predictions
