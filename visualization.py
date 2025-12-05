"""
Visualization Module
Creates plots for clustering and regression analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

class Visualizer:
    """Create visualizations for ML analysis"""
    
    @staticmethod
    def plot_elbow_curve(k_range, wcss_values, silhouette_scores, optimal_k, save_path=None):
        """
        Plot elbow curve for k-means
        
        Parameters:
        -----------
        k_range : list
            Range of k values
        wcss_values : list
            WCSS values for each k
        silhouette_scores : list
            Silhouette scores for each k
        optimal_k : int
            Optimal k value
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # WCSS curve
        ax1.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12, fontweight='bold')
        ax1.set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xticks(k_range)
        
        # Silhouette score curve
        ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xticks(k_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_missing_values(missing_data, save_path=None):
        """
        Plot missing values distribution
        
        Parameters:
        -----------
        missing_data : dict
            Dictionary with column names and missing counts
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        columns = list(missing_data.keys())
        counts = list(missing_data.values())
        
        ax.bar(columns, counts, color='coral', edgecolor='black')
        ax.set_xlabel('Columns', fontsize=12, fontweight='bold')
        ax.set_ylabel('Missing Count', fontsize=12, fontweight='bold')
        ax.set_title('Missing Values Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_outliers_before_after(df_before, df_after, columns, save_path=None):
        """
        Plot box plots before and after outlier treatment
        
        Parameters:
        -----------
        df_before : pd.DataFrame
            Original dataframe
        df_after : pd.DataFrame
            Dataframe after outlier treatment
        columns : list
            Columns to plot
        save_path : str, optional
            Path to save figure
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(2, n_cols, figsize=(14, 8))
        
        if n_cols == 1:
            axes = axes.reshape(2, 1)
        
        for idx, col in enumerate(columns):
            # Before
            axes[0, idx].boxplot(df_before[col].dropna())
            axes[0, idx].set_title(f'{col} (Before)', fontweight='bold')
            axes[0, idx].set_ylabel('Value')
            axes[0, idx].grid(True, alpha=0.3)
            
            # After
            axes[1, idx].boxplot(df_after[col].dropna())
            axes[1, idx].set_title(f'{col} (After)', fontweight='bold')
            axes[1, idx].set_ylabel('Value')
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_clusters(X, labels, centroids, feature_cols, save_path=None):
        """
        Plot clusters in 2D space
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        labels : np.ndarray
            Cluster labels
        centroids : np.ndarray
            Centroid coordinates
        feature_cols : list
            Feature column names
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points colored by cluster
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='black')
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, 
                  edgecolors='black', linewidth=2, label='Centroids', zorder=5)
        
        ax.set_xlabel(feature_cols[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_cols[1], fontsize=12, fontweight='bold')
        ax.set_title('K-Means Clustering Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12, fontweight='bold')
        
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_regression_comparison(results, save_path=None):
        """
        Plot regression model comparison
        
        Parameters:
        -----------
        results : dict
            Dictionary with model results
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(results.keys())
        r2_scores = [results[m]['test_r2'] for m in models]
        mse_scores = [results[m]['test_mse'] for m in models]
        
        # R² comparison
        axes[0].bar(models, r2_scores, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # MSE comparison
        axes[1].bar(models, mse_scores, color='salmon', edgecolor='black')
        axes[1].set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        axes[1].set_title('Model MSE Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred, residuals=None, save_path=None):
        """
        Plot actual vs predicted values and residuals
        
        Parameters:
        -----------
        y_test : np.ndarray
            Actual test values
        y_pred : np.ndarray
            Predicted values
        residuals : np.ndarray, optional
            Residuals (actual - predicted)
        save_path : str, optional
            Path to save figure
        """
        if residuals is None:
            residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors='black')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Residuals
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
