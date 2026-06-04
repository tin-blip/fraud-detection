"""
Visualization utilities for fraud detection project
Contains functions for EDA, model evaluation, and SHAP plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Set default figure size
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plot class distribution (fraud vs legitimate)
    
    Parameters:
    -----------
    y : array-like
        Target variable
    title : str
        Plot title
    save_path : str
        Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    class_counts = pd.Series(y).value_counts()
    axes[0].bar(['Legitimate', 'Fraud'], class_counts.values, color=['steelblue', 'coral'])
    axes[0].set_title(title, fontsize=14)
    axes[0].set_ylabel('Count')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.2f%%',
                colors=['steelblue', 'coral'], explode=(0, 0.1))
    axes[1].set_title('Class Balance', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_numeric_distribution(df, numeric_cols, target_col='class', save_path=None):
    """
    Plot distribution of numeric features by target class
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    numeric_cols : list
        List of numeric column names
    target_col : str
        Target column name
    save_path : str
        Path to save the figure (optional)
    """
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        for label in [0, 1]:
            data = df[df[target_col] == label][col]
            label_name = 'Legitimate' if label == 0 else 'Fraud'
            axes[idx].hist(data, bins=30, alpha=0.7, label=label_name)
        
        axes[idx].set_title(f'{col} Distribution', fontsize=12)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_categorical_fraud_rate(df, categorical_cols, target_col='class', save_path=None):
    """
    Plot fraud rate by categorical variables
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    target_col : str
        Target column name
    save_path : str
        Path to save the figure (optional)
    """
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        fraud_rate = df.groupby(col)[target_col].mean() * 100
        fraud_rate.sort_values(ascending=False, inplace=True)
        
        axes[idx].bar(range(len(fraud_rate)), fraud_rate.values, color='coral')
        axes[idx].set_xticks(range(len(fraud_rate)))
        axes[idx].set_xticklabels(fraud_rate.index, rotation=45, ha='right')
        axes[idx].set_title(f'Fraud Rate by {col}', fontsize=12)
        axes[idx].set_ylabel('Fraud Rate (%)')
    
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

# Heatmap
    heatmap_data = results_df.set_index('Model')[['Precision', 'Recall', 'F1-Score', 'AUC-PR']]
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.3f', ax=axes[1])
    axes[1].set_title('Model Performance Heatmap', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_feature_importance(feature_names, importance_values, model_name="Model", 
                           top_n=15, save_path=None):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importance_values : array-like
        Feature importance values
    model_name : str
        Name of the model
    top_n : int
        Number of top features to show
    save_path : str
        Path to save the figure (optional)
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    return importance_df


def plot_time_series_fraud(df, time_col='purchase_time', target_col='class', save_path=None):
    """
    Plot fraud rate over time
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    time_col : str
        Time column name
    target_col : str
        Target column name
    save_path : str
        Path to save the figure (optional)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df[time_col]).dt.date
    daily_fraud = df.groupby('date')[target_col].mean() * 100
    
    plt.figure(figsize=(14, 6))
    plt.plot(daily_fraud.index, daily_fraud.values, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.title('Fraud Rate Over Time', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_hourly_fraud_pattern(df, time_col='purchase_time', target_col='class', save_path=None):
    """
    Plot fraud rate by hour of day
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    time_col : str
        Time column name
    target_col : str
        Target column name
    save_path : str
        Path to save the figure (optional)
    """
    df = df.copy()
    df['hour'] = pd.to_datetime(df[time_col]).dt.hour
    hourly_fraud = df.groupby('hour')[target_col].mean() * 100
    
    plt.figure(figsize=(12, 5))
    plt.bar(hourly_fraud.index, hourly_fraud.values, color='coral', edgecolor='black')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.title('Fraud Rate by Hour of Day', fontsize=14)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    return hourly_fraud