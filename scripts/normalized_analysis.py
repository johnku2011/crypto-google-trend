import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def normalize_trends(df, trend_col, window_size=12):
    """Normalize trends within a rolling window"""
    # Calculate rolling mean and std
    rolling_mean = df[trend_col].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df[trend_col].rolling(window=window_size, min_periods=1).std()
    
    # Calculate z-score
    z_score = (df[trend_col] - rolling_mean) / rolling_std
    
    # Calculate relative strength within window
    window_max = df[trend_col].rolling(window=window_size, min_periods=1).max()
    relative_strength = df[trend_col] / window_max * 100
    
    return z_score, relative_strength

def apply_normalization(df):
    """Apply normalization to all trend columns"""
    # Apply normalization to Bitcoin and Ethereum trends
    df['bitcoin_zscore'], df['bitcoin_relative'] = normalize_trends(df, 'bitcoin')
    df['ethereum_zscore'], df['ethereum_relative'] = normalize_trends(df, 'ethereum')
    df['crypto_zscore'], df['crypto_relative'] = normalize_trends(df, 'crypto')
    
    # Also create columns with names that main.py is expecting
    df['bitcoin_normalized'] = df['bitcoin_zscore']
    df['ethereum_normalized'] = df['ethereum_zscore']
    df['crypto_normalized'] = df['crypto_zscore']
    
    return df

def plot_normalized_analysis(df):
    """Create visualizations for normalized analysis"""
    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot 1: Bitcoin normalized trends
    plt.subplot(2, 2, 1)
    plt.plot(df['Month'], df['bitcoin_zscore'], label='Z-Score')
    plt.plot(df['Month'], df['bitcoin_relative'], label='Relative Strength')
    plt.xticks(rotation=45)
    plt.title('Bitcoin: Normalized Trends')
    plt.legend()
    plt.grid(True)

    # Plot 2: Ethereum normalized trends
    plt.subplot(2, 2, 2)
    plt.plot(df['Month'], df['ethereum_zscore'], label='Z-Score')
    plt.plot(df['Month'], df['ethereum_relative'], label='Relative Strength')
    plt.xticks(rotation=45)
    plt.title('Ethereum: Normalized Trends')
    plt.legend()
    plt.grid(True)

    # Plot 3: Correlation between normalized trends and prices
    plt.subplot(2, 2, 3)
    normalized_corr = df[['bitcoin_zscore', 'ethereum_zscore', 'bitcoin_mean', 'ethereum_mean']].corr()
    sns.heatmap(normalized_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation: Normalized Trends vs Prices')

    # Plot 4: Scatter plot of normalized trends vs prices
    plt.subplot(2, 2, 4)
    plt.scatter(df['bitcoin_zscore'], df['bitcoin_mean'], alpha=0.5)
    plt.xlabel('Bitcoin Normalized Trend (Z-Score)')
    plt.ylabel('Bitcoin Price')
    plt.title('Normalized Trend vs Price Scatter')
    plt.grid(True)

    plt.tight_layout()
    project_root = get_project_root()
    plt.savefig(os.path.join(project_root, 'outputs', 'normalized_trend_analysis.png'))
    plt.close()

def run_analysis(df):
    """Run the complete normalized trend analysis"""
    print("Running normalized trend analysis...")
    df = apply_normalization(df)
    plot_normalized_analysis(df)
    print("Normalized trend analysis completed. Results saved to outputs/normalized_trend_analysis.png")
    return df 