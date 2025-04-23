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

def load_and_prepare_data():
    """Load and prepare the data for analysis"""
    project_root = get_project_root()
    trends_df = pd.read_csv(os.path.join(project_root, 'data', 'google_trend.csv'))
    price_df = pd.read_csv(os.path.join(project_root, 'data', 'crypto_price_data.csv'))

    # Convert date columns to datetime
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df['Month'] = price_df['Date'].dt.strftime('%Y-%m')

    # Calculate monthly averages for prices
    monthly_prices = price_df.groupby('Month').agg({
        'bitcoin_mean': 'mean',
        'ethereum_mean': 'mean',
        'solana_mean': 'mean'
    }).reset_index()

    # Merge the datasets
    merged_df = pd.merge(trends_df, monthly_prices, on='Month', how='inner')
    return merged_df

def calculate_correlations(df):
    """Calculate correlations between trends and prices"""
    numeric_columns = ['bitcoin', 'ethereum', 'crypto', 'bitcoin_mean', 'ethereum_mean', 'solana_mean']
    return df[numeric_columns].corr()

def calculate_lead_lag_correlation(series1, series2, max_lag=12):
    """Calculate lead-lag correlations between two series"""
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0:
            corr = series1.shift(-lag).corr(series2)
        else:
            corr = series1.corr(series2.shift(lag))
        correlations.append(corr)
    return pd.Series(correlations, index=lags)

def plot_analysis(df, correlations):
    """Create visualizations for the analysis"""
    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot 1: Time series comparison
    plt.subplot(2, 2, 1)
    plt.plot(df['Month'], df['bitcoin'], label='Bitcoin Trend')
    plt.plot(df['Month'], df['bitcoin_mean']/df['bitcoin_mean'].max(), label='Bitcoin Price (normalized)')
    plt.xticks(rotation=45)
    plt.title('Bitcoin: Google Trends vs Price')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df['Month'], df['ethereum'], label='Ethereum Trend')
    plt.plot(df['Month'], df['ethereum_mean']/df['ethereum_mean'].max(), label='Ethereum Price (normalized)')
    plt.xticks(rotation=45)
    plt.title('Ethereum: Google Trends vs Price')
    plt.legend()

    # Plot 2: Correlation heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')

    # Plot 3: Scatter plots
    plt.subplot(2, 2, 4)
    plt.scatter(df['bitcoin'], df['bitcoin_mean'], alpha=0.5)
    plt.xlabel('Bitcoin Google Trend')
    plt.ylabel('Bitcoin Price')
    plt.title('Bitcoin Trend vs Price Scatter')

    plt.tight_layout()
    project_root = get_project_root()
    plt.savefig(os.path.join(project_root, 'outputs', 'original_trend_analysis.png'))
    plt.close()

def plot_lead_lag_analysis(df):
    """Plot lead-lag analysis results"""
    # Calculate lead-lag correlations
    bitcoin_lead_lag = calculate_lead_lag_correlation(df['bitcoin'], df['bitcoin_mean'])
    ethereum_lead_lag = calculate_lead_lag_correlation(df['ethereum'], df['ethereum_mean'])

    # Plot lead-lag correlations
    plt.figure(figsize=(15, 6))

    # Bitcoin lead-lag plot
    plt.subplot(1, 2, 1)
    plt.plot(bitcoin_lead_lag.index, bitcoin_lead_lag.values, 'b-', label='Correlation')
    plt.axvline(x=0, color='r', linestyle='--', label='No Lag')
    plt.fill_between(bitcoin_lead_lag.index, bitcoin_lead_lag.values, alpha=0.2)
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation')
    plt.title('Bitcoin: Lead-Lag Analysis\nNegative lag = Trends predict Price')
    plt.grid(True)
    plt.legend()

    # Ethereum lead-lag plot
    plt.subplot(1, 2, 2)
    plt.plot(ethereum_lead_lag.index, ethereum_lead_lag.values, 'g-', label='Correlation')
    plt.axvline(x=0, color='r', linestyle='--', label='No Lag')
    plt.fill_between(ethereum_lead_lag.index, ethereum_lead_lag.values, alpha=0.2)
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation')
    plt.title('Ethereum: Lead-Lag Analysis\nNegative lag = Trends predict Price')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    project_root = get_project_root()
    plt.savefig(os.path.join(project_root, 'outputs', 'original_lead_lag_analysis.png'))
    plt.close()

def run_analysis():
    """Run the complete original trend analysis"""
    print("Running original trend analysis...")
    df = load_and_prepare_data()
    correlations = calculate_correlations(df)
    plot_analysis(df, correlations)
    plot_lead_lag_analysis(df)
    print("Original trend analysis completed. Results saved to outputs/original_trend_analysis.png and outputs/original_lead_lag_analysis.png")
    return df, correlations 