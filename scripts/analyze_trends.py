import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

# Load the data
trends_df = pd.read_csv('google_trend.csv')
price_df = pd.read_csv('crypto_price_data.csv')

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

# Select only numeric columns for correlation
numeric_columns = ['bitcoin', 'ethereum', 'crypto', 'bitcoin_mean', 'ethereum_mean', 'solana_mean']
correlations = merged_df[numeric_columns].corr()

# Create visualizations for original analysis
plt.figure(figsize=(15, 10))

# Plot 1: Time series comparison
plt.subplot(2, 2, 1)
plt.plot(merged_df['Month'], merged_df['bitcoin'], label='Bitcoin Trend')
plt.plot(merged_df['Month'], merged_df['bitcoin_mean']/merged_df['bitcoin_mean'].max(), label='Bitcoin Price (normalized)')
plt.xticks(rotation=45)
plt.title('Bitcoin: Google Trends vs Price')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(merged_df['Month'], merged_df['ethereum'], label='Ethereum Trend')
plt.plot(merged_df['Month'], merged_df['ethereum_mean']/merged_df['ethereum_mean'].max(), label='Ethereum Price (normalized)')
plt.xticks(rotation=45)
plt.title('Ethereum: Google Trends vs Price')
plt.legend()

# Plot 2: Correlation heatmap
plt.subplot(2, 2, 3)
sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

# Plot 3: Scatter plots
plt.subplot(2, 2, 4)
plt.scatter(merged_df['bitcoin'], merged_df['bitcoin_mean'], alpha=0.5)
plt.xlabel('Bitcoin Google Trend')
plt.ylabel('Bitcoin Price')
plt.title('Bitcoin Trend vs Price Scatter')

plt.tight_layout()
plt.savefig('trend_analysis.png')
plt.close()

# Calculate lead-lag correlations
def calculate_lead_lag_correlation(series1, series2, max_lag=12):
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0:
            corr = series1.shift(-lag).corr(series2)
        else:
            corr = series1.corr(series2.shift(lag))
        correlations.append(corr)
    return pd.Series(correlations, index=lags)

# Calculate lead-lag correlations for both Bitcoin and Ethereum
bitcoin_lead_lag = calculate_lead_lag_correlation(merged_df['bitcoin'], merged_df['bitcoin_mean'])
ethereum_lead_lag = calculate_lead_lag_correlation(merged_df['ethereum'], merged_df['ethereum_mean'])

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
plt.savefig('lead_lag_analysis.png')
plt.close()

# Data Cleansing and Normalization Functions
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

def detect_trend_breakouts(df, trend_col, window_size=12, threshold=2):
    """Detect significant breakouts in trends"""
    z_score, _ = normalize_trends(df, trend_col, window_size)
    breakouts = (z_score > threshold) | (z_score < -threshold)
    return breakouts

# Apply normalization to Bitcoin and Ethereum trends
merged_df['bitcoin_zscore'], merged_df['bitcoin_relative'] = normalize_trends(merged_df, 'bitcoin')
merged_df['ethereum_zscore'], merged_df['ethereum_relative'] = normalize_trends(merged_df, 'ethereum')

# Detect breakouts
merged_df['bitcoin_breakout'] = detect_trend_breakouts(merged_df, 'bitcoin')
merged_df['ethereum_breakout'] = detect_trend_breakouts(merged_df, 'ethereum')

# Function to analyze market direction after breakouts
def analyze_breakout_performance(df, trend_col, price_col, breakout_col, window_after=6):
    breakouts = df[df[breakout_col]]
    results = []
    
    for idx in breakouts.index:
        if idx + window_after < len(df):
            price_before = df[price_col].iloc[idx]
            price_after = df[price_col].iloc[idx + window_after]
            pct_change = ((price_after - price_before) / price_before) * 100
            
            results.append({
                'date': df['Month'].iloc[idx],
                'trend_value': df[trend_col].iloc[idx],
                'z_score': df[f'{trend_col}_zscore'].iloc[idx],
                'price_change_pct': pct_change
            })
    
    return pd.DataFrame(results)

# Analyze breakout performance for different time windows
time_windows = [1, 3, 6]
btc_breakouts = {}
eth_breakouts = {}

for window in time_windows:
    btc_breakouts[window] = analyze_breakout_performance(merged_df, 'bitcoin', 'bitcoin_mean', 'bitcoin_breakout', window)
    eth_breakouts[window] = analyze_breakout_performance(merged_df, 'ethereum', 'ethereum_mean', 'ethereum_breakout', window)

# Create visualizations for breakout analysis
plt.figure(figsize=(20, 12))

# Bitcoin Analysis
plt.subplot(2, 1, 1)
plt.plot(merged_df['Month'], merged_df['bitcoin_zscore'], label='Bitcoin Z-Score', color='blue', alpha=0.6)
plt.plot(merged_df['Month'], merged_df['bitcoin_mean']/merged_df['bitcoin_mean'].max()*100, 
         label='Bitcoin Price (normalized)', color='orange', alpha=0.6)
plt.scatter(btc_breakouts[6]['date'], btc_breakouts[6]['z_score'], 
           c='red', marker='^', label='Breakout Points')
plt.axhline(y=2, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
plt.title('Bitcoin: Normalized Trends and Breakouts')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Ethereum Analysis
plt.subplot(2, 1, 2)
plt.plot(merged_df['Month'], merged_df['ethereum_zscore'], label='Ethereum Z-Score', color='blue', alpha=0.6)
plt.plot(merged_df['Month'], merged_df['ethereum_mean']/merged_df['ethereum_mean'].max()*100, 
         label='Ethereum Price (normalized)', color='orange', alpha=0.6)
plt.scatter(eth_breakouts[6]['date'], eth_breakouts[6]['z_score'], 
           c='red', marker='^', label='Breakout Points')
plt.axhline(y=2, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
plt.title('Ethereum: Normalized Trends and Breakouts')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('normalized_trends_analysis.png')
plt.close()

# Print analysis results
print("\nOriginal Analysis Findings:")
print("\nBitcoin:")
print(f"- Maximum correlation: {bitcoin_lead_lag.max():.3f} at lag {bitcoin_lead_lag.idxmax()} months")
print(f"- Best predictive correlation: {bitcoin_lead_lag[bitcoin_lead_lag.index < 0].max():.3f} at lag {bitcoin_lead_lag[bitcoin_lead_lag.index < 0].idxmax()} months")

print("\nEthereum:")
print(f"- Maximum correlation: {ethereum_lead_lag.max():.3f} at lag {ethereum_lead_lag.idxmax()} months")
print(f"- Best predictive correlation: {ethereum_lead_lag[ethereum_lead_lag.index < 0].max():.3f} at lag {ethereum_lead_lag[ethereum_lead_lag.index < 0].idxmax()} months")

# Print breakout analysis for different time windows
for window in time_windows:
    print(f"\nBreakout Analysis ({window} month window):")
    
    print("\nBitcoin Breakouts:")
    print(f"Number of significant breakouts: {len(btc_breakouts[window])}")
    print("\nBreakout Events:")
    for _, breakout in btc_breakouts[window].iterrows():
        print(f"Date: {breakout['date']}, Z-Score: {breakout['z_score']:.2f}, "
              f"{window}-month Price Change: {breakout['price_change_pct']:.2f}%")

    print("\nEthereum Breakouts:")
    print(f"Number of significant breakouts: {len(eth_breakouts[window])}")
    print("\nBreakout Events:")
    for _, breakout in eth_breakouts[window].iterrows():
        print(f"Date: {breakout['date']}, Z-Score: {breakout['z_score']:.2f}, "
              f"{window}-month Price Change: {breakout['price_change_pct']:.2f}%")

    # Calculate success rates for breakouts
    def calculate_breakout_success(breakouts, threshold=0):
        positive_breakouts = breakouts[breakouts['z_score'] > 0]
        negative_breakouts = breakouts[breakouts['z_score'] < 0]
        
        positive_success = sum(1 for _, b in positive_breakouts.iterrows() if b['price_change_pct'] > threshold)
        negative_success = sum(1 for _, b in negative_breakouts.iterrows() if b['price_change_pct'] < -threshold)
        
        return {
            'positive_success_rate': (positive_success/len(positive_breakouts)*100) if len(positive_breakouts) > 0 else 0,
            'negative_success_rate': (negative_success/len(negative_breakouts)*100) if len(negative_breakouts) > 0 else 0
        }

    btc_breakout_success = calculate_breakout_success(btc_breakouts[window])
    eth_breakout_success = calculate_breakout_success(eth_breakouts[window])

    print(f"\n{window}-month Breakout Success Rates:")
    print(f"Bitcoin Positive Breakouts: {btc_breakout_success['positive_success_rate']:.1f}%")
    print(f"Bitcoin Negative Breakouts: {btc_breakout_success['negative_success_rate']:.1f}%")
    print(f"Ethereum Positive Breakouts: {eth_breakout_success['positive_success_rate']:.1f}%")
    print(f"Ethereum Negative Breakouts: {eth_breakout_success['negative_success_rate']:.1f}%") 