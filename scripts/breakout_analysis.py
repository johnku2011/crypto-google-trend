import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def detect_breakouts(df, trend_col, window_size=12, threshold=2):
    """Detect significant breakouts in trends"""
    # Calculate rolling mean and std
    rolling_mean = df[trend_col].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df[trend_col].rolling(window=window_size, min_periods=1).std()
    
    # Calculate z-score
    z_score = (df[trend_col] - rolling_mean) / rolling_std
    
    # Detect positive and negative breakouts separately
    positive_breakouts = z_score > threshold
    negative_breakouts = z_score < -threshold
    
    return positive_breakouts, negative_breakouts, z_score

def analyze_breakout_performance(df, trend_col, price_col, positive_breakouts, negative_breakouts, window_after=6):
    """Analyze market performance after breakouts"""
    # Initialize results DataFrame with all required columns
    results = pd.DataFrame(columns=['date', 'price_before', 'price_after', 'pct_change', 
                                  'trend_value', 'z_score', 'type'])
    
    # Analyze positive breakouts
    pos_indices = positive_breakouts[positive_breakouts].index
    for idx in pos_indices:
        if idx + window_after < len(df):
            price_before = df[price_col].iloc[idx]
            price_after = df[price_col].iloc[idx + window_after]
            
            if pd.isna(price_before) or pd.isna(price_after):
                continue
                
            pct_change = ((price_after - price_before) / price_before) * 100
            
            results.loc[len(results)] = {
                'date': df['Month'].iloc[idx],
                'price_before': price_before,
                'price_after': price_after,
                'pct_change': pct_change,
                'trend_value': df[trend_col].iloc[idx],
                'z_score': df[f'{trend_col}_zscore'].iloc[idx] if f'{trend_col}_zscore' in df.columns else None,
                'type': 'positive'
            }
    
    # Analyze negative breakouts
    neg_indices = negative_breakouts[negative_breakouts].index
    for idx in neg_indices:
        if idx + window_after < len(df):
            price_before = df[price_col].iloc[idx]
            price_after = df[price_col].iloc[idx + window_after]
            
            if pd.isna(price_before) or pd.isna(price_after):
                continue
                
            pct_change = ((price_after - price_before) / price_before) * 100
            
            results.loc[len(results)] = {
                'date': df['Month'].iloc[idx],
                'price_before': price_before,
                'price_after': price_after,
                'pct_change': pct_change,
                'trend_value': df[trend_col].iloc[idx],
                'z_score': df[f'{trend_col}_zscore'].iloc[idx] if f'{trend_col}_zscore' in df.columns else None,
                'type': 'negative'
            }
    
    return results

def plot_breakout_analysis(df, results, trend_col, window_after):
    """Plot breakout analysis results"""
    if len(results) == 0:
        print(f"No breakouts found for {trend_col} with {window_after}-month window")
        return
    
    # Remove any rows with NaN values
    results = results.dropna()
    
    if len(results) == 0:
        print(f"No valid data points found for {trend_col} with {window_after}-month window")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Breakout points on trend
    plt.subplot(2, 2, 1)
    plt.plot(df['Month'], df[trend_col], label='Trend')
    
    # Plot positive breakouts
    pos_breakouts = results[results['type'] == 'positive']
    if len(pos_breakouts) > 0:
        plt.scatter(pos_breakouts['date'], pos_breakouts['trend_value'], 
                   color='red', label='Positive Breakouts')
    
    # Plot negative breakouts
    neg_breakouts = results[results['type'] == 'negative']
    if len(neg_breakouts) > 0:
        plt.scatter(neg_breakouts['date'], neg_breakouts['trend_value'], 
                   color='blue', label='Negative Breakouts')
    
    plt.xticks(rotation=45)
    plt.title(f'{trend_col.capitalize()}: Breakout Points')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Price changes after breakouts
    plt.subplot(2, 2, 2)
    if len(pos_breakouts) > 0:
        plt.bar(pos_breakouts['date'], pos_breakouts['pct_change'], color='red', label='Positive')
    if len(neg_breakouts) > 0:
        plt.bar(neg_breakouts['date'], neg_breakouts['pct_change'], color='blue', label='Negative')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.title(f'Price Change {window_after} Months After Breakout')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Distribution of price changes
    plt.subplot(2, 2, 3)
    if len(results['pct_change'].dropna()) > 0:
        plt.hist(results['pct_change'].dropna(), bins=20)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Distribution of Price Changes')
        plt.xlabel('Percentage Change')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No valid data points', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Distribution of Price Changes')
    
    # Plot 4: Scatter plot of trend value vs price change
    plt.subplot(2, 2, 4)
    if len(results.dropna(subset=['trend_value', 'pct_change'])) > 0:
        plt.scatter(results['trend_value'], results['pct_change'], 
                   c=['red' if t == 'positive' else 'blue' for t in results['type']])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Trend Value vs Price Change')
        plt.xlabel('Trend Value')
        plt.ylabel('Price Change (%)')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No valid data points', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Trend Value vs Price Change')
    
    plt.tight_layout()
    project_root = get_project_root()
    plt.savefig(os.path.join(project_root, 'outputs', f'breakout_analysis_{trend_col}_{window_after}months.png'))
    plt.close()

def run_analysis(df, trend_cols=['bitcoin', 'ethereum'], windows=[1, 3, 6], threshold=2):
    """Run breakout analysis for multiple time windows"""
    print("Running breakout analysis...")
    
    all_results = {}
    
    for trend_col in trend_cols:
        for window in windows:
            print(f"\nAnalyzing {trend_col} with {window}-month window...")
            
            # Detect breakouts
            positive_breakouts, negative_breakouts, z_scores = detect_breakouts(
                df, trend_col, window_size=window, threshold=threshold
            )
            
            # Analyze performance
            price_col = f'{trend_col}_mean'
            results = analyze_breakout_performance(
                df, trend_col, price_col, positive_breakouts, negative_breakouts, window_after=window
            )
            
            # Calculate statistics
            pos_results = results[results['type'] == 'positive'] if len(results) > 0 else pd.DataFrame()
            neg_results = results[results['type'] == 'negative'] if len(results) > 0 else pd.DataFrame()
            
            pos_success_rate = (pos_results['pct_change'] > 0).mean() * 100 if len(pos_results) > 0 else 0
            neg_success_rate = (neg_results['pct_change'] < 0).mean() * 100 if len(neg_results) > 0 else 0
            
            # Store results
            all_results[f'{trend_col}_{window}m'] = {
                'positive_breakouts': len(pos_results),
                'negative_breakouts': len(neg_results),
                'pos_success_rate': pos_success_rate,
                'neg_success_rate': neg_success_rate,
                'results': results
            }
            
            # Print notable breakouts
            print(f"\n{trend_col.capitalize()}:")
            if len(pos_results) > 0:
                print(f"{pos_success_rate:.1f}% success rate for positive breakouts")
                notable = pos_results.nlargest(2, 'pct_change')
                for _, row in notable.iterrows():
                    print(f"Notable: {row['date']} (Z={row['z_score']:.2f}) → {row['pct_change']:.2f}%")
            
            if len(neg_results) > 0:
                print(f"{neg_success_rate:.1f}% success rate for negative breakouts")
                notable = neg_results.nsmallest(2, 'pct_change')
                for _, row in notable.iterrows():
                    print(f"Notable: {row['date']} (Z={row['z_score']:.2f}) → {row['pct_change']:.2f}%")
            
            # Plot results if breakouts were found
            if len(results) > 0:
                plot_breakout_analysis(df, results, trend_col, window)
    
    print("\nBreakout analysis completed.")
    return df, all_results