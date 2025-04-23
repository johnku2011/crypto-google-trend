import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def ensure_normalized_data(df, trend_cols=['bitcoin', 'ethereum'], window_size=12):
    """Ensure normalized versions of trend data exist"""
    for trend_col in trend_cols:
        # Check if normalized columns already exist
        if f'{trend_col}_zscore' not in df.columns:
            # Calculate rolling mean and std
            rolling_mean = df[trend_col].rolling(window=window_size, min_periods=1).mean()
            rolling_std = df[trend_col].rolling(window=window_size, min_periods=1).std()
            
            # Calculate z-score
            df[f'{trend_col}_zscore'] = (df[trend_col] - rolling_mean) / rolling_std
            
            # Calculate relative strength within window
            window_max = df[trend_col].rolling(window=window_size, min_periods=1).max()
            df[f'{trend_col}_relative'] = df[trend_col] / window_max * 100
    
    return df

def detect_price_trends(df, price_col, window=3, pct_threshold=10):
    """Detect positive and negative price trends based on percentage changes and consistent movement"""
    # Calculate percentage change over the window (for threshold-based detection)
    df[f'{price_col}_pct_change'] = df[price_col].pct_change(window) * 100
    
    # Identify positive and negative price trends based on percentage threshold
    df[f'{price_col}_positive_trend'] = df[f'{price_col}_pct_change'] > pct_threshold
    df[f'{price_col}_negative_trend'] = df[f'{price_col}_pct_change'] < -pct_threshold
    
    # Add columns for bull/bear markets based on consistent price movement
    for w in [1, 3, 6]:
        # For bull market: Check if price has been consistently increasing over the window
        bull_markets = []
        bear_markets = []
        
        for i in range(len(df)):
            if i < w:  # Not enough history
                bull_markets.append(False)
                bear_markets.append(False)
                continue
                
            # Get price values over the window
            prices = [df.iloc[i-j][price_col] for j in range(w+1)]
            
            # Bull market: Check if prices are consistently increasing
            is_bull = all(prices[j] < prices[j-1] for j in range(w, 0, -1))
            bull_markets.append(is_bull)
            
            # Bear market: Check if prices are consistently decreasing
            is_bear = all(prices[j] > prices[j-1] for j in range(w, 0, -1))
            bear_markets.append(is_bear)
        
        # Add to dataframe
        df[f'{price_col}_bull_{w}m'] = bull_markets
        df[f'{price_col}_bear_{w}m'] = bear_markets
    
    return df

def detect_trend_peaks_troughs(df, norm_trend_col, threshold=1.5):
    """Detect peaks and troughs in normalized Google Trends data"""
    # Get the base coin name from the norm_trend_col (e.g., 'bitcoin' from 'bitcoin_zscore')
    base_col = norm_trend_col.split('_')[0]
    
    # For normalized data, we can use simple threshold crossing
    df[f'{norm_trend_col}_peak'] = df[norm_trend_col] > threshold
    df[f'{norm_trend_col}_trough'] = df[norm_trend_col] < -threshold
    
    # Also create the columns that main.py is expecting
    df[f'{base_col}_norm_peak'] = df[f'{norm_trend_col}_peak']
    df[f'{base_col}_norm_trough'] = df[f'{norm_trend_col}_trough']
    
    # Also find local maxima and minima for peaks that don't cross threshold
    # Find peaks
    peak_indices, _ = find_peaks(df[norm_trend_col], prominence=0.5)
    
    # Find troughs (peaks in negative data)
    trough_indices, _ = find_peaks(-df[norm_trend_col], prominence=0.5)
    
    # Mark peaks and troughs
    if len(peak_indices) > 0:
        for idx in peak_indices:
            if not df.loc[idx, f'{norm_trend_col}_peak']:  # Only mark if not already marked
                df.loc[idx, f'{norm_trend_col}_peak'] = True
                df.loc[idx, f'{base_col}_norm_peak'] = True
        
    if len(trough_indices) > 0:
        for idx in trough_indices:
            if not df.loc[idx, f'{norm_trend_col}_trough']:  # Only mark if not already marked
                df.loc[idx, f'{norm_trend_col}_trough'] = True
                df.loc[idx, f'{base_col}_norm_trough'] = True
    
    return df

def analyze_trend_price_relationship(df, norm_trend_col, price_col, forward_window=3, pct_threshold=10, use_consistent_movement=True):
    """Analyze if normalized trend peaks/troughs indicate future price trends"""
    results = {
        'peak_followed_by_negative': 0,
        'peak_followed_by_drop': 0,
        'peak_total': 0,
        'trough_followed_by_positive': 0,
        'trough_followed_by_rise': 0,
        'trough_total': 0
    }
    
    # Analyze peaks
    peaks = df[df[f'{norm_trend_col}_peak']]
    results['peak_total'] = len(peaks)
    
    for idx in peaks.index:
        if idx + forward_window < len(df):
            # Check if a negative trend follows
            if use_consistent_movement:
                # Use the consistent movement definition of bear market
                if df.loc[idx+forward_window, f'{price_col}_bear_{forward_window}m']:
                    results['peak_followed_by_negative'] += 1
            else:
                # Use the original percentage-based definition
                if df.loc[idx+1:idx+forward_window, f'{price_col}_negative_trend'].any():
                    results['peak_followed_by_negative'] += 1
                
            # Check price change after peak
            price_before = df.loc[idx, price_col]
            price_after = df.loc[idx+forward_window, price_col]
            if not pd.isna(price_before) and not pd.isna(price_after):
                pct_change = ((price_after - price_before) / price_before) * 100
                if pct_change < -pct_threshold:  # Significant drop
                    results['peak_followed_by_drop'] += 1
    
    # Analyze troughs
    troughs = df[df[f'{norm_trend_col}_trough']]
    results['trough_total'] = len(troughs)
    
    for idx in troughs.index:
        if idx + forward_window < len(df):
            # Check if a positive trend follows
            if use_consistent_movement:
                # Use the consistent movement definition of bull market
                if df.loc[idx+forward_window, f'{price_col}_bull_{forward_window}m']:
                    results['trough_followed_by_positive'] += 1
            else:
                # Use the original percentage-based definition
                if df.loc[idx+1:idx+forward_window, f'{price_col}_positive_trend'].any():
                    results['trough_followed_by_positive'] += 1
                
            # Check price change after trough
            price_before = df.loc[idx, price_col]
            price_after = df.loc[idx+forward_window, price_col]
            if not pd.isna(price_before) and not pd.isna(price_after):
                pct_change = ((price_after - price_before) / price_before) * 100
                if pct_change > pct_threshold:  # Significant rise
                    results['trough_followed_by_rise'] += 1
    
    return results

def plot_normalized_trend_analysis(df, base_col, norm_trend_col, price_col):
    """Plot the relationship between normalized trend peaks/troughs and price trends"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Normalized trend with peaks and troughs
    plt.subplot(2, 1, 1)
    plt.plot(df['Month'], df[norm_trend_col], label=f'Normalized {base_col.capitalize()} Trend')
    
    # Highlight threshold lines
    plt.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Peak Threshold')
    plt.axhline(y=-1.5, color='green', linestyle='--', alpha=0.5, label='Trough Threshold')
    
    # Plot peaks
    peaks = df[df[f'{norm_trend_col}_peak']]
    plt.scatter(peaks['Month'], peaks[norm_trend_col], color='red', marker='^', s=100, label='Peaks')
    
    # Plot troughs
    troughs = df[df[f'{norm_trend_col}_trough']]
    plt.scatter(troughs['Month'], troughs[norm_trend_col], color='green', marker='v', s=100, label='Troughs')
    
    plt.title(f'Normalized {base_col.capitalize()} Trend Peaks and Troughs')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Price with positive and negative trends
    plt.subplot(2, 1, 2)
    plt.plot(df['Month'], df[price_col], label=f'{base_col.capitalize()} Price')
    
    # Highlight positive trend phases
    positive_phases = df[df[f'{price_col}_positive_trend']]
    plt.scatter(positive_phases['Month'], positive_phases[price_col], color='green', marker='o', alpha=0.5, label='Positive Trend')
    
    # Highlight negative trend phases
    negative_phases = df[df[f'{price_col}_negative_trend']]
    plt.scatter(negative_phases['Month'], negative_phases[price_col], color='red', marker='o', alpha=0.5, label='Negative Trend')
    
    plt.title(f'{base_col.capitalize()} Price with Positive and Negative Trends')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    project_root = get_project_root()
    plt.savefig(os.path.join(project_root, 'outputs', f'normalized_trend_analysis_{base_col}.png'))
    plt.close()

def run_analysis(df, trend_cols=['bitcoin', 'ethereum'], forward_windows=[1, 3, 6], threshold=1.5, pct_threshold=10, use_consistent_movement=True):
    """Run normalized trend-price analysis"""
    print("\nRunning Normalized Trend-Price Analysis...")
    
    # Ensure normalized data exists
    df = ensure_normalized_data(df, trend_cols)
    
    all_results = {}
    
    for base_col in trend_cols:
        print(f"\nAnalyzing {base_col.capitalize()} (Normalized):")
        norm_trend_col = f'{base_col}_zscore'
        price_col = f'{base_col}_mean'
        
        # Detect price trends
        df = detect_price_trends(df, price_col, pct_threshold=pct_threshold)
        
        # Detect trend peaks and troughs using normalized data
        df = detect_trend_peaks_troughs(df, norm_trend_col, threshold=threshold)
        
        # Plot overall analysis
        plot_normalized_trend_analysis(df, base_col, norm_trend_col, price_col)
        
        window_results = {}
        for window in forward_windows:
            print(f"\n{window}-Month Forward Window:")
            
            # Analyze relationship between trends and price movements
            results = analyze_trend_price_relationship(
                df, norm_trend_col, price_col, forward_window=window, 
                pct_threshold=pct_threshold, use_consistent_movement=use_consistent_movement
            )
            
            # Save the months of successful hits
            peak_bear_months = []
            trough_bull_months = []
            
            # Analyze peaks
            peaks = df[df[f'{norm_trend_col}_peak']]
            for idx in peaks.index:
                if idx + window < len(df):
                    # Check if followed by bear market
                    if use_consistent_movement:
                        if df.loc[idx+window, f'{price_col}_bear_{window}m']:
                            peak_bear_months.append(df.loc[idx, 'Month'])
                    else:
                        if df.loc[idx+1:idx+window, f'{price_col}_negative_trend'].any():
                            peak_bear_months.append(df.loc[idx, 'Month'])
            
            # Analyze troughs
            troughs = df[df[f'{norm_trend_col}_trough']]
            for idx in troughs.index:
                if idx + window < len(df):
                    # Check if followed by bull market
                    if use_consistent_movement:
                        if df.loc[idx+window, f'{price_col}_bull_{window}m']:
                            trough_bull_months.append(df.loc[idx, 'Month'])
                    else:
                        if df.loc[idx+1:idx+window, f'{price_col}_positive_trend'].any():
                            trough_bull_months.append(df.loc[idx, 'Month'])
            
            # Calculate success rates
            peak_negative_rate = (results['peak_followed_by_negative'] / results['peak_total'] * 100 
                             if results['peak_total'] > 0 else 0)
            peak_drop_rate = (results['peak_followed_by_drop'] / results['peak_total'] * 100 
                          if results['peak_total'] > 0 else 0)
            trough_positive_rate = (results['trough_followed_by_positive'] / results['trough_total'] * 100 
                               if results['trough_total'] > 0 else 0)
            trough_rise_rate = (results['trough_followed_by_rise'] / results['trough_total'] * 100 
                            if results['trough_total'] > 0 else 0)
            
            # Print results
            bull_bear_definition = "consistently increasing/decreasing prices" if use_consistent_movement else "threshold-based trends"
            print(f"Using {bull_bear_definition} to define bull/bear markets")
            print(f"Normalized Peaks followed by Bear Market: {peak_negative_rate:.1f}% ({results['peak_followed_by_negative']}/{results['peak_total']})")
            if peak_bear_months:
                print(f"  Months with peak → bear market: {', '.join(peak_bear_months)}")
            
            print(f"Normalized Peaks followed by >{pct_threshold}% Price Drop: {peak_drop_rate:.1f}% ({results['peak_followed_by_drop']}/{results['peak_total']})")
            
            print(f"Normalized Troughs followed by Bull Market: {trough_positive_rate:.1f}% ({results['trough_followed_by_positive']}/{results['trough_total']})")
            if trough_bull_months:
                print(f"  Months with trough → bull market: {', '.join(trough_bull_months)}")
            
            print(f"Normalized Troughs followed by >{pct_threshold}% Price Rise: {trough_rise_rate:.1f}% ({results['trough_followed_by_rise']}/{results['trough_total']})")
            
            # Store results
            window_results[window] = {
                'peak_negative_rate': peak_negative_rate,
                'peak_drop_rate': peak_drop_rate,
                'trough_positive_rate': trough_positive_rate,
                'trough_rise_rate': trough_rise_rate,
                'peak_total': results['peak_total'],
                'trough_total': results['trough_total'],
                'peak_bear_months': peak_bear_months,
                'trough_bull_months': trough_bull_months
            }
        
        all_results[base_col] = window_results
    
    print("\nNormalized Trend-Price Analysis completed.")
    return df, all_results 