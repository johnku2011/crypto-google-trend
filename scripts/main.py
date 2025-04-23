import pandas as pd
import sys
import os

# Configuration parameters
PRICE_CHANGE_THRESHOLD = 10  # Percentage threshold for significant price changes (used for rise/drop calculations)

# Add scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import analysis modules
from trend_analysis import run_analysis as run_original_analysis
from normalized_analysis import run_analysis as run_normalized_analysis
from breakout_analysis import run_analysis as run_breakout_analysis
from market_phase_analysis import run_analysis as run_market_phase_analysis
from normalized_market_phase_analysis import run_analysis as run_normalized_market_phase_analysis

def main():
    # Make sure outputs directory exists
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Run original trend analysis
    df, correlations = run_original_analysis()
    
    # Run normalized analysis
    df = run_normalized_analysis(df)
    
    # Run breakout analysis
    df, breakout_results = run_breakout_analysis(df)
    
    # Run market phase analysis (original trends)
    df, market_phase_results = run_market_phase_analysis(df, pct_threshold=PRICE_CHANGE_THRESHOLD, use_consistent_movement=True)
    
    # Run market phase analysis (normalized trends)
    df, norm_market_phase_results = run_normalized_market_phase_analysis(df, pct_threshold=PRICE_CHANGE_THRESHOLD, use_consistent_movement=True)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("------------------")
    print("\nOriginal Correlations:")
    print(correlations)
    
    print("\nBreakout Analysis Results:")
    for key, result in breakout_results.items():
        print(f"\n{key}:")
        print(f"Number of positive breakouts: {result['positive_breakouts']}")
        print(f"Number of negative breakouts: {result['negative_breakouts']}")
        print(f"Positive breakout success rate: {result['pos_success_rate']:.2f}%")
        print(f"Negative breakout success rate: {result['neg_success_rate']:.2f}%")
    
    print("\nOriginal Market Phase Analysis Summary:")
    for coin, windows in market_phase_results.items():
        print(f"\n{coin.capitalize()}:")
        for window, result in windows.items():
            print(f"  {window}-Month Window:")
            print(f"    Peak → Bear Market Rate: {result['peak_negative_rate']:.1f}%")
            if 'peak_bear_months' in result and result['peak_bear_months']:
                print(f"    Peak → Bear Market Months: {', '.join(result['peak_bear_months'])}")
            
            print(f"    Peak → >{PRICE_CHANGE_THRESHOLD}% Price Drop Rate: {result['peak_drop_rate']:.1f}%") 
            
            print(f"    Trough → Bull Market Rate: {result['trough_positive_rate']:.1f}%")
            if 'trough_bull_months' in result and result['trough_bull_months']:
                print(f"    Trough → Bull Market Months: {', '.join(result['trough_bull_months'])}")
                
            print(f"    Trough → >{PRICE_CHANGE_THRESHOLD}% Price Rise Rate: {result['trough_rise_rate']:.1f}%")
    
    print("\nNormalized Market Phase Analysis Summary:")
    for coin, windows in norm_market_phase_results.items():
        print(f"\n{coin.capitalize()} (Normalized):")
        for window, result in windows.items():
            print(f"  {window}-Month Window:")
            print(f"    Peak → Bear Market Rate: {result['peak_negative_rate']:.1f}%")
            if 'peak_bear_months' in result and result['peak_bear_months']:
                print(f"    Peak → Bear Market Months: {', '.join(result['peak_bear_months'])}")
                
            print(f"    Peak → >{PRICE_CHANGE_THRESHOLD}% Price Drop Rate: {result['peak_drop_rate']:.1f}%") 
            
            print(f"    Trough → Bull Market Rate: {result['trough_positive_rate']:.1f}%")
            if 'trough_bull_months' in result and result['trough_bull_months']:
                print(f"    Trough → Bull Market Months: {', '.join(result['trough_bull_months'])}")
                
            print(f"    Trough → >{PRICE_CHANGE_THRESHOLD}% Price Rise Rate: {result['trough_rise_rate']:.1f}%")
    
    print("\nComparison of Original vs Normalized Analysis:")
    for coin in market_phase_results.keys():
        print(f"\n{coin.capitalize()}:")
        for window in market_phase_results[coin].keys():
            orig = market_phase_results[coin][window]
            norm = norm_market_phase_results[coin][window]
            print(f"  {window}-Month Window:")
            print(f"    Peak → Bear: Original {orig['peak_negative_rate']:.1f}% vs Normalized {norm['peak_negative_rate']:.1f}%")
            print(f"    Trough → Bull: Original {orig['trough_positive_rate']:.1f}% vs Normalized {norm['trough_positive_rate']:.1f}%")

    # Export results to CSV
    export_results_to_csv(df, correlations, breakout_results, market_phase_results, norm_market_phase_results, outputs_dir)
    
    print("\nResults have been exported to CSV files in the 'outputs' directory.")

def export_results_to_csv(df, correlations, breakout_results, market_phase_results, norm_market_phase_results, outputs_dir):
    """Export all analysis results to CSV files"""
    # Export the main dataframe with all calculated metrics
    df.to_csv(os.path.join(outputs_dir, 'full_analysis_data.csv'), index=False)
    
    # Export correlation matrix
    correlations.to_csv(os.path.join(outputs_dir, 'correlations.csv'))
    
    # Export breakout analysis results
    breakout_data = []
    for key, result in breakout_results.items():
        coin, window = key.split('_')
        breakout_data.append({
            'Coin': coin,
            'Window (months)': window,
            'Positive Breakouts': result['positive_breakouts'],
            'Negative Breakouts': result['negative_breakouts'],
            'Positive Success Rate (%)': result['pos_success_rate'],
            'Negative Success Rate (%)': result['neg_success_rate']
        })
    
    pd.DataFrame(breakout_data).to_csv(os.path.join(outputs_dir, 'breakout_analysis.csv'), index=False)
    
    # Export market phase analysis results (original)
    market_phase_data = []
    for coin, windows in market_phase_results.items():
        for window, result in windows.items():
            data_row = {
                'Coin': coin,
                'Window (months)': window,
                'Analysis Type': 'Original',
                'Peak → Bear Market Rate (%)': result['peak_negative_rate'],
                f'Peak → >{PRICE_CHANGE_THRESHOLD}% Price Drop Rate (%)': result['peak_drop_rate'],
                'Trough → Bull Market Rate (%)': result['trough_positive_rate'],
                f'Trough → >{PRICE_CHANGE_THRESHOLD}% Price Rise Rate (%)': result['trough_rise_rate'],
                'Peak Count': result['peak_total'],
                'Trough Count': result['trough_total']
            }
            
            # Add months for successful cases if available
            if 'peak_bear_months' in result:
                data_row['Peak Bear Market Months'] = '; '.join(result.get('peak_bear_months', []))
            
            if 'trough_bull_months' in result:
                data_row['Trough Bull Market Months'] = '; '.join(result.get('trough_bull_months', []))
                
            market_phase_data.append(data_row)
    
    # Add normalized market phase results
    for coin, windows in norm_market_phase_results.items():
        for window, result in windows.items():
            data_row = {
                'Coin': coin,
                'Window (months)': window,
                'Analysis Type': 'Normalized',
                'Peak → Bear Market Rate (%)': result['peak_negative_rate'],
                f'Peak → >{PRICE_CHANGE_THRESHOLD}% Price Drop Rate (%)': result['peak_drop_rate'],
                'Trough → Bull Market Rate (%)': result['trough_positive_rate'],
                f'Trough → >{PRICE_CHANGE_THRESHOLD}% Price Rise Rate (%)': result['trough_rise_rate'],
                'Peak Count': result['peak_total'],
                'Trough Count': result['trough_total']
            }
            
            # Add months for successful cases if available
            if 'peak_bear_months' in result:
                data_row['Peak Bear Market Months'] = '; '.join(result.get('peak_bear_months', []))
            
            if 'trough_bull_months' in result:
                data_row['Trough Bull Market Months'] = '; '.join(result.get('trough_bull_months', []))
                
            market_phase_data.append(data_row)
    
    pd.DataFrame(market_phase_data).to_csv(os.path.join(outputs_dir, 'market_phase_analysis.csv'), index=False)
    
    # Export detailed signal data showing months with peaks/troughs and subsequent trends
    export_detailed_signal_data(df, outputs_dir)

def export_detailed_signal_data(df, outputs_dir):
    """Export detailed data showing months with peaks/troughs and subsequent trends"""
    # Create a copy of the DataFrame with only essential columns
    signals_df = df[['Month']].copy()
    
    # Add relevant trend and price columns for Bitcoin and Ethereum
    for coin in ['bitcoin', 'ethereum']:
        price_col = f'{coin}_mean'
        
        # Add trend data
        signals_df[f'{coin}_trend'] = df[coin]
        signals_df[f'{coin}_price'] = df[price_col]
        
        # Add normalized trend data if it exists
        norm_col = f'{coin}_normalized'
        zscore_col = f'{coin}_zscore'
        if norm_col in df.columns:
            signals_df[f'{coin}_normalized_trend'] = df[norm_col]
        elif zscore_col in df.columns:
            signals_df[f'{coin}_normalized_trend'] = df[zscore_col]
        
        # Add peak/trough flags - we need to check if these columns exist
        if 'trend_peak' in df.columns and coin == 'bitcoin':
            signals_df[f'{coin}_peak'] = df['trend_peak']
        elif f'{coin}_trend_peak' in df.columns:
            signals_df[f'{coin}_peak'] = df[f'{coin}_trend_peak']
        else:
            signals_df[f'{coin}_peak'] = False
            
        if 'trend_trough' in df.columns and coin == 'bitcoin':
            signals_df[f'{coin}_trough'] = df['trend_trough']
        elif f'{coin}_trend_trough' in df.columns:
            signals_df[f'{coin}_trough'] = df[f'{coin}_trend_trough']
        else:
            signals_df[f'{coin}_trough'] = False
        
        # Add normalized peak/trough flags - check both possible column naming patterns
        norm_peak_col = f'{coin}_norm_peak'
        zscore_peak_col = f'{coin}_zscore_peak'
        if norm_peak_col in df.columns:
            signals_df[f'{coin}_norm_peak'] = df[norm_peak_col]
        elif zscore_peak_col in df.columns:
            signals_df[f'{coin}_norm_peak'] = df[zscore_peak_col]
        else:
            signals_df[f'{coin}_norm_peak'] = False
            
        norm_trough_col = f'{coin}_norm_trough'
        zscore_trough_col = f'{coin}_zscore_trough'
        if norm_trough_col in df.columns:
            signals_df[f'{coin}_norm_trough'] = df[norm_trough_col]
        elif zscore_trough_col in df.columns:
            signals_df[f'{coin}_norm_trough'] = df[zscore_trough_col]
        else:
            signals_df[f'{coin}_norm_trough'] = False
        
        # Add 1, 3, and 6 month price changes
        for window in [1, 3, 6]:
            signals_df[f'{coin}_pct_change_{window}m'] = df[price_col].pct_change(window) * 100
        
        # Add positive/negative trend flags
        if 'positive_trend' in df.columns and coin == 'bitcoin':
            signals_df[f'{coin}_positive_trend'] = df['positive_trend']
        elif f'{coin}_positive_trend' in df.columns:
            signals_df[f'{coin}_positive_trend'] = df[f'{coin}_positive_trend']
        else:
            signals_df[f'{coin}_positive_trend'] = False
            
        if 'negative_trend' in df.columns and coin == 'bitcoin':
            signals_df[f'{coin}_negative_trend'] = df['negative_trend']
        elif f'{coin}_negative_trend' in df.columns:
            signals_df[f'{coin}_negative_trend'] = df[f'{coin}_negative_trend']
        else:
            signals_df[f'{coin}_negative_trend'] = False
            
        # Add bull/bear market flags for each time window
        for window in [1, 3, 6]:
            bull_col = f'{price_col}_bull_{window}m'
            bear_col = f'{price_col}_bear_{window}m'
            
            if bull_col in df.columns:
                signals_df[f'{coin}_bull_{window}m'] = df[bull_col]
            else:
                signals_df[f'{coin}_bull_{window}m'] = False
                
            if bear_col in df.columns:
                signals_df[f'{coin}_bear_{window}m'] = df[bear_col]
            else:
                signals_df[f'{coin}_bear_{window}m'] = False
    
    # Export to CSV
    signals_df.to_csv(os.path.join(outputs_dir, 'detailed_signals.csv'), index=False)
    
    # Create a filtered version with only the peak/trough months for easier analysis
    bitcoin_signals = signals_df[(signals_df['bitcoin_peak'] == True) | 
                               (signals_df['bitcoin_trough'] == True) | 
                               (signals_df['bitcoin_norm_peak'] == True) | 
                               (signals_df['bitcoin_norm_trough'] == True)]
    
    if not bitcoin_signals.empty:
        bitcoin_signals.to_csv(os.path.join(outputs_dir, 'bitcoin_signals.csv'), index=False)
    
    ethereum_signals = signals_df[(signals_df['ethereum_peak'] == True) | 
                                (signals_df['ethereum_trough'] == True) | 
                                (signals_df['ethereum_norm_peak'] == True) | 
                                (signals_df['ethereum_norm_trough'] == True)]
    
    if not ethereum_signals.empty:
        ethereum_signals.to_csv(os.path.join(outputs_dir, 'ethereum_signals.csv'), index=False)
    
    # Create special signals file focused on the specific question of trend peaks/troughs and subsequent market phases
    peak_trough_analysis = []
    
    for coin in ['bitcoin', 'ethereum']:
        price_col = f'{coin}_mean'
        
        # Handle peaks - use more careful column checking
        peaks_col = 'trend_peak' if coin == 'bitcoin' and 'trend_peak' in df.columns else f'{coin}_trend_peak'
        if peaks_col in df.columns:
            peaks = df[df[peaks_col] == True]
            for idx, peak in peaks.iterrows():
                for window in [1, 3, 6]:
                    if idx + window < len(df):
                        # Calculate future price change
                        future_price = df.iloc[idx + window][price_col]
                        price_change_pct = ((future_price - peak[price_col]) / peak[price_col]) * 100
                        
                        # Check if followed by negative trend
                        neg_trend_col = 'negative_trend' if coin == 'bitcoin' and 'negative_trend' in df.columns else f'{coin}_negative_trend'
                        followed_by_negative = False
                        if neg_trend_col in df.columns:
                            followed_by_negative = df.iloc[idx:idx+window+1][neg_trend_col].any()
                        
                        # Add to analysis
                        peak_trough_analysis.append({
                            'Coin': coin,
                            'Signal Type': 'Peak',
                            'Analysis Type': 'Original',
                            'Month': peak['Month'],
                            'Window (months)': window,
                            'Trend Value': peak[coin],
                            'Price': peak[price_col],
                            'Future Price': future_price,
                            'Price Change (%)': price_change_pct,
                            'Followed by Negative Trend': followed_by_negative,
                            f'Price Drop > {PRICE_CHANGE_THRESHOLD}%': price_change_pct < -PRICE_CHANGE_THRESHOLD
                        })
        
        # Handle troughs - use more careful column checking
        troughs_col = 'trend_trough' if coin == 'bitcoin' and 'trend_trough' in df.columns else f'{coin}_trend_trough'
        if troughs_col in df.columns:
            troughs = df[df[troughs_col] == True]
            for idx, trough in troughs.iterrows():
                for window in [1, 3, 6]:
                    if idx + window < len(df):
                        # Calculate future price change
                        future_price = df.iloc[idx + window][price_col]
                        price_change_pct = ((future_price - trough[price_col]) / trough[price_col]) * 100
                        
                        # Check if followed by positive trend
                        pos_trend_col = 'positive_trend' if coin == 'bitcoin' and 'positive_trend' in df.columns else f'{coin}_positive_trend'
                        followed_by_positive = False
                        if pos_trend_col in df.columns:
                            followed_by_positive = df.iloc[idx:idx+window+1][pos_trend_col].any()
                        
                        # Add to analysis
                        peak_trough_analysis.append({
                            'Coin': coin,
                            'Signal Type': 'Trough',
                            'Analysis Type': 'Original',
                            'Month': trough['Month'],
                            'Window (months)': window,
                            'Trend Value': trough[coin],
                            'Price': trough[price_col],
                            'Future Price': future_price,
                            'Price Change (%)': price_change_pct,
                            'Followed by Positive Trend': followed_by_positive,
                            f'Price Rise > {PRICE_CHANGE_THRESHOLD}%': price_change_pct > PRICE_CHANGE_THRESHOLD
                        })
        
        # Handle normalized peaks
        norm_peak_col = f'{coin}_norm_peak'
        if norm_peak_col in df.columns:
            norm_peaks = df[df[norm_peak_col] == True]
            for idx, peak in norm_peaks.iterrows():
                for window in [1, 3, 6]:
                    if idx + window < len(df):
                        # Calculate future price change
                        future_price = df.iloc[idx + window][price_col]
                        price_change_pct = ((future_price - peak[price_col]) / peak[price_col]) * 100
                        
                        # Check if followed by negative trend
                        neg_trend_col = 'negative_trend' if coin == 'bitcoin' and 'negative_trend' in df.columns else f'{coin}_negative_trend'
                        followed_by_negative = False
                        if neg_trend_col in df.columns:
                            followed_by_negative = df.iloc[idx:idx+window+1][neg_trend_col].any()
                        
                        # Check if normalized trend column exists
                        norm_col = f'{coin}_normalized'
                        trend_value = peak[norm_col] if norm_col in df.columns else 0
                        
                        # Add to analysis
                        peak_trough_analysis.append({
                            'Coin': coin,
                            'Signal Type': 'Peak',
                            'Analysis Type': 'Normalized',
                            'Month': peak['Month'],
                            'Window (months)': window,
                            'Trend Value': trend_value,
                            'Price': peak[price_col],
                            'Future Price': future_price,
                            'Price Change (%)': price_change_pct,
                            'Followed by Negative Trend': followed_by_negative,
                            f'Price Drop > {PRICE_CHANGE_THRESHOLD}%': price_change_pct < -PRICE_CHANGE_THRESHOLD
                        })
        
        # Handle normalized troughs
        norm_trough_col = f'{coin}_norm_trough'
        if norm_trough_col in df.columns:
            norm_troughs = df[df[norm_trough_col] == True]
            for idx, trough in norm_troughs.iterrows():
                for window in [1, 3, 6]:
                    if idx + window < len(df):
                        # Calculate future price change
                        future_price = df.iloc[idx + window][price_col]
                        price_change_pct = ((future_price - trough[price_col]) / trough[price_col]) * 100
                        
                        # Check if followed by positive trend
                        pos_trend_col = 'positive_trend' if coin == 'bitcoin' and 'positive_trend' in df.columns else f'{coin}_positive_trend'
                        followed_by_positive = False
                        if pos_trend_col in df.columns:
                            followed_by_positive = df.iloc[idx:idx+window+1][pos_trend_col].any()
                        
                        # Check if normalized trend column exists
                        norm_col = f'{coin}_normalized'
                        trend_value = trough[norm_col] if norm_col in df.columns else 0
                        
                        # Add to analysis
                        peak_trough_analysis.append({
                            'Coin': coin,
                            'Signal Type': 'Trough',
                            'Analysis Type': 'Normalized',
                            'Month': trough['Month'],
                            'Window (months)': window,
                            'Trend Value': trend_value,
                            'Price': trough[price_col],
                            'Future Price': future_price,
                            'Price Change (%)': price_change_pct,
                            'Followed by Positive Trend': followed_by_positive,
                            f'Price Rise > {PRICE_CHANGE_THRESHOLD}%': price_change_pct > PRICE_CHANGE_THRESHOLD
                        })
    
    # Export to CSV
    if peak_trough_analysis:
        df_signals = pd.DataFrame(peak_trough_analysis)
        df_signals.to_csv(os.path.join(outputs_dir, 'peak_trough_signals.csv'), index=False)
        
        # Create a chronological view of signals for easier timeline analysis
        timeline_data = []
        
        for coin in ['bitcoin', 'ethereum']:
            # Original signals
            coin_signals = df_signals[(df_signals['Coin'] == coin) & 
                                     (df_signals['Analysis Type'] == 'Original') & 
                                     (df_signals['Window (months)'] == 1)]  # Use 1-month window to avoid duplicates
            
            for _, row in coin_signals.iterrows():
                signal_type = row['Signal Type']
                month = row['Month']
                trend_value = row['Trend Value']
                price = row['Price']
                
                # Price changes at different windows
                price_changes = {}
                for window in [1, 3, 6]:
                    matching_row = df_signals[(df_signals['Coin'] == coin) & 
                                            (df_signals['Analysis Type'] == 'Original') & 
                                            (df_signals['Signal Type'] == signal_type) &
                                            (df_signals['Month'] == month) &
                                            (df_signals['Window (months)'] == window)]
                    
                    if not matching_row.empty:
                        price_changes[f'{window}m_change'] = matching_row['Price Change (%)'].iloc[0]
                        if signal_type == 'Peak':
                            price_changes[f'{window}m_bear'] = matching_row['Followed by Negative Trend'].iloc[0]
                            price_changes[f'{window}m_drop{PRICE_CHANGE_THRESHOLD}'] = matching_row[f'Price Drop > {PRICE_CHANGE_THRESHOLD}%'].iloc[0]
                        else:  # Trough
                            price_changes[f'{window}m_bull'] = matching_row['Followed by Positive Trend'].iloc[0]
                            price_changes[f'{window}m_rise{PRICE_CHANGE_THRESHOLD}'] = matching_row[f'Price Rise > {PRICE_CHANGE_THRESHOLD}%'].iloc[0]
                
                timeline_data.append({
                    'Month': month,
                    'Coin': coin,
                    'Signal Type': signal_type,
                    'Analysis Type': 'Original',
                    'Trend Value': trend_value,
                    'Price': price,
                    **price_changes
                })
            
            # Normalized signals
            norm_signals = df_signals[(df_signals['Coin'] == coin) & 
                                     (df_signals['Analysis Type'] == 'Normalized') & 
                                     (df_signals['Window (months)'] == 1)]  # Use 1-month window to avoid duplicates
            
            for _, row in norm_signals.iterrows():
                signal_type = row['Signal Type']
                month = row['Month']
                trend_value = row['Trend Value']
                price = row['Price']
                
                # Price changes at different windows
                price_changes = {}
                for window in [1, 3, 6]:
                    matching_row = df_signals[(df_signals['Coin'] == coin) & 
                                            (df_signals['Analysis Type'] == 'Normalized') & 
                                            (df_signals['Signal Type'] == signal_type) &
                                            (df_signals['Month'] == month) &
                                            (df_signals['Window (months)'] == window)]
                    
                    if not matching_row.empty:
                        price_changes[f'{window}m_change'] = matching_row['Price Change (%)'].iloc[0]
                        if signal_type == 'Peak':
                            price_changes[f'{window}m_bear'] = matching_row['Followed by Negative Trend'].iloc[0]
                            price_changes[f'{window}m_drop{PRICE_CHANGE_THRESHOLD}'] = matching_row[f'Price Drop > {PRICE_CHANGE_THRESHOLD}%'].iloc[0]
                        else:  # Trough
                            price_changes[f'{window}m_bull'] = matching_row['Followed by Positive Trend'].iloc[0]
                            price_changes[f'{window}m_rise{PRICE_CHANGE_THRESHOLD}'] = matching_row[f'Price Rise > {PRICE_CHANGE_THRESHOLD}%'].iloc[0]
                
                timeline_data.append({
                    'Month': month,
                    'Coin': coin,
                    'Signal Type': signal_type,
                    'Analysis Type': 'Normalized',
                    'Trend Value': trend_value,
                    'Price': price,
                    **price_changes
                })
        
        # Create timeline DataFrame and sort by date
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df['Month'] = pd.to_datetime(timeline_df['Month'])
            timeline_df = timeline_df.sort_values(['Coin', 'Month'])
            timeline_df.to_csv(os.path.join(outputs_dir, 'signal_timeline.csv'), index=False)

if __name__ == "__main__":
    main() 