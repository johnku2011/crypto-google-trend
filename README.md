# Crypto Google Trends Analysis

This project analyzes the relationship between Google Trends data for cryptocurrencies and their price movements, with a focus on using trend peaks and troughs as potential market signals.

## Overview

The analysis explores whether Google search interest for cryptocurrencies like Bitcoin and Ethereum can serve as leading indicators for price movements. It implements various methods:

1. **Basic Correlation Analysis**: Measures direct relationships between Google Trends and crypto prices
2. **Lead-Lag Analysis**: Examines if trends predict price movements (or vice versa)
3. **Breakout Detection**: Identifies significant spikes in search interest as potential signals
4. **Market Phase Analysis**: Studies how trend peaks/troughs relate to subsequent market phases

## Key Features

- Analysis of both raw Google Trends data and normalized versions (using Z-scores)
- Multiple time windows (1, 3, and 6 months) for forward-looking analysis
- Configurable price change threshold (default: 10%) for significant movements
- Detailed signal analysis showing success rates of various indicators
- Comprehensive data export for further analysis

## Configuration

The main configuration parameters are set in `scripts/main.py`:

```python
# Configuration parameters
PRICE_CHANGE_THRESHOLD = 10  # Percentage threshold for significant price changes
```

This threshold is used throughout the analysis to identify significant price movements. You can adjust this value based on your risk tolerance and the volatility characteristics of the assets being analyzed.

## Running the Analysis

To run the analysis:

```bash
python run.py
```

This will:
1. Load data from the `data` directory
2. Run all analysis modules
3. Generate visualizations and CSV exports in the `outputs` directory
4. Print summary statistics to the console

## Outputs

The analysis produces the following outputs:

1. **Visualizations**:
   - Time series plots showing trends vs. prices
   - Correlation heatmaps
   - Lead-lag relationship charts
   - Market phase analyses with peaks/troughs marked

2. **CSV Data**:
   - `full_analysis_data.csv`: Complete dataset with all calculated metrics
   - `correlations.csv`: Correlation matrix between trends and prices
   - `breakout_analysis.csv`: Breakout detection results
   - `market_phase_analysis.csv`: Success rates of market phase predictions
   - `detailed_signals.csv`: Comprehensive data for all time periods
   - `bitcoin_signals.csv` & `ethereum_signals.csv`: Filtered data showing only signal periods
   - `peak_trough_signals.csv`: Detailed analysis of each peak/trough event
   - `signal_timeline.csv`: Chronological view of signals with outcome metrics

## Interpreting Results

The key metrics to focus on include:

- **Peak → Bear Market Rate**: Percentage of trend peaks followed by negative price trends
- **Peak → >X% Price Drop Rate**: Percentage of trend peaks followed by price drops exceeding the threshold
- **Trough → Bull Market Rate**: Percentage of trend troughs followed by positive price trends
- **Trough → >X% Price Rise Rate**: Percentage of trend troughs followed by price rises exceeding the threshold

Where X is the configurable price change threshold (default: 10%).

## Data Sources

- `data/crypto_price_data.csv`: Historical price data for Bitcoin, Ethereum, and Solana
- `data/google_trend.csv`: Google Trends data for crypto-related search terms

## Project Structure

```
├── data/                 # Data files
│   ├── google_trend.csv  # Google Trends data
│   └── crypto_price_data.csv # Cryptocurrency price data
│
├── scripts/              # Analysis scripts
│   ├── trend_analysis.py           # Original trend analysis
│   ├── normalized_analysis.py      # Normalized trend analysis
│   ├── breakout_analysis.py        # Breakout detection and analysis
│   ├── market_phase_analysis.py    # Market phase analysis (original)
│   ├── normalized_market_phase_analysis.py # Market phase analysis (normalized)
│   └── main.py                     # Main analysis controller
│
├── outputs/              # Analysis outputs (charts, figures)
│
├── run.py                # Main script to run the analysis
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

The analysis requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy

You can install these packages with:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Data Files

- `google_trend.csv`: Monthly Google Trends data for Bitcoin, Ethereum, and general "crypto" searches
- `crypto_price_data.csv`: Monthly price data for cryptocurrencies

### Running the Analysis

To run the complete analysis, execute:

```bash
python run.py
```

This will:
1. Run the original trend correlation analysis
2. Perform normalized trend analysis
3. Detect and analyze breakouts
4. Analyze market phases with original and normalized data
5. Save all visualizations to the `outputs/` directory

## Technical Methodology

### Data Normalization

Two types of normalization techniques are applied to the Google Trends data:

1. **Z-Score Normalization**
   - Formula: `Z = (X - μ) / σ`
   - Implementation: For each point in the time series, we calculate the z-score using a rolling window (default: 12 months)
   - The rolling mean (μ) and standard deviation (σ) allow the normalization to adapt to changing baseline popularity
   - This helps identify unusual spikes or drops relative to recent history
   - Code snippet:
     ```python
     rolling_mean = df[trend_col].rolling(window=window_size, min_periods=1).mean()
     rolling_std = df[trend_col].rolling(window=window_size, min_periods=1).std()
     z_score = (df[trend_col] - rolling_mean) / rolling_std
     ```

2. **Relative Strength Normalization**
   - Each data point is expressed as a percentage of the maximum value within a rolling window
   - Formula: `Relative_Strength = (Value / Window_Max) * 100`
   - This measures how strong the current trend is relative to recent peaks
   - Code snippet:
     ```python
     window_max = df[trend_col].rolling(window=window_size, min_periods=1).max()
     relative_strength = df[trend_col] / window_max * 100
     ```

### Market Phase Detection

For the purpose of this analysis, we use simplified metrics to identify significant price movements, which should be distinguished from traditional bull and bear market definitions:

- **Positive Price Trend**
  - A period where the price has increased by more than a threshold percentage (default: 10%) over a specific window (1, 3, or 6 months)
  - Formula: `Price_Change = ((Price_Current - Price_Previous) / Price_Previous) * 100`
  - If `Price_Change > threshold`, the period is classified as having a positive trend
  - Code snippet:
    ```python
    df['pct_change'] = df[price_col].pct_change(window) * 100
    df['bull_market'] = df['pct_change'] > pct_threshold
    ```

- **Negative Price Trend**
  - A period where the price has decreased by more than a threshold percentage (default: 10%) over a specific window
  - Formula: Same as positive trend, but looking for negative changes
  - If `Price_Change < -threshold`, the period is classified as having a negative trend
  - Code snippet:
    ```python
    df['bear_market'] = df['pct_change'] < -pct_threshold
    ```

> **Note:** These thresholds are more sensitive than traditional definitions of bull and bear markets, which typically require longer-term movements (e.g., 20% rise/fall from recent lows/highs over an extended period). Our approach is designed to capture shorter-term price reactions that might follow Google Trends signals rather than long-term market regime changes.

### Peak and Trough Detection

The analysis uses two different methods to detect peaks and troughs in the trend data:

1. **For Original Data (market_phase_analysis.py)**
   - Uses SciPy's `find_peaks` function with a prominence parameter (default: 5)
   - Prominence measures how much a peak stands out relative to surrounding values
   - Troughs are detected as peaks in the negative of the data
   - Code snippet:
     ```python
     from scipy.signal import find_peaks
     peak_indices, _ = find_peaks(df[trend_col], prominence=prominence)
     trough_indices, _ = find_peaks(-df[trend_col], prominence=prominence)
     ```

2. **For Normalized Data (normalized_market_phase_analysis.py)**
   - Primary method: Threshold crossing on Z-scores
     - A peak is when the Z-score exceeds a threshold (default: 1.5)
     - A trough is when the Z-score falls below the negative threshold
   - Secondary method: Also uses `find_peaks` with lower prominence (0.5) to catch local maxima/minima
   - Combined approach ensures both absolute deviations and relative local extremes are detected
   - Code snippet:
     ```python
     df[f'{norm_trend_col}_peak'] = df[norm_trend_col] > threshold
     df[f'{norm_trend_col}_trough'] = df[norm_trend_col] < -threshold
     
     # Additional peaks/troughs detection
     peak_indices, _ = find_peaks(df[norm_trend_col], prominence=0.5)
     ```

### Breakout Analysis

Breakout analysis identifies significant deviations in Google Trends data and analyzes subsequent price movements:

1. **Breakout Detection**
   - Uses Z-score methodology with a higher threshold (default: 2)
   - Identifies both positive breakouts (Z > threshold) and negative breakouts (Z < -threshold)
   - Analyzes price changes following these breakouts over various time windows (1, 3, 6 months)

2. **Success Rate Calculation**
   - For positive breakouts: Percentage of cases where prices increased after the breakout
   - For negative breakouts: Percentage of cases where prices decreased after the breakout
   - Formula: `Success_Rate = (Successful_Cases / Total_Cases) * 100`

### Market Phase Predictive Analysis

The core analysis examines whether Google Trends can predict directional price movements:

1. **Peak → Negative Price Trend Prediction**
   - Measures how often a peak in Google Trends is followed by a significant price drop
   - Success rate: Percentage of peaks that were followed by a ≥10% price decrease
   - Hypothesis: High public interest might signal short-term price tops

2. **Trough → Positive Price Trend Prediction**
   - Measures how often a trough in Google Trends is followed by a significant price rise
   - Success rate: Percentage of troughs that were followed by a ≥10% price increase
   - Hypothesis: Low public interest might signal short-term price bottoms

For each analysis, results are calculated across different time windows (1, 3, and 6 months) to identify the optimal predictive timeframe. While these movements don't necessarily correspond to full market regime changes (i.e., true bull or bear markets), they represent significant price movements that could be valuable for trading strategies.

## Results

The analysis provides insights into:
- Correlations between Google Trends and crypto prices
- Lead-lag relationships (whether trends lead prices or vice versa)
- Success rates of using Google Trends breakouts as trading signals
- The potential of trend peaks/troughs as contrarian indicators for market phases

Detailed output is displayed in the console and visualizations are saved to the `outputs/` directory.

## License

This project is licensed under the MIT License. 