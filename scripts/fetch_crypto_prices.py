import yfinance as yf
import pandas as pd
from datetime import datetime

def get_historical_prices(symbol, start_date, end_date):
    """
    Fetch historical price data from Yahoo Finance
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        df = ticker.history(start=start_date, end=end_date, interval='1mo')
        
        if df.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Rename columns to match our needs
        df = df[['Open', 'Close', 'High', 'Low']]
        df.columns = ['open', 'close', 'high', 'low']
        
        # Add mean price
        df['mean'] = df[['open', 'close', 'high', 'low']].mean(axis=1)
        
        return df
        
    except Exception as e:
        print(f"Error processing data for {symbol}: {str(e)}")
        return None

def main():
    # Define coins and their Yahoo Finance symbols with their launch dates
    coins = {
        'bitcoin': {'symbol': 'BTC-USD', 'start_date': '2009-01-01'},
        'ethereum': {'symbol': 'ETH-USD', 'start_date': '2015-07-30'},
        'solana': {'symbol': 'SOL-USD', 'start_date': '2020-04-10'}
    }
    
    # End date
    end_date = '2025-03-31'
    
    # Create an empty DataFrame to store all data
    all_data = pd.DataFrame()
    
    # Fetch data for each coin
    for coin_name, coin_info in coins.items():
        print(f"Fetching data for {coin_name} from {coin_info['start_date']}...")
        try:
            data = get_historical_prices(coin_info['symbol'], coin_info['start_date'], end_date)
            if data is not None:
                # Add coin name to columns
                data.columns = [f"{coin_name}_{col}" for col in data.columns]
                all_data = pd.concat([all_data, data], axis=1)
        except Exception as e:
            print(f"Error fetching {coin_name}: {str(e)}")
    
    # Save to CSV only if we have data
    if not all_data.empty:
        all_data.to_csv('crypto_price_data.csv')
        print("Data saved to crypto_price_data.csv")
    else:
        print("No data was successfully fetched")

if __name__ == "__main__":
    main() 