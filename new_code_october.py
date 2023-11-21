import os
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from functools import reduce
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

today = date.today()
end = today.strftime("%Y-%m-%d")
start_1 = (date.today() - timedelta(days=365))
start_2 = start_1.strftime("%Y-%m-%d")

print(start_2)
print(end)

def get_stock(ticker):
    data = yf.download(ticker, start=start_2, end=end)
    data[f'{ticker}'] = data["Close"]
    data = data[[f'{ticker}']]
    return data

def combine_stocks(tickers):
    data_frames = [get_stock(ticker) for ticker in tickers]
    
    # Reset the index before merging
    data_frames = [df.reset_index() for df in data_frames]
    
    # Merge data frames on 'Date'
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='outer'), data_frames)
    
    print(df_merged.head())
    return df_merged

stocks = ["MRNA", "PFE", "JNJ", "GOOGL", 
          "META", "AAPL", "COST", "WMT", "KR", "JPM", 
          "BAC", "HSBC"]
portfolio = combine_stocks(stocks)

# Convert the 'Date' column to datetime format
portfolio['Date'] = pd.to_datetime(portfolio['Date'])

# Calculate the daily returns
returns = portfolio.set_index('Date').pct_change().dropna()

 # Replace with the actual path to your folder
#folder_path = "/Users/dylandienstbier/Desktop/LIFE/Python/Tests" 
# Specify the CSV file name
#csv_file_name = "stocks.csv"

# Join the folder path and file name to create the complete file path
#file_path = os.path.join(folder_path, csv_file_name)

# Save the DataFrame to a CSV file
#portfolio.to_csv(file_path, index=False)

###print(f"CSV file '{csv_file_name}' saved to '{folder_path}'.")

mu = mean_historical_return(portfolio)
S = CovarianceShrinkage(portfolio).ledoit_wolf()

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))
