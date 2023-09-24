import requests
import pandas as pd
from datetime import datetime

# API endpoint URL
base_url = "https://yfapi.net/v8/finance/chart/"

tickers = ["%5ETNX", "SPY", "EURUSD=X"]

substring= "?range=10y&region=US&interval=1d&lang=en"

# Headers with your API key
headers = {
    'X-API-KEY': 'kHNl6Wsad892SYyWmo8pAo5hpSlq9bA5NIXoD8N7'
}
for ticker in tickers:
    url=base_url+ticker+substring
    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract the timezone information from the response
        exchange_timezone = data['chart']['result'][0]['meta']['exchangeTimezoneName']
        # Extract the GMT offset
        gmtoffset = data['chart']['result'][0]['meta']['gmtoffset']

        # Extract the timestamp and adjusted close price data
        timestamps = data['chart']['result'][0]['timestamp']
        prices = data['chart']['result'][0]['indicators']['quote'][0]

        # Create a DataFrame
        df = pd.DataFrame({'Timestamp': timestamps, 'Close': prices['close'], 'Open': prices['open'],
                        'High': prices['high'], 'Low': prices['low'], 'Volume': prices['volume']})
        
        if exchange_timezone == 'America/New_York':
            
            # Convert timestamps to datetime objects in New York time (EDT)
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s', origin='unix', errors='coerce')
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:

            # Convert timestamps to datetime objects using GMT offset
            df['Date'] = pd.to_datetime(df['Timestamp'] - gmtoffset, unit='s', origin='unix', errors='coerce', utc=True)
            df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %H:%M:%S')



        # Set "Date" as the index
        df.set_index('Date', inplace=True)

        # Drop the "Timestamp" column
        df.drop(columns=['Timestamp'], inplace=True)
        # Drop empty bars
        df.dropna(inplace=True)


        # Print the DataFrame
        print(df.head())
        df.to_csv(f"./historical_data/{ticker}_1d.csv")

        # Check if there is volume information
        if 'Volume' in df.columns:
            print("Volume information available.")
        else:
            print("No volume information available.")

    else:
        print("Failed to fetch data. Status code:", response.status_code)

