import requests
import pandas as pd
from datetime import datetime

# API endpoint URL
url = "https://yfapi.net/v8/finance/chart/%5ETNX?range=10y&region=US&interval=1d&lang=en"

# Headers with your API key
headers = {
    'X-API-KEY': 'kHNl6Wsad892SYyWmo8pAo5hpSlq9bA5NIXoD8N7'
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract the timestamp and adjusted close price data
    timestamps = data['chart']['result'][0]['timestamp']
    prices = data['chart']['result'][0]['indicators']['quote'][0]

    # Create a DataFrame
    df = pd.DataFrame({'Timestamp': timestamps, 'Close': prices['close'], 'Open': prices['open'],
                       'High': prices['high'], 'Low': prices['low'], 'Volume': prices['volume']})

    # Convert timestamps to datetime objects
    df['Date'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in df['Timestamp']]

    # Convert to New York time (EDT)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

    # Set "Date" as the index
    df.set_index('Date', inplace=True)

    # Drop the "Timestamp" column
    df.drop(columns=['Timestamp'], inplace=True)
    # Drop empty bars
    df.dropna(inplace=True)


    # Print the DataFrame
    print(df.head())
    df.to_csv("./historical_data/TNX_1d.csv")

    # Check if there is volume information
    if 'Volume' in df.columns:
        print("Volume information available.")
    else:
        print("No volume information available.")

else:
    print("Failed to fetch data. Status code:", response.status_code)

