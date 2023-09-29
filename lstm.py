import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
from scipy.interpolate import interp1d

import pandas as pd
import numpy as np


def sort_ascending_datetime(df):
    #Leftmost column has the name "timestamp" or "Date"
    return df.sort_values("timestamp", ascending=True )

def find_cut_off_datetimes(target_df, list_with_indicator_dfs):
    # Start datetime is determined by the target DataFrame (EURUSD)
    start_datetime = target_df['timestamp'].min()
    
    # Initialize stop datetime with a very late date
    stop_datetime = pd.Timestamp('9999-12-31')
    
    # Iterate through indicator DataFrames to find the earliest stop date
    for df in [target_df] + list_with_indicator_dfs:
        stop_date = df['timestamp'].max()
        if stop_date < stop_datetime:
            stop_datetime = stop_date
    
    return start_datetime, stop_datetime

def cut_off_head_and_tail_of_data(start_datetime, stop_datetime, target_df, list_with_indicator_dfs):
    # Filter the target DataFrame
    target_df = target_df[(target_df['timestamp'] >= start_datetime) & (target_df['timestamp'] <= stop_datetime)]
    
    # Filter each indicator DataFrame in the list
    filtered_indicator_dfs = []
    for indicator_df in list_with_indicator_dfs:
        filtered_indicator_df = indicator_df[(indicator_df['timestamp'] >= start_datetime) & (indicator_df['timestamp'] <= stop_datetime)]
        filtered_indicator_dfs.append(filtered_indicator_df)
    
    return target_df, filtered_indicator_dfs


def update_datetimes_so_they_are_aligned(target_df, indicator_df):
    # Get a set of unique timestamps from the target dataframe
    target_timestamps = set(target_df['timestamp'])
    print("Length of target:", len(target_timestamps))



    # Create a copy of the indicator DataFrame
    modified_indicator_df = indicator_df.copy()

    # Remove timestamps not present in the target dataframe from modified_indicator_df
    #   Create a boolean Series indicating whether each timestamp is in target_timestamps
    is_in_target = modified_indicator_df['timestamp'].isin(target_timestamps)

    #   Create a DataFrame containing only the rows where timestamps are NOT in target_timestamps
    rows_to_drop = modified_indicator_df[~is_in_target]

    #   Drop the rows from modified_indicator_df using the index of rows_to_drop
    modified_indicator_df.drop(rows_to_drop.index, inplace=True)

    print("Length of indicator before adding missing datetimes:",len( modified_indicator_df) )

    # Extract the timestamps from the indicator DataFrame
    indicator_timestamps = set(modified_indicator_df['timestamp'])

    # Find missing timestamps
    missing_timestamps = target_timestamps - indicator_timestamps
    print("Missing timestamps:", missing_timestamps)
    for missing_timestamp in missing_timestamps:
        if missing_timestamp in indicator_timestamps:
            print("Duplicate timestamp found:", missing_timestamp)

    # Create a prototype empty dictionary based on the keys of the first element
    prototype_dict = modified_indicator_df.iloc[0].copy().to_dict()
    for key in prototype_dict.keys():
        prototype_dict[key] = None

    # Create records for missing timestamps and append them to the indicator DataFrame
    missing_records = []
    for missing_timestamp in missing_timestamps:
        new_record = prototype_dict.copy()
        new_record['timestamp'] = datetime(missing_timestamp.year, missing_timestamp.month, missing_timestamp.day)
        missing_records.append(new_record)

    # Append the missing records to the indicator DataFrame
    if missing_records:
        print("Length of missing records:", len(missing_records))
        # Create a DataFrame from the list of dictionaries (missing_records)
        missing_records_df = pd.DataFrame(missing_records)

        # Concatenate the missing_records_df with the modified_indicator_df
        modified_indicator_df = pd.concat([modified_indicator_df, missing_records_df], ignore_index=True)

    modified_indicator_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)

    header = modified_indicator_df.columns.tolist()
    print("Header:", header)

    if len(modified_indicator_df.axes[0]) != len(target_timestamps):
        raise ValueError("Length of indicator is not equal to the target")
    
    # Sort the DataFrame by timestamps in ascending order and reset the index
    modified_indicator_df = modified_indicator_df.sort_values(by='timestamp').reset_index(drop=True)

    return modified_indicator_df








import pandas as pd
import numpy as np

def clean_df_from_nonsense(df):
    for index, row in df.iterrows():
        for column in df.columns:
            # Skip the first column (assuming it contains datetime values)
            if column == 'timestamp':
                continue
            cell_value = row[column]
            if pd.isna(cell_value) or cell_value == ".":
                    # If the value is NaN or ".", replace it with None
                    df.at[index, column] = None
    return df


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def interpolate_missing_data(df):
    # Define a list of columns to convert to numeric, excluding the first column ('timestamp')
    columns_to_convert = [col for col in df.columns if col != 'timestamp']
    print("Columns to convert:", columns_to_convert)
    
    # Convert selected columns to numeric, converting non-numeric values to NaN
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        for column in columns_to_convert:
            if pd.isna(row[column]):
                # Check if there are valid numeric values in adjacent rows
                if index > 0 and index < len(df) - 1:
                    prev_value = df.at[index - 1, column]
                    next_value = df.at[index + 1, column]
                    if not pd.isna(prev_value) and not pd.isna(next_value):
                        # Interpolate the missing value
                        df.at[index, column] = (prev_value + next_value) / 2
    
    return df

def interpolate_missing_data2(df):
    # Fill missing values using linear interpolation
    df.interpolate(method='linear', inplace=True)
    return df










# Set the option to display all columns without truncation
pd.set_option('display.max_columns', None)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the data from CSV files into separate DataFrames
df_EURUSD = pd.read_csv('./historical_data/EURUSD_1d.csv')
df_SPY = pd.read_csv('./historical_data/SPY_1d.csv')
df_TNX = pd.read_csv('./historical_data/TNX_1d.csv')

# Convert 'timestamp' column to datetime
df_EURUSD['timestamp'] = pd.to_datetime(df_EURUSD['timestamp'])
df_SPY['timestamp'] = pd.to_datetime(df_SPY['timestamp'])
df_TNX['timestamp'] = pd.to_datetime(df_TNX['timestamp'])

# Drop duplicate rows with respect to the 'timestamp' column
df_EURUSD.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
df_SPY.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
df_TNX.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)

# Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

#Reset the index of df_EURUSD
df_EURUSD.reset_index(drop=True, inplace=True)

# Put the indicators in a list
indicators = [df_SPY, df_TNX]

# Find start_datetime and stop_datetime
start_datetime, stop_datetime = find_cut_off_datetimes(df_EURUSD, indicators)

print("start_datetime, stop_datetime:", start_datetime, stop_datetime)

# Cut off the head and the tail of the timestamp columns
df_EURUSD, indicators = cut_off_head_and_tail_of_data(start_datetime, stop_datetime, df_EURUSD, indicators)
df_SPY = indicators[0]
df_TNX = indicators[1]


# Insert missing datetimes in the indicators and remove excess timestamps not in the target_df
df_SPY = update_datetimes_so_they_are_aligned(df_EURUSD, df_SPY)
df_TNX = update_datetimes_so_they_are_aligned(df_EURUSD, df_TNX)

# Clean the indicators from nonsense values
df_SPY = clean_df_from_nonsense(df_SPY)
df_TNX = clean_df_from_nonsense(df_TNX)


# Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

# Interpolate missing data in the indicators
df_SPY = interpolate_missing_data(df_SPY)
df_TNX = interpolate_missing_data(df_TNX)

# Call the function for the "SPY" DataFrame
df_SPY = interpolate_missing_data2(df_SPY)
df_TNX = interpolate_missing_data2(df_TNX)

# Reset index
df_EURUSD.reset_index(drop=True, inplace=True)
df_SPY.reset_index(drop=True, inplace=True)
df_TNX.reset_index(drop=True, inplace=True)

# Save the DataFrame to CSV files
df_EURUSD.to_csv('EURUSD_to_edit.csv')
df_SPY.to_csv('SPY_to_edit.csv')
df_TNX.to_csv('TNX_to_edit.csv')

# Create MinMaxScaler instances for each DataFrame
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()

# Normalize the numeric columns of each DataFrame
df_EURUSD[['open', 'high', 'close', 'low']] = scaler1.fit_transform(df_EURUSD[['open', 'high', 'close', 'low']])
df_SPY[['open', 'high', 'close', 'low', 'volume']] = scaler2.fit_transform(df_SPY[['open', 'high', 'close', 'low', 'volume']])
df_TNX[['value']] = scaler3.fit_transform(df_TNX[['value']])

# Save the DataFrame to CSV files after normalization
df_EURUSD.to_csv('EURUSD_after_norm.csv')
df_SPY.to_csv('SPY_after_norm.csv')
df_TNX.to_csv('TNX_after_norm.csv')

print("Length EURUSD:", len(df_EURUSD))
print("Length SPY:", len(df_SPY))
print("Length TNX:", len(df_TNX))


# Concatenate DataFrames and reset column names
# merged_df = pd.concat([df_EURUSD, df_SPY, df_TNX], axis=1)
# Concatenate DataFrames with MultiIndex columns
X = pd.concat([df_EURUSD, df_SPY, df_TNX], axis=1, keys=['EURUSD', 'SPY', 'TNX'])

X.to_csv("merged_df.csv")


# Flatten the column index
X.columns = ['_'.join(col).strip() for col in X.columns.values]

# Reset the index
X.reset_index(inplace=True, drop=True)

# Save to a CSV file
X.to_csv("merged_df_reset_index.csv")

# Set 'timestamp' column as the index
X.set_index('EURUSD_timestamp', inplace=True)
X = X.rename_axis('timestamp')
# Drop the redundant timestamp columns
X.drop(columns=['SPY_timestamp', 'TNX_timestamp'], inplace=True)

X.to_csv("merged_df_final.csv")



# Define your target variable (y),  in df_EURUSD
y = df_EURUSD[['timestamp','open','high','low','close']]
# Rename columns in y DataFrame to match X DataFrame
y.rename(columns={'open': 'EURUSD_open', 'high': 'EURUSD_high', 'low': 'EURUSD_low', 'close': 'EURUSD_close'}, inplace=True)
y.set_index('timestamp',inplace=True)
y.to_csv("y.csv")

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the lengths of each split
total_length = len(X)
train_length = int(train_ratio * total_length)
val_length = int(val_ratio * total_length)

# Split the data
X_train = X[:train_length]
y_train = y[:train_length]
X_val = X[train_length:train_length + val_length]
y_val = y[train_length:train_length + val_length]
X_test = X[train_length + val_length:]
y_test = y[train_length + val_length:]


# Create and compile the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(4))  # Output layer with 4 units for OHLC prediction
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# Reshape data for LSTM input
X_train = X_train.values
X_test = X_test.values
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Check data types
print("X_train data type:", X_train.dtype)
print("Data Types of Columns in y_train:")
print(y_train.dtypes)

# # Convert data types to float32 if needed
# X_train = X_train.astype('float32')
# y_train = y_train.astype('float32')

# Check for NaN or infinity values
if np.isnan(X_train).any().any() or np.isnan(y_train).any().any() or np.isinf(X_train).any().any() or np.isinf(y_train).any().any():
    print("Data contains NaN or infinity values. Please preprocess your data to handle these issues.")

# Check data shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Initialize a list to store predicted bars
predicted_bars = []

# # Create a MinMaxScaler instance for your target variable (y)
# scaler = MinMaxScaler()

# Fit the scaler on your training target variable (y_train)
scaler1.fit(y_train)

# Make predictions on the test set
for i in range(len(X_test)):
    X_sample = X_test[i].reshape(1, X_test.shape[1], 1)
    y_pred = model.predict(X_sample)
    print("y_pred shape:", y_pred.shape)


    # Reshape y_pred to match the shape of your target variable (OHLC)
    y_pred = y_pred.reshape(1, 4)  # Assuming you are predicting OHLC with the datetimes
    print("y_pred shape:", y_pred.shape)


    # Inverse transform the predictions
    # Inverse transform the predictions for OHLC values (columns 1 to 4)
    #y_pred[:, 1:5] = scaler1.inverse_transform(y_pred[:, 1:5])
    y_pred = scaler1.inverse_transform(y_pred)

    print("y_pred shape:", y_pred.shape)

    
    predicted_bars.append(y_pred[0])


# Convert the list of predicted bars to a NumPy array
# Assuming predicted_bars has columns in this order: timestamp, open, high, low, close
#predicted_bars_df = pd.DataFrame(predicted_bars, columns=['timestamp', 'open', 'high', 'low', 'close'])
predicted_bars_df = pd.DataFrame(predicted_bars, columns=['open', 'high', 'low', 'close'])


# Calculate RMSE
# Convert the list of predicted bars to a NumPy array
rmse = np.sqrt(mean_squared_error(y_test, predicted_bars_df))
print(f'Root Mean Squared Error: {rmse}')

# Prepare data for candlestick chart
y_test[:, 1:5] = scaler1.inverse_transform(y_test[:, 1:5])
ohlc_actual = y_test
ohlc_actual.rename(columns={
    'EURUSD_open': 'open',
    'EURUSD_high': 'high',
    'EURUSD_low': 'low',
    'EURUSD_close': 'close'
}, inplace=True)
ohlc_predicted = predicted_bars

print("Data type of y_test:", y_test.dtypes)
print("Data type of predicted_bars:", predicted_bars.dtype)

y_test.to_csv("y_test:csv")
predicted_bars_df.to_csv("predicted_bars.csv")



# Create candlestick chart for actual and predicted OHLC bars
fig, ax = plt.subplots(figsize=(12, 6))

candlestick_ohlc(ax, ohlc_actual, width=0.4, colorup='g', colordown='r', alpha=0.8)
candlestick_ohlc(ax, ohlc_predicted, width=0.2, colorup='b', colordown='b', alpha=0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.title('Actual vs. Predicted OHLC Bars')
plt.xlabel('Date')
plt.ylabel('OHLC Price')
plt.legend(['Actual', 'Predicted'])

plt.show()

