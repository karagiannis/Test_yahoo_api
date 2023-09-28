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

def filter_out_data(start_datetime, stop_datetime, target_df, list_with_indicator_dfs):
    # Filter the target DataFrame
    target_df = target_df[(target_df['timestamp'] >= start_datetime) & (target_df['timestamp'] <= stop_datetime)]
    
    # Filter each indicator DataFrame in the list
    filtered_indicator_dfs = []
    for indicator_df in list_with_indicator_dfs:
        filtered_indicator_df = indicator_df[(indicator_df['timestamp'] >= start_datetime) & (indicator_df['timestamp'] <= stop_datetime)]
        filtered_indicator_dfs.append(filtered_indicator_df)
    
    return target_df, filtered_indicator_dfs


def update_datetimes_so_they_are_aligned(target_df, list_with_indicator_dfs):
    # Get a set of unique timestamps from the target dataframe
    target_timestamps = set(target_df['timestamp'])
    
    # Iterate through the indicator dataframes
    for indicator_df in list_with_indicator_dfs:
        # Remove timestamps not present in target_df
        indicator_df.drop(indicator_df[~indicator_df['timestamp'].isin(target_timestamps)].index, inplace=True)
        
        # Insert missing timestamps
        missing_timestamps = target_timestamps - set(indicator_df['timestamp'])
        for missing_timestamp in missing_timestamps:
            indicator_df.loc[indicator_df.shape[0]] = {'timestamp': missing_timestamp, 'value': None}

    return list_with_indicator_dfs


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
    
    # Convert selected columns to numeric, converting non-numeric values to NaN
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        for column in df.columns:
            if pd.isna(row[column]):
                # Check if there are valid numeric values in adjacent rows
                if index > 0 and index < len(df) - 1:
                    prev_value = df.at[index - 1, column]
                    next_value = df.at[index + 1, column]
                    if not pd.isna(prev_value) and not pd.isna(next_value):
                        # Interpolate the missing value
                        df.at[index, column] = (prev_value + next_value) / 2
    
    return df









# Set the option to display all columns without truncation
pd.set_option('display.max_columns', None)


# Read the data from CSV files into separate DataFrames
df_EURUSD = pd.read_csv('./historical_data/EURUSD_1d.csv')
df_SPY = pd.read_csv('./historical_data/SPY_1d.csv')
df_TNX = pd.read_csv('./historical_data/TNX_1d.csv')




#Create datetime objects
# Convert 'timestamp' column to datetime
df_EURUSD['timestamp'] = pd.to_datetime(df_EURUSD['timestamp'])
df_SPY['timestamp'] = pd.to_datetime(df_SPY['timestamp'])
df_TNX['timestamp'] = pd.to_datetime(df_TNX['timestamp'])

# df_EURUSD.reset_index(drop=True, inplace=True)
# df_EURUSD.set_index('timestamp', inplace=True)


# df_SPY.reset_index(drop=True, inplace=True)
# df_SPY.set_index('timestamp', inplace=True)

# df_TNX.reset_index(drop=True, inplace=True)
# df_TNX.set_index('timestamp', inplace=True)


# df_EURUSD.to_csv("EURUSD_after_reset.csv")
# df_SPY.to_csv("SPY_after_reset.csv")
# df_TNX.to_csv("TNX_after_reset.csv")

#Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

df_EURUSD.to_csv("EURUSD_after_sort.csv")
df_SPY.to_csv("SPY_after_sort.csv")
df_TNX.to_csv("TNX_after_sort.csv")

#Put the indicators in a list
indicators = [df_SPY,df_TNX]

#Find stat_datime and stop_datime
start_datetime, stop_datetime = find_cut_off_datetimes(df_EURUSD, indicators)

print("start_datetime, stop_datetime:", start_datetime, stop_datetime)

#Cut off the head and the tail of the timestamp columns
df_EURUSD, indicators= filter_out_data(start_datetime, stop_datetime, df_EURUSD, indicators)


df_SPY = indicators[0]
df_TNX = indicators[1]
#Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

# df_EURUSD.reset_index(drop=True, inplace=True)
# df_EURUSD.set_index('timestamp', inplace=True)

# df_SPY.reset_index(drop=True, inplace=True)
# df_SPY.set_index('timestamp', inplace=True)

# df_TNX.reset_index(drop=True, inplace=True)
# df_TNX.set_index('timestamp', inplace=True)


df_EURUSD.to_csv("EURUSD_head_tail_filtered.csv")
df_SPY.to_csv("SPY_head-tail_filtered.csv")
df_TNX.to_csv("TNX_head_tail_filtered.csv")

#Insert missing datetimes in the indicators and remove excess timestamps not in the target_df
indicators=update_datetimes_so_they_are_aligned(df_EURUSD, indicators)
#Sort ascending datetimes


# Save the DataFrame to a CSV file
# df_SPY.to_csv('SPY_after_update_datimes.csv', index=False)
# df_TNX.to_csv('TNX__after_update_datimes.csv', index=False)

df_SPY = indicators[0]
df_TNX = indicators[1]

#Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

# df_EURUSD.reset_index(drop=True, inplace=True)
# df_EURUSD.set_index('timestamp', inplace=True)

# df_SPY.reset_index(drop=True, inplace=True)
# df_SPY.set_index('timestamp', inplace=True)

# df_TNX.reset_index(drop=True, inplace=True)
# df_TNX.set_index('timestamp', inplace=True)

# df_SPY.to_csv('SPY_before_clean_nonsense.csv', index=False)
# df_TNX.to_csv('TNX__before_clean_nonsense.csv', index=False)

df_SPY = clean_df_from_nonsense(df_SPY)
df_TNX = clean_df_from_nonsense(df_TNX)

#Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

# df_EURUSD.reset_index(drop=True, inplace=True)
# df_EURUSD.set_index('timestamp', inplace=True)

# df_SPY.reset_index(drop=True, inplace=True)
# df_SPY.set_index('timestamp', inplace=True)

# df_TNX.reset_index(drop=True, inplace=True)
# df_TNX.set_index('timestamp', inplace=True)

df_SPY.to_csv('SPY_after_clean_nonsense.csv', index=False)
df_TNX.to_csv('TNX__after_clean_nonsense.csv', index=False)

df_SPY = interpolate_missing_data(df_SPY)
df_TNX = interpolate_missing_data(df_TNX)

#Sort ascending datetimes
df_EURUSD = df_EURUSD.sort_values(by='timestamp', ascending=True)
df_SPY = df_SPY.sort_values(by='timestamp', ascending=True)
df_TNX = df_TNX.sort_values(by='timestamp', ascending=True)

df_EURUSD.reset_index(drop=True, inplace=True)
df_EURUSD.set_index('timestamp', inplace=True)

df_SPY.reset_index(drop=True, inplace=True)
df_SPY.set_index('timestamp', inplace=True)

df_TNX.reset_index(drop=True, inplace=True)
df_TNX.set_index('timestamp', inplace=True)

# Save the DataFrame to a CSV file
df_EURUSD.to_csv('EURUSD_to_edit.csv')
df_SPY.to_csv('SPY_to_edit.csv')
df_TNX.to_csv('TNX_to_edit.csv')
# Manually edit csv to set the topmost and lowest values as needed

# Read the edited CSV file back into a DataFrame
# df_SPY = pd.read_csv('SPY_to_edit.csv')
# df_TNX = pd.read_csv('TNX_to_edit.csv')




# Create a MinMaxScaler instance for each DataFrame
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()



# Normalize the numeric columns of each DataFrame
df_EURUSD[['open','high','close','low']] = scaler1.fit_transform(df_EURUSD[['open','high','close','low']])
df_SPY[['open','high','close','low','volume']] = scaler2.fit_transform(df_SPY[['open','high','close','low','volume']])
df_TNX[['value']] = scaler3.fit_transform(df_TNX[['value']])

# Save the DataFrame to a CSV file
df_EURUSD.to_csv('EURUSD_after_norma.csv')
df_SPY.to_csv('SPY_after_norma.csv')
df_TNX.to_csv('TNX_after_norma.csv')

print("Length EURUS_df", len(df_EURUSD))
print("Length SPY_df", len(df_SPY))
print("Length TNX_df", len(df_TNX))

# Concatenate DataFrames and reset column names
# merged_df = pd.concat([df_EURUSD, df_SPY, df_TNX], axis=1)
# Concatenate DataFrames with MultiIndex columns
merged_df = pd.concat([df_EURUSD, df_SPY, df_TNX], axis=1, keys=['EURUSD', 'SPY', 'TNX'])

merged_df.to_csv("merged_df.csv")


# # Flatten the column index
# merged_df.columns = ['_'.join(col).strip() for col in merged_df.columns.values]

# # Reset the index
# merged_df.reset_index(inplace=True, drop=True)

# # Save to a CSV file
# merged_df.to_csv("merged_df_reset_index.csv")

# # Rename the column to 'Date' for consistency
# merged_df.columns = merged_df.columns.set_levels(['Date'], level=1)

# # Set 'Date' column as the index
# merged_df.set_index('Date', inplace=True)


# # Display the updated DataFrame
# # print("merged_df:", merged_df.head())
# merged_df.to_csv("merged_df_date.csv")






# Define your input features (X) and target variable (y) for the entire OHLC bar
# Assuming 'merged_df' has a multi-index for columns
X = merged_df.loc[:, pd.IndexSlice[:, ['Open_lag_1', 'High_lag_1', 'Low_lag_1', 'Close_lag_1']]]
# print(X.head())
y = merged_df.loc[:, pd.IndexSlice[:, ['Open', 'High', 'Low', 'Close']]]
# print(y.head())

X.to_csv("X.csv")
y.to_csv("y.csv")

# Define the split ratios (adjust as needed)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print(" y_temp shape:", y_temp.shape)
# print("X_val shape:", X_val.shape)
# print("X_test shape:", X_test.shape)
# print("y_val shape:", y_val.shape)
# print("y_test shape:", y_test.shape)



# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)


# print("Head of X_train:")
# print(X_train.head())
# X_train.to_csv("X_train.csv")

# print("Head of X_test:")
# print(X_test.head())

# print("Head of y_train:")
# print(y_train.head())

# print("Head of y_test:")
# print(y_test.head())

print("X_train head:")
print(X_train.head())
print("y_train head:")
print(y_train.head())
print("y_temp head:")
print(y_temp.head())
print("X_val head:")
print(X_val.head())
print("X_test head:")
print(X_test.head())
print("y_val head:")
print(y_val.head())
print("y_test head:")
print(y_test.head())



# Create and compile the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(4))  # Output layer with 4 units for OHLC prediction
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# Reshape data for LSTM input
X_train = X_train.values
X_test = X_test.values
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
print("X_train head:")
print(X_train[:5])
print("X_test head:")
print(X_test[:5])

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

print("X_train head:")
print(X_train[:5])
print("X_test head:")
print(X_test[:5])

# print("X_train:")
# print(X_train[:5])  
# print("")
# print("Length of X_train:", len(X_train))

# print("X_test:")
# print(X_test[:5])  
# print("")
# print("Length of X_test:", len(X_test))

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Initialize a list to store predicted bars
predicted_bars = []

# Create a MinMaxScaler instance for your target variable (y)
scaler = MinMaxScaler()

# Fit the scaler on your training target variable (y_train)
scaler.fit(y_train)

# Make predictions on the test set
for i in range(len(X_test)):
    X_sample = X_test[i].reshape(1, X_test.shape[1], 1)
    y_pred = model.predict(X_sample)
    print("y_pred shape:", y_pred.shape)


    # Reshape y_pred to match the shape of your target variable (OHLC)
    y_pred = y_pred.reshape(1, 4)  # Assuming you are predicting OHLC
    print("y_pred shape:", y_pred.shape)


    # Inverse transform the predictions
    y_pred = scaler.inverse_transform(y_pred)
    print("y_pred shape:", y_pred.shape)

    
    predicted_bars.append(y_pred[0])


# Convert the list of predicted bars to a NumPy array
predicted_bars = np.array(predicted_bars)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predicted_bars))
print(f'Root Mean Squared Error: {rmse}')

# Prepare data for candlestick chart
ohlc_actual = y_test
ohlc_predicted = predicted_bars

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

