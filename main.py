# Title: The Right Way to Trade Volatility
# Name: Bani Bedi
# Instructor: Lukas Hager
# Course: Econ 481

# -------------------------------------------------------------------------------------------------------

# Instructions:
# Step 1: Kindly download file into you Google Drive or Computer's Drive (both are included for your ease)
# Step 2: Amend file_start line based on where you have stored the downloaded file
# Step 3: Run the entire code
# Step 4: Enjoy!

# Note: This project was originally created on Google Collab

# -------------------------------------------------------------------------------------------------------

# Link to Github 
# https://github.com/banibedi/Econ481.git

# -------------------------------------------------------------------------------------------------------

# Read Data from Drive (Remove if you are not using Google Drive / Colab to run code)
from google.colab import drive
drive.mount('/content/drive')

! pip install ta

# Import Libraries
import pandas as pd
import numpy as np
import openpyxl
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
import requests
import ta
from ta.trend import ADXIndicator
from datetime import date
from dateutil.relativedelta import relativedelta
import os
from dateutil.relativedelta import relativedelta

# Where the Files are Saved (Note: Kindly amend this to where you have saved my downloaded files, Lukas)
file_start = r'C:\\Users\\bbedi\\Documents\\' # If the files are being pulled from my C Drive
file_start = r'/content/drive/MyDrive/Project/' # If the files are being pulled from my Google Drive

# **************************************** Data Analysis Graphs ****************************************

# Download File with High, Low, and Previous Day's Closing Price for EURUSD
file_path = file_start + r'EURUSD Spot Data High Low.xlsx'
fx_spot_data_raw = pd.read_excel(file_path)

# Create Closing Price Column
fx_spot_data_raw['PX_CLOSE'] = fx_spot_data_raw['PX_CLOSE_1D'].shift(-1)

# Daily Percentage Change
fx_spot_data_raw['Pct_Change'] = fx_spot_data_raw['PX_CLOSE'].pct_change() * 100

# Drop Any NaN Values
fx_spot_data_raw = fx_spot_data_raw.dropna(subset=['Pct_Change'])

# Plot the Percentage Change
plt.figure(figsize=(12, 6))
plt.plot(fx_spot_data_raw['Date'], fx_spot_data_raw['Pct_Change'], label='Daily % Change')
plt.xlabel('Date')
plt.ylabel('Percentage Change (%)')
plt.title('Daily Percentage Change of EURUSD')
plt.legend()
plt.grid(True)
plt.xlim(pd.Timestamp('2013-01-01'), pd.Timestamp('2023-12-31'))
plt.show()

# Calculate Rolling Standard Deviation
window = 14
fx_spot_data_raw['Rolling_Std'] = fx_spot_data_raw['Pct_Change'].rolling(window=window).std()

# Drop NaN Values
fx_spot_data_raw = fx_spot_data_raw.dropna(subset=['Rolling_Std'])

# Plot Rolling Standard Deviation
plt.figure(figsize=(12, 6))
plt.plot(fx_spot_data_raw['Date'], fx_spot_data_raw['Rolling_Std'], label='Rolling Standard Deviation (%)', color='orange')
plt.xlabel('Date')
plt.ylabel('Standard Deviation (%)')
plt.title('Rolling Standard Deviation of Daily Percentage Change of EURUSD')
plt.legend()
plt.grid(True)
plt.xlim(pd.Timestamp('2013-01-01'), pd.Timestamp('2023-12-31'))
plt.show()

# **************************************** Calculating Average True Range (ATR) ****************************************

# Calculate ATR
def calculate_atr(fx_spot_data_raw, period=14):
    fx_spot_data_raw['H-L'] = fx_spot_data_raw['PX_HIGH'] - fx_spot_data_raw['PX_LOW'] # High Price - Low Price
    fx_spot_data_raw['H-Cp'] = abs(fx_spot_data_raw['PX_HIGH'] - fx_spot_data_raw['PX_CLOSE_1D']) # High Price - Previous Day's Closing Price
    fx_spot_data_raw['L-Cp'] = abs(fx_spot_data_raw['PX_LOW'] - fx_spot_data_raw['PX_CLOSE_1D']) # Low Price - Previous Day's Closing Price
    fx_spot_data_raw['TR'] = fx_spot_data_raw[['H-L', 'H-Cp', 'L-Cp']].max(axis=1) # True Range Formula (Needed to Calculate ATR)
    fx_spot_data_raw['ATR'] = fx_spot_data_raw['TR'].rolling(window=period).mean() # Calculating ATR
    return fx_spot_data_raw

data_with_atr = calculate_atr(fx_spot_data_raw)

# Differentiate High vs. Low Volatility Based on ATR (Standard-Deviation-Based)
mean_atr = data_with_atr['ATR'].mean() # Calculating Mean of ATR Data
std_atr = data_with_atr['ATR'].std() # Calculating Standard Deviation of ATR Data
threshold = mean_atr + 2 * std_atr # Calculating Threshold (Mean + 2 Standard Deviations Away From Mean)

# Define Date Range for "Cannot be Calculated"
cannot_be_calculated_start = pd.Timestamp('2012-12-01') # Defining Unnecessary Dates (Start)
cannot_be_calculated_end = pd.Timestamp('2012-12-31') # Defining Unnecessary Dates (End)

# Label 'High Volatility' and 'Low Volatility' Based on Standard Deviation Threshold
data_with_atr['Volatility Analysis with Standard Deviation'] = data_with_atr.apply(
    lambda row: (
        'High Volatility' if row['ATR'] >= threshold else (
            'ATR Cannot Be Calculated' if cannot_be_calculated_start <= row['Date'] <= cannot_be_calculated_end else 'Low Volatility'
        )
    ),
    axis=1
)

# Differentiate High vs. Low Volatility Based on ATR (Percentage-Based)
data_with_atr['Previous ATR'] = data_with_atr['ATR'].shift(1)

data_with_atr['Volatility Analysis with Percentage'] = data_with_atr.apply(
    lambda row: (
        'High Volatility' if row['ATR'] > row['Previous ATR'] * 1.05 else ( # If previous ATR is 5% or more than current ATR, it will be classfied as a "High Volatility" day
            'ATR Cannot Be Calculated' if cannot_be_calculated_start <= row['Date'] <= cannot_be_calculated_end else 'Low Volatility'
        )
    ),
    axis=1
)

# Drop 'Previous ATR' Column
data_with_atr.drop(columns=['Previous ATR'], inplace=True)

# Save Result to CSV File
output_path = file_start + r'EURUSD_Spot_Data_with_ATR.csv'
data_with_atr.to_csv(output_path, index=False)

print(f"Data with ATR and Volatility Analysis has been saved to {output_path}")

# **************************************** Black Swan Events ****************************************

# Define Black Swan Events with Labels
black_swan_events = { # Any Event that Could Not Be Reasonably Predicted
    '2013-03-16': 'Cyprus Financial Crash',
    '2014-02-27': 'Russia Annexes Crimea',
    '2015-01-15': 'Swiss Franc Unpegging',
    '2015-06-12': 'Chinese Stock Market Crash',
    '2016-06-23': 'Brexit',
    '2016-11-08': 'Donald Trump Unexpectedly Elected',
    '2017-09-03': 'North Korea Missile Tests',
    '2018-02-05': 'Volatility Surge (VIX Spike)',
    '2018-03-01': 'US-China Trade War',
    '2019-06-09': 'Hong Kong Protests',
    '2019-08-11': 'Argentine Debt Crisis',
    '2020-03-11': 'Outbreak of Covid',
    '2020-03-09': 'Oil Price War',
    '2020-04-20': 'Negative Oil Prices',
    '2021-01-06': 'US Capitol Riot',
    '2021-01-28': 'GameStop Short Squeeze',
    '2021-03-23': 'Suez Canal Blockage',
    '2022-02-24': 'Russian Ukrainian War',
    '2023-03-10': 'SVB Collapse'
}

# Convert Black Swan Dictionary to a DataFrame
black_swan_df = pd.DataFrame(list(black_swan_events.items()), columns=['Date', 'Black Swan Event'])
black_swan_df['Date'] = pd.to_datetime(black_swan_df['Date'])

# Merge Black Swan Events with ATR Data
data_with_atr['Date'] = pd.to_datetime(data_with_atr['Date'])
data_with_atr = data_with_atr.merge(black_swan_df, on='Date', how='left')

# Label Black Swan events as "High Volatility", "Low Volatility", or Other
data_with_atr['Volatility Analysis with Black Swan'] = data_with_atr.apply(
    lambda row: ('High Volatility' if pd.notna(row['Black Swan Event']) else (
            'ATR Cannot Be Calculated' if cannot_be_calculated_start <= row['Date'] <= cannot_be_calculated_end else 'Low Volatility'
        )
    ),
    axis=1
)

# Save to CSV File
output_path = file_start + r'EURUSD_Spot_Data_with_ATR_and_Black_Swan_Events.csv'
data_with_atr.to_csv(output_path, index=False)

print(f"Data with ATR and Volatility Analysis with Black Swan Events has been saved to {output_path}")

# **************************************** Economic Data ****************************************

# Merge Economic Data
df = []

country_region_pairs = {
    'Eurozone': 'Eurozone',
    'United States': 'United States'
}

for country, region in country_region_pairs.items():
    for t in ['Central Banks', 'Economic Events', 'Economic Releases']:
        for year in ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']:
            df.append(pd.read_excel(file_start + f'{country}/{year} - {region} - {t}.xlsx'))

# Concatenate all DataFrames into One
merged_df = pd.concat(df, ignore_index=True)

# Convert 'Date Time' Column to Datetime
merged_df['Date Time'] = pd.to_datetime(merged_df['Date Time'], errors='coerce')
merged_df = merged_df.dropna(subset=['Date Time'])

# Sort by 'Date Time' Column
merged_df = merged_df.sort_values(by='Date Time')
merged_df['Date'] = merged_df['Date Time'].dt.date
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Drop 'Date Time' Column
merged_df.drop('Date Time', axis=1, inplace=True)

# Reset the Index
merged_df.reset_index(drop=True, inplace=True)

# Save the Merged DataFrame to a New Excel File
output_path = file_start + r'Merged_Data.xlsx'
merged_df.to_excel(output_path, index=False)

print(f"Data has been merged and saved to {output_path}")

# Reset Index
merged_df.reset_index(drop=True, inplace=True)

# Rename "Unnamed: 1" to "Country"
merged_df = merged_df.rename(columns={"Unnamed: 1": "Country"})

# Delete Unnecessary Columns
merged_df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 5"], axis=1, inplace=True)

from re import sub

# Filter the Dataframe
def filter_df(data_with_atr, high_vol_events, column_name, file_name):

    data_with_atr.drop(columns=[f'Event {column_name}'], inplace=True, errors='ignore')
    high_vol_events = high_vol_events[['Date', 'Event']]
    high_vol_events = high_vol_events.rename(columns={'Event': f'Event {column_name}'})
    high_vol_events = high_vol_events.drop_duplicates(subset=['Date'])

    print(high_vol_events)

    # Merge High Vol Data into ATR Data
    data_with_atr = data_with_atr.merge(
        high_vol_events,
        how = 'left',
        on = 'Date',
    )

    # Add New Column 'Volatility Analysis with Initial Jobless Claims'
    data_with_atr[column_name] = data_with_atr.apply(
        lambda row: 'High Volatility' if pd.notna(row[f'Event {column_name}']) else (
            'Cannot be Calculated' if cannot_be_calculated_start <= row['Date'] <= cannot_be_calculated_end else 'Low Volatility'
        ),
        axis=1
    )

    # Save Result to a CSV File
    output_path = file_start + file_name
    data_with_atr.to_csv(output_path, index=False)

    print(f"Data with ATR and {column_name} has been saved to {output_path}")
    return data_with_atr

# **************************************** Economic Data: Interest Rate Decisions ****************************************

# Filter for High Volatility Events (Interest Rate Decisions)
high_vol_events = merged_df[merged_df['Event'].isin(["FOMC Rate Decision", "ECB Deposit Facility Rate"])] # Search for these events in merged dataframe
column_name = 'Volatility Analysis with IR Decisions'
file_name = r'EURUSD_Spot_Data_with_ATR_Black_Swan_and_IR_Decisions.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: Nonfarm Payrolls ****************************************

# Filter for High Volatility Events (Non-Farm Payrolls)
high_vol_events = merged_df[merged_df['Event'].isin(["Change in Nonfarm Payrolls"]) # Search for these events in merged dataframe
column_name = 'Volatility Analysis with Nonfarm Payrolls'
file_name = r'EURUSD_Spot_Data_with_All_and_Nonfarm_Payrolls.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: US Initial Jobless Claims ****************************************

# Filter for High Volatility Events (Initial Jobless Claims)
high_vol_events = merged_df[merged_df['Event'].isin(["Initial Jobless Claims"])] # Search for these events in merged dataframe
column_name = 'Volatility Analysis with Initial Jobless Claims'
file_name = r'EURUSD_Spot_Data_with_All_and_Initial_Jobless.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: US Continuing Claims ****************************************

# Filter for High Volatility Events (Continuing Claims)
high_vol_events = merged_df[merged_df['Event'].isin(["Continuing Claims"])] # Search for these events in merged dataframe
column_name = 'Volatility Analysis with Continuing Claims'
file_name = r'EURUSD_Spot_Data_with_All_and_Continuing_Claims.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: US Unemployment Rate ****************************************

# Filter for High Volatility Events (Unemployment Rate)
high_vol_events = merged_df[(merged_df['Event'] == "Unemployment Rate") & (merged_df['Country'] == "US")] # Search for these events in merged dataframe for both Eurozone and US
column_name = 'Volatility Analysis with Unemployment Rate'
file_name = r'EURUSD_Spot_Data_with_All_and_Unemployment_Rate.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: Eurozone CPI Data ****************************************

# Filter for High Volatility Events (CPI YoY and MoM for Specified Countries)
high_vol_events = merged_df[
    (merged_df['Event'].isin(["CPI YoY", "CPI Core YoY", "CPI MoM", "CPI Core MoM"])) & # Search for these events in merged dataframe
    (merged_df['Country'].isin(["EC", "FR", "GE", "IT"])) # Only these countries
]
column_name = 'Volatility Analysis with Eurozone CPI'
file_name = r'EURUSD_Spot_Data_with_All_and_EU_CPI.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: Eurozone PPI Data ****************************************

# Filter for High Volatility Events (PPI YoY and MoM for specified countries)
high_vol_events = merged_df[
    (merged_df['Event'].isin(["PPI YoY", "PPI MoM"])) & # Search for these events in merged dataframe
    (merged_df['Country'].isin(["EC", "FR", "GE", "IT"])) # Only these countries in Europe as they are the biggest economies
]
column_name = 'Volatility Analysis with Eurozone PPI'
file_name = r'EURUSD_Spot_Data_with_All_and_EU_PPI.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: United States CPI Data ****************************************

# Filter for High Volatility Events (CPI YoY and MoM for specified countries)
high_vol_events = merged_df[
    (merged_df['Event'].isin(["CPI YoY", "CPI Core YoY", "CPI MoM", "CPI Core MoM"])) & # Search for these events in merged dataframe
    (merged_df['Country'].isin(["US"])) # Only in the US
]
column_name = 'Volatility Analysis with US CPI'
file_name = r'EURUSD_Spot_Data_with_All_and_US_CPI.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Economic Data: United States PPI Data ****************************************

# Filter for High Volatility Events (PPI YoY and MoM for specified countries)
high_vol_events = merged_df[
    (merged_df['Event'].isin(["PPI YoY", "PPI MoM"])) & # Search for these events in merged dataframe
    (merged_df['Country'].isin(["US"])) # Only in teh US
]
column_name = 'Volatility Analysis with US PPI'
file_name = r'EURUSD_Spot_Data_with_Everything.csv'
data_with_atr = filter_df(data_with_atr, high_vol_events, column_name, file_name)

# **************************************** Downloading FX Spot Data ****************************************

# Note: Given the high volume of data, this may take a while (up to 15 minutes)

# Set API Key and Additional Information
API_KEY = 'CshoD8tRaO3QwDO56mLnq00kMNsx44Zy'
START_DATE = '2013-01-01'
END_DATE = '2023-12-31'

# Retrieve the FX Data from Polygon
def fetch_fx_data(start_date, end_date, interval):
    url = f'https://api.polygon.io/v2/aggs/ticker/C:EURUSD/range/5/minute/2023-01-01/2023-01-31'
    url = f'https://api.polygon.io/v2/aggs/ticker/C:EURUSD/range/{interval}/minute/{start_date}/{end_date}'
    print(url)
    params = {
        'adjust': 'true',
        'sort': 'asc',
        'limit': '50000',
        'apiKey': API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()

# Retrieve Specific Data
def json_to_dataframe(json_data, interval):
    time_series = json_data.get('results', {})
    fx_spot_data = pd.DataFrame(time_series, columns=['t', 'o', 'h', 'l', 'c'])
    fx_spot_data = fx_spot_data.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close"}) # Only Needing Time, Open, High, Low, and Close Prices
    fx_spot_data.time = fx_spot_data.time*1000000
    fx_spot_data.time = pd.to_datetime(fx_spot_data.time)
    fx_spot_data.set_index('time', inplace = True)
    return fx_spot_data

# Create the Trading Signal
def process_fx_data(fx_data):
    fx_data['next_close'] = fx_data['close'].shift(-1)
    fx_data.dropna(inplace=True)

    fx_data['Signal'] = fx_data.apply(lambda row: 'Long' if row['next_close'] > row['close'] else 'Short', axis=1) # If the closing price of the previous day is greater than the current day's closing price, then short EURUSD, else go long the pair
    fx_data['Pct_Change'] = fx_data.apply(lambda row: (row['next_close'] - row['close']) / row['close'] * 100, axis=1) # Calculate the percentage change (difference)
    return fx_data

# Retrieve the Data
def get_fx_data(intervals):
    data_frames = {}
    for interval in intervals:
        for month in range(132):
            start_date = date(2013,1,1)+relativedelta(months=month)
            end_date = start_date+relativedelta(months=1)+relativedelta(days=-1)

            json_data = fetch_fx_data(start_date, end_date, interval)
            fx_spot_data = json_to_dataframe(json_data, interval)
            fx_spot_data = process_fx_data(fx_spot_data)

            if interval in data_frames:
                data_frames[interval] = pd.concat([data_frames[interval], fx_spot_data])
            else:
                data_frames[interval] = fx_spot_data

    return data_frames

# List of Intervals to Fetch Data
intraday_intervals = ['5', '15', '30', '60'] # Minute data (i.e. 5 = 5 minute interval data)

# Get Intraday Data
intraday_data_frames = get_fx_data(intraday_intervals)
print(intraday_data_frames)

# Save to Excel File with Multiple Sheets
output_path = file_start + r'EURUSD_Spot_Data_with_Signals.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for interval, fx_spot_data in intraday_data_frames.items():
        fx_spot_data.to_excel(writer, sheet_name=f'{interval}_data')

print(f"Data saved to {output_path}")

# **************************************** DMI and ADX Calculation ****************************************

# Function to Calculate DMI and ADX
def calculate_dmi_adx(df, period=14):
    df['DMI+'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=period) # Calculate DMI+ using TA library
    df['DMI-'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=period) # Calculate DMI- using TA library

    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
    df['ADX'] = adx_indicator.adx()
    df['ADX_prev'] = df['ADX'].shift(1)
    df = df.dropna(subset=['ADX_prev'])

    df['DMI_diff_abs'] = (df['DMI+'] - df['DMI-']).abs() # Calculate the difference between DMI+ and DMI- in absolute value terms

    # Determine Long or Short
    df['DMI Signal'] = df.apply(lambda row: 'Long' if row['DMI+'] > row['DMI-'] else 'Short', axis=1)

    # Determine Trend Strength
    df['Trend Strength'] = df.apply(lambda row: 'Strong Trend' if row['ADX'] > 20 and row['ADX'] > row['ADX_prev'] else (
                           'Weakening Trend' if row['ADX'] > 20 and row['ADX'] < row['ADX_prev'] else (
                           'Weak Trend' if row['ADX'] < 20 and row['ADX'] < row['ADX_prev'] else (
                           'Strengthening Trend' if row['ADX'] < 20 and row['ADX'] > row['ADX_prev'] else 'Undefined'))), axis=1)

    return df

# Calculate Percentage Changes
def calculate_percentage_changes(df):
    df['Pct_Change_DMI+'] = df['DMI+'].pct_change() * 100
    df['Pct_Change_DMI-'] = df['DMI-'].pct_change() * 100
    df['Pct_Change_Spot'] = df['close'].pct_change() * 100
    df = df.dropna(subset=['Pct_Change_DMI+', 'Pct_Change_DMI-', 'Pct_Change_Spot'])
    return df

# Note: first 13 values will be NA because DMI requires 14 days of data; API does not have information before 1-1-2013; 13-day data is intentionally not omitted

# List of Intervals to Fetch Data
intraday_intervals = ['60', '30', '15', '5']

# Dictionary to Store Data Frames for Each Interval
intraday_data_frames = {}

# Calculate DMI and ADX for Each Interval and Store into a Data Frame
for interval in intraday_intervals:
    print(f'Calculating DMI and ADX values for {interval} minute interval')

    input_path = file_start + r'EURUSD_Spot_Data_with_Signals.xlsx'
    data = pd.read_excel(input_path, sheet_name=f'{interval}_data')
    data.set_index('time', inplace=True)

    print("Original Data:")
    print(data.head())

    data = calculate_dmi_adx(data)
    print("Data with DMI and ADX:")
    print(data.head())

    intraday_data_frames[interval] = data

# Calculate Percentage Changes for Each Interval Without Saving
for interval, data in intraday_data_frames.items():
    print(f'Calculating percentage changes for {interval} minute interval')
    data_with_pct_changes = calculate_percentage_changes(data)
    print(f"Data with Percentage Changes for {interval} minute interval:")
    print(data_with_pct_changes.head())

# Save All Data Frames to Excel File with Multiple Sheets
output_path = file_start + r'EURUSD_Spot_Data_with_DMI_ADX_Signals.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for interval, data in intraday_data_frames.items():
        data.to_excel(writer, sheet_name=f'{interval}_data')

print(f"Data saved to {output_path}")

# **************************************** Regression Analysis ****************************************

# Perform Regression Analysis
def perform_regression_analysis(data, interval):
    # Drop the first 15 rows
    data = data.iloc[15:]

    # Define X and Y variables
    X = data['Pct_Change_Spot']
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    # Regression 1: Percentage Change in Spot Price (X) vs. DMI+ (Y)
    Y1 = data['Pct_Change_DMI+']
    model1 = sm.OLS(Y1, X).fit()
    print(f"Regression results for {interval} interval - DMI+")
    print(model1.summary())

    # Regression 2: Percentage Change in Spot Price (X) vs. DMI- (Y)
    Y2 = data['Pct_Change_DMI-']
    model2 = sm.OLS(Y2, X).fit()
    print(f"Regression results for {interval} interval - DMI-")
    print(model2.summary())

    return model1, model2

# List of Intervals to Fetch Data
intraday_intervals = ['5', '15', '30', '60']

# Dictionary to Store Regression Results
regression_results = {}

# Interpret Results

# Variables to Track Highest and Lowest R-Squared, Adjusted R-Squared, and F-Statistics for DMI+
highest_r2_dmi_plus = {'value': float('-inf'), 'interval': None}
lowest_r2_dmi_plus = {'value': float('inf'), 'interval': None}
highest_adj_r2_dmi_plus = {'value': float('-inf'), 'interval': None}
lowest_adj_r2_dmi_plus = {'value': float('inf'), 'interval': None}
highest_f_stat_dmi_plus = {'value': float('-inf'), 'interval': None}
lowest_f_stat_dmi_plus = {'value': float('inf'), 'interval': None}

# Variables to Track Highest and Lowest R-Squared, Adjusted R-Squared, and F-Statistics for DMI-
highest_r2_dmi_minus = {'value': float('-inf'), 'interval': None}
lowest_r2_dmi_minus = {'value': float('inf'), 'interval': None}
highest_adj_r2_dmi_minus = {'value': float('-inf'), 'interval': None}
lowest_adj_r2_dmi_minus = {'value': float('inf'), 'interval': None}
highest_f_stat_dmi_minus = {'value': float('-inf'), 'interval': None}
lowest_f_stat_dmi_minus = {'value': float('inf'), 'interval': None}

# Process Each Interval
for interval in intraday_intervals:
    print(f"Processing interval: {interval} minutes")

    # Load Pre-Processed Data for the Interval
    input_path = file_start + r'EURUSD_Spot_Data_with_DMI_ADX_Signals.xlsx'
    data = pd.read_excel(input_path, sheet_name=f'{interval}_data')
    data.set_index('time', inplace=True)

    print("Data with DMI and ADX and percentage changes:")
    print(data.head(30))

    # Perform Regression Analysis
    model1, model2 = perform_regression_analysis(data, interval)
    regression_results[interval] = {'DMI+': model1, 'DMI-': model2}

    # Track the Highest and Lowest R-Squared, Adjusted R-Squared, and F-Statistics for DMI+
    if model1.rsquared > highest_r2_dmi_plus['value']:
        highest_r2_dmi_plus['value'] = model1.rsquared
        highest_r2_dmi_plus['interval'] = interval
    if model1.rsquared < lowest_r2_dmi_plus['value']:
        lowest_r2_dmi_plus['value'] = model1.rsquared
        lowest_r2_dmi_plus['interval'] = interval
    if model1.rsquared_adj > highest_adj_r2_dmi_plus['value']:
        highest_adj_r2_dmi_plus['value'] = model1.rsquared_adj
        highest_adj_r2_dmi_plus['interval'] = interval
    if model1.rsquared_adj < lowest_adj_r2_dmi_plus['value']:
        lowest_adj_r2_dmi_plus['value'] = model1.rsquared_adj
        lowest_adj_r2_dmi_plus['interval'] = interval
    if model1.fvalue > highest_f_stat_dmi_plus['value']:
        highest_f_stat_dmi_plus['value'] = model1.fvalue
        highest_f_stat_dmi_plus['interval'] = interval
    if model1.fvalue < lowest_f_stat_dmi_plus['value']:
        lowest_f_stat_dmi_plus['value'] = model1.fvalue
        lowest_f_stat_dmi_plus['interval'] = interval

    # Track the Highest and Lowest R-Squared, Adjusted R-Squared, and F-Statistics for DMI-
    if model2.rsquared > highest_r2_dmi_minus['value']:
        highest_r2_dmi_minus['value'] = model2.rsquared
        highest_r2_dmi_minus['interval'] = interval
    if model2.rsquared < lowest_r2_dmi_minus['value']:
        lowest_r2_dmi_minus['value'] = model2.rsquared
        lowest_r2_dmi_minus['interval'] = interval
    if model2.rsquared_adj > highest_adj_r2_dmi_minus['value']:
        highest_adj_r2_dmi_minus['value'] = model2.rsquared_adj
        highest_adj_r2_dmi_minus['interval'] = interval
    if model2.rsquared_adj < lowest_adj_r2_dmi_minus['value']:
        lowest_adj_r2_dmi_minus['value'] = model2.rsquared_adj
        lowest_adj_r2_dmi_minus['interval'] = interval
    if model2.fvalue > highest_f_stat_dmi_minus['value']:
        highest_f_stat_dmi_minus['value'] = model2.fvalue
        highest_f_stat_dmi_minus['interval'] = interval
    if model2.fvalue < lowest_f_stat_dmi_minus['value']:
        lowest_f_stat_dmi_minus['value'] = model2.fvalue
        lowest_f_stat_dmi_minus['interval'] = interval

# Print Results
print(f"Highest R-squared for DMI+: {highest_r2_dmi_plus['value']} at {highest_r2_dmi_plus['interval']} minute interval")
print(f"Lowest R-squared for DMI+: {lowest_r2_dmi_plus['value']} at {lowest_r2_dmi_plus['interval']} minute interval")

print("\n\n")

print(f"Highest adjusted R-squared for DMI+: {highest_adj_r2_dmi_plus['value']} at {highest_adj_r2_dmi_plus['interval']} minute interval")
print(f"Lowest adjusted R-squared for DMI+: {lowest_adj_r2_dmi_plus['value']} at {lowest_adj_r2_dmi_plus['interval']} minute interval")

print("\n\n")

print(f"Highest F-statistic for DMI+: {highest_f_stat_dmi_plus['value']} at {highest_f_stat_dmi_plus['interval']} minute interval")
print(f"Lowest F-statistic for DMI+: {lowest_f_stat_dmi_plus['value']} at {lowest_f_stat_dmi_plus['interval']} minute interval")

print("\n\n")

print(f"Highest R-squared for DMI-: {highest_r2_dmi_minus['value']} at {highest_r2_dmi_minus['interval']} minute interval")
print(f"Lowest R-squared for DMI-: {lowest_r2_dmi_minus['value']} at {lowest_r2_dmi_minus['interval']} minute interval")

print("\n\n")

print(f"Highest adjusted R-squared for DMI-: {highest_adj_r2_dmi_minus['value']} at {highest_adj_r2_dmi_minus['interval']} minute interval")
print(f"Lowest adjusted R-squared for DMI-: {lowest_adj_r2_dmi_minus['value']} at {lowest_adj_r2_dmi_minus['interval']} minute interval")

print("\n\n")

print(f"Highest F-statistic for DMI-: {highest_f_stat_dmi_minus['value']} at {highest_f_stat_dmi_minus['interval']} minute interval")
print(f"Lowest F-statistic for DMI-: {lowest_f_stat_dmi_minus['value']} at {lowest_f_stat_dmi_minus['interval']} minute interval")

# Save Regression Results to a Text File
with open(file_start + r'EURUSD_Spot_Data_with_DMI_ADX_Regression_Results.txt', 'w') as f:
    for interval, models in regression_results.items():
        f.write(f"Regression results for {interval} interval - DMI+\n")
        f.write(models['DMI+'].summary().as_text())
        f.write("\n\n")
        f.write(f"Regression results for {interval} interval - DMI-\n")
        f.write(models['DMI-'].summary().as_text())
        f.write("\n\n")

print("Regression results saved to EURUSD_Spot_Data_with_DMI_ADX_Regression_Results.txt")

'''
# **************************************** UNABLE TO COMPLETE PORTION ****************************************

# I've put close to 70 hours of work into this project, if not, more. Unfortunately, I was only able to complete it up to here.
# I did my best to find time as I also work 60-80 hours a week.
# I hope the code that I have written in this project (the code above) can show you how much I have learned in this class.
# This was a really fun project for me and I plan to continue it post-class as well
# Thanks, Lukas!

# **************************************** Isolated Regression Analysis ****************************************

# List of volatility columns
volatility_columns = [
    'Volatility Analysis with Standard Deviation',
    'Volatility Analysis with Percentage',
    'Volatility Analysis with Black Swan',
    'Volatility Analysis with IR Decisions',
    'Volatility Analysis with Nonfarm Payrolls',
    'Volatility Analysis with Initial Jobless Claims',
    'Volatility Analysis with Continuing Claims',
    'Volatility Analysis with Unemployment Rate',
    'Volatility Analysis with Eurozone CPI',
    'Volatility Analysis with Eurozone PPI',
    'Volatility Analysis with US CPI',
    'Volatility Analysis with US PPI'
]

# Incomplete GARCH Model Related Code (included in code in case you were interested in looking at it; I plan on continuing this project during the summer)

# **************************************** Return Calculations ****************************************

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio

def calculate_basic_return(close_prices):
    returns = close_prices.pct_change().dropna()
    cumulative_return = (returns + 1).prod() - 1
    return cumulative_return

def calculate_rmsfe(actual, predicted):
    rmsfe = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmsfe

def calculate_max_drawdown(close_prices):
    cumulative_returns = (close_prices.pct_change() + 1).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

# **************************************** Control Group ****************************************

# Function for control group strategies
def control_group_strategy(df, start_position='long'):
    df['Control_Signal'] = np.nan
    df['Control_Returns'] = np.nan

    if start_position == 'long':
        current_position = 1  # Long
    else:
        current_position = -1  # Short

    for i in range(1, len(df)):
        if current_position == 1:
            df.loc[df.index[i], 'Control_Signal'] = 1
            df.loc[df.index[i], 'Control_Returns'] = df.loc[df.index[i], 'Pct_Change_Close']
            if df.loc[df.index[i], 'close'] < df.loc[df.index[i-1], 'close']:
                current_position = -1
        else:
            df.loc[df.index[i], 'Control_Signal'] = -1
            df.loc[df.index[i], 'Control_Returns'] = -df.loc[df.index[i], 'Pct_Change_Close']
            if df.loc[df.index[i], 'close'] > df.loc[df.index[i-1], 'close']:
                current_position = 1

    sharpe_ratio = calculate_sharpe_ratio(df['Control_Returns'].dropna())
    cumulative_return = calculate_basic_return(df['close'])
    max_drawdown = calculate_max_drawdown(df['close'])
    return sharpe_ratio, cumulative_return, max_drawdown

 # Control group: long position
    long_sharpe_ratio, long_cumulative_return, long_max_drawdown = control_group_strategy(data, start_position='long')
    print(f"Control Group (Long) for interval {interval}:")
    print(f"Sharpe Ratio: {long_sharpe_ratio}")
    print(f"Cumulative Return: {long_cumulative_return}")
    print(f"Max Drawdown: {long_max_drawdown}")

    # Control group: short position
    short_sharpe_ratio, short_cumulative_return, short_max_drawdown = control_group_strategy(data, start_position='short')
    print(f"Control Group (Short) for interval {interval}:")
    print(f"Sharpe Ratio: {short_sharpe_ratio}")
    print(f"Cumulative Return: {short_cumulative_return}")
    print(f"Max Drawdown: {short_max_drawdown}")

    # Save to Excel File with Multiple Sheets
    output_path = file_start + r'EURUSD_Spot_Data_with_DMI_ADX_Signals.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        data.to_excel(writer, sheet_name=f'{interval}_data')

print(f"Data saved to {output_path}")

# **************************************** Checking for Stationarity with Augmented Dicker-Fuller Test ****************************************

# Note: Log Returns are time-additive and normalize the data to reduce the impact of large price movements, making it better for statistical modeling

# Function to Perform ADF Test
def perform_adf_test(data, interval):
    # Calculate Log Returns
    data['Log_Returns'] = np.log(data['close'] / data['close'].shift(1))
    data = data.dropna(subset=['Log_Returns'])

    # Perform ADF Test on Log Returns
    result = adfuller(data['Log_Returns'])

    # Interpret the Results
    print(f'\nAugmented Dickey-Fuller Test Results for interval {interval} minute:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    # Determine Stationarity
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

# List of Intervals to Fetch Data
intraday_intervals = ['5', '10', '15', '30', '60']
#intraday_intervals = ['60']

# Save to Excel File with Multiple Sheets
output_path = file_start + r'EURUSD_Spot_Data_with_Signals.xlsx'

# Perform ADF test for each interval
for interval in intraday_intervals:
    fx_data = pd.read_excel(output_path, sheet_name=f'{interval}_data')
    perform_adf_test(fx_data, interval)

  # **************************************** Plotting ACF and PACF ****************************************

# Note: This will show rapid decay for stationary data while non-stationary will have a slow delay

# Set Plot Size
plt.figure(figsize=(12, 6))

# Store AIC Values
aic_values = []
max_lag = 10

interval = '60'

output_path = file_start + r'EURUSD_Spot_Data_with_Signals.xlsx'
data = pd.read_excel(output_path, sheet_name=f'{interval}_data')

data.set_index('time', inplace=True)

data['Log_Returns'] = np.log(data['close'] / data['close'].shift(1))
data = data.dropna(subset=['Log_Returns'])

print('received data', data)

# Calculate AIC Values
for lag in range(1, max_lag + 1):
    print('lag', lag)
    model = ARIMA(data['Log_Returns'], order=(lag, 0, lag))
    results = model.fit()
    aic_values.append(results.aic)

aic_lag = np.argmin(aic_values) + 1

print(f'Optimal lag according to AIC: {aic_lag}')

# Plot ACF
plt.subplot(211)
plot_acf(data['Log_Returns'], lags = aic_lag, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')

# Plot PACF
plt.subplot(212)
plot_pacf(data['Log_Returns'], lags = aic_lag, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')

# Display Plot
plt.tight_layout()
plt.show()

# **************************************** ARCH-LM Test ****************************************

# Note: This tests if there are periods of increased or decreased volatility (heteroskedasticity) for GARCH

# Fit AR Model to Get Residuals
model = ARIMA(data['Log_Returns'], order=(aic_lag, 0, 0))
results = model.fit()

# Perform ARCH-LM Test on Residuals
arch_test = het_arch(results.resid)

# Interpret the Results
print('ARCH-LM Test Statistic:', arch_test[0])
print('p-value:', arch_test[1])
print('F-statistic:', arch_test[2])
print('F-test p-value:', arch_test[3])

# **************************************** Descriptive Statistics ****************************************

# Calculate Descriptive Statistics for Log Returns
descriptive_stats = data['Log_Returns'].describe()
print(descriptive_stats)

# Skewness and Kurtosis
skewness = data['Log_Returns'].skew()
kurtosis = data['Log_Returns'].kurt()
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")

# **************************************** GARCH Model ****************************************

# Define Range of p and q Values
p_values = range(0, 6)
q_values = range(0, 6)

# Initialize Lists to Store Results
best_aic = np.inf
best_model_aic = None

# Loop All Combinations of p and q
for p in p_values:
    for q in q_values:
        try:
            # Fit GARCH Model
            model = arch_model(data_with_atr['Log_Returns'], vol='Garch', p=p, q=q)
            result = model.fit(disp="off")

            # Update Best Model Based on AIC
            if result.aic < best_aic:
                best_aic = result.aic
                best_model_aic = result

        except:
            continue

# Print the Best Model
print("Best model based on AIC:")
print(best_model_aic.summary())

# **************************************** Forecasting with GARCH Model ****************************************

best_model = best_model_aic

# Generate a Forecast for the Next 10 Days
forecast_horizon = 10
forecast = best_model.forecast(start=len(data_with_atr), horizon=forecast_horizon)

# Extract Forecasted Variances
forecast_variances = forecast.variance.iloc[-1].values

# Convert Variances to Standard Deviations
forecast_volatilities = np.sqrt(forecast_variances)

# Create a DataFrame for the Forecast
forecast_dates = pd.date_range(start=data_with_atr['Date'].iloc[-1], periods=forecast_horizon + 1, freq='B')[1:]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Volatility': forecast_volatilities})

# Calculate Mean and Standard Deviation of Forecasted Variances
mean_garch = forecast_volatilities.mean()
std_garch = forecast_volatilities.std()

# Set Threshold for High and Low Volatility Using Standard Deviations
threshold_high = mean_garch + 2 * std_garch
threshold_low = mean_garch - 2 * std_garch

# Add GARCH Volatility Label to the Forecast DataFrame
forecast_df['Volatility Analysis with GARCH'] = forecast_df['Forecasted_Volatility'].apply(
    lambda x: 'High Volatility' if x >= threshold_high else ('Low Volatility' if x <= threshold_low else 'Medium Volatility')
)

# Merge Forecasted Volatilities Back to the Main DataFrame
data_with_atr = pd.concat([data_with_atr, forecast_df], ignore_index=True)

# Save Result to Excel File
output_path = file_start + r'EURUSD_Spot_Data_with_All_and_GARCH_Volatility.xlsx'
data_with_atr.to_excel(output_path, index=False)

print(f"Data with ATR, Black Swan Events, IR Decisions, Nonfarm Payrolls, US Unemployment Rate, Eurozone CPI/PPI, US CPI/PPI, and GARCH has been saved to {output_path}")

# Plot Forecasted Volatility
plt.figure(figsize=(10, 6))
plt.plot(data_with_atr['Date'], data_with_atr['Log_Returns'].rolling(window=30).std(), label='Historical Volatility', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Volatility'], label='Forecasted Volatility', color='red')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Historical and Forecasted Volatility')
plt.legend()
plt.grid(True)
plt.show()
'''