# Exercise 0

def github() -> str:

    """
    This function will return Bani Bedi's GitHub page for Problem Set 3.
    """
    
    return "https://github.com/banibedi/Econ481.git"

# Exercise 1

!pip install pyxlsb

import pandas as pd
import requests

def import_yearly_data(years: list) -> pd.DataFrame:
    """
    Import yearly data from the 'Direct Emitters' tab of Excel sheets for specified years.
    Concatenate the data into a single DataFrame, skipping the first three rows.
    Use the fourth row as the header.
    Adds a 'year' column to reference the year of the data.
    """
    yearly_data = pd.DataFrame()
    for y in years:
      df = pd.read_excel(f'https://lukashager.netlify.app/econ-481/data/ghgp_data_{y}.xlsx', skiprows=3, sheet_name='Direct Emitters')
      df[f'year'] = y
      yearly_data = pd.concat([yearly_data, df], axis=0)

    return yearly_data

yearly_data = import_yearly_data(['2019', '2020', '2021', '2022'])
print(yearly_data)

# Exercise 2

import pandas as pd

def import_parent_companies(years: list) -> pd.DataFrame:
    """
    The function "import_parent_companies" takes a list of years and returns a concatenated DataFrame of the corresponding tabs in the parent companies excel sheet.
    The variable "year" references the year from which the data is pulled.
    """

    dataframes = []
    for year in years:
        file_path = f"https://lukashager.netlify.app/econ-481/data/ghgp_data_parent_company_09_2023.xlsb"
        df = pd.read_excel(file_path, sheet_name=f'{year}')  # Specifies 'Parent Companies' sheet
        df['year'] = year  # Adds a new column to the data frame called 'year'
        df = df.dropna(how='all')  # Drops rows that are entirely null
        dataframes.append(df)

    concatenated_df = pd.concat(dataframes, ignore_index=True)  # Concatenates all DataFrame objects
    return concatenated_df

# Test
years_list = ['2019', '2020', '2021', '2022']
company_data = import_parent_companies(years_list)
print(company_data)

# Exercise 3
def n_null(df: pd.DataFrame, col: str) -> int:
    """
    Write a function called n_null that takes a dataframe and column name and returns an integer that shows you the number of null values in that column.
    """
    return df[col].isnull().sum()

print(n_null(company_data, 'FRS ID (FACILITY)'))
print(n_null(company_data, 'GHGRP FACILITY ID'))

# Exercise 4
def clean_data(emissions_data: pd.DataFrame, parent_data: pd.DataFrame) -> pd.DataFrame:
     """
    Write a function called clean_data.
    This takes the data frame of emissions sheets and parent companies and outputs a dataframe.
    ALl columns are lower case.
    Subsets the data on a specific threshold.
    Rearranages the data in a specific format.
    """
    # Left join the parent companies data onto the EPA data
    merged_data = pd.merge(emissions_data, parent_data,
                           left_on=['Facility Id', 'year'],
                           right_on=['GHGRP FACILITY ID', 'year'],
                           how='left')

    # Subset the data to the specified variables
    subset_data = merged_data[['Facility Id', 'year', 'State',
                               'Industry Type (sectors)', 'Total reported direct emissions',
                               'PARENT CO. STATE', 'PARENT CO. PERCENT OWNERSHIP']]

    # Make all the column names lower-case
    subset_data.columns = [col.lower() for col in subset_data.columns]

    return subset_data

cleaned_data = clean_data(yearly_data, company_data)
print(cleaned_data)

# Exercise 5
def aggregate_emissions(cleaned_data: pd.DataFrame, group_vars: list) -> pd.DataFrame:
     """
    Write a function called aggregate_emissions.
    Takes the dataframe from Exercise 4 and a list of variables and produces statitistical values.
    Aggregates total reported direct emissions and parent co. ownership by state level.
    Returns data sorted by highest to lowest mean total reported direct emissions.
    """
    # Group by the specified variables and calculate the required statistics
    aggregated_data = cleaned_data.groupby(group_vars).agg(
        min_total_reported_direct_emissions=('total reported direct emissions', 'min'),
        median_total_reported_direct_emissions=('total reported direct emissions', 'median'),
        mean_total_reported_direct_emissions=('total reported direct emissions', 'mean'),
        max_total_reported_direct_emissions=('total reported direct emissions', 'max'),
        min_parent_co_percent_ownership=('parent co. percent ownership', 'min'),
        median_parent_co_percent_ownership=('parent co. percent ownership', 'median'),
        mean_parent_co_percent_ownership=('parent co. percent ownership', 'mean'),
        max_parent_co_percent_ownership=('parent co. percent ownership', 'max')
    )

    # Sort by highest to lowest mean total reported direct emissions
    aggregated_data = aggregated_data.sort_values(by='mean_total_reported_direct_emissions', ascending=False)

    return aggregated_data

group_vars = ['year']
aggregated_data = aggregate_emissions(cleaned_data, group_vars)
print(aggregated_data)

