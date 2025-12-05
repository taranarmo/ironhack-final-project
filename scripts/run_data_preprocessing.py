import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For plotting
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Load the exported CSVs - using country_stat_data for actual ASN counts
connectivity_data = pd.read_csv('data/raw/connectivity_data.csv')
neighbour_data = pd.read_csv('data/raw/neighbour_data.csv')
country_stat_data = pd.read_csv('data/raw/country_stat_data.csv')

print('Connectivity Data Shape:', connectivity_data.shape)
print('Neighbour Data Shape:', neighbour_data.shape)
print('Country Stat Data Shape:', country_stat_data.shape)

print('\nConnectivity Data Columns:', connectivity_data.columns.tolist())
print('\nCountry Stat Data Columns:', country_stat_data.columns.tolist())

# Process country stat data - this contains the actual ASN counts (cs_asns_ris and cs_asns_stats)
print('\nProcessing country stat data...')
stat_df = country_stat_data.copy()
stat_df['date'] = pd.to_datetime(stat_df['cs_stats_timestamp'])
stat_df = stat_df.rename(columns={'cs_country_iso2': 'country'})
stat_df = stat_df.sort_values(['country', 'date']).reset_index(drop=True)
print(f'Country stat data processed: {len(stat_df)} records')
print(f'Countries covered: {stat_df["country"].nunique()}')

# Process connectivity data
print('\nProcessing connectivity data...')
conn_df = connectivity_data.copy()
conn_df['date'] = pd.to_datetime(conn_df['date'])
conn_df = conn_df.rename(columns={'asn_country': 'country'})  # Standardize country column name
conn_df = conn_df.sort_values(['country', 'date']).reset_index(drop=True)
print(f'Connectivity data processed: {len(conn_df)} records')
print(f'Countries covered: {conn_df["country"].nunique()}')

# Process neighbour data
print('\nProcessing neighbour data...')
neigh_df = neighbour_data.copy()
print(f'Neighbour data processed: {len(neigh_df)} records')

# Create time-based features for country stat data
def add_time_features(df, date_col):
    df = df.copy()
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    return df

# Apply to country stat data
stat_df = add_time_features(stat_df, 'date')
print("Added time features to country stat data")

# Apply to connectivity data
conn_df = add_time_features(conn_df, 'date')
print("Added time features to connectivity data")

# Create lagged features for country stat data (using ASN data from country_stat)
def create_lagged_features(df, value_col, lags=[1, 7, 14, 30], group_col='country'):
    df = df.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df.groupby(group_col)[value_col].shift(lag)
    
    return df

# Apply to key metrics in country stat data (using cs_asns_ris as the primary ASN count)
if 'cs_asns_ris' in stat_df.columns:
    stat_df = create_lagged_features(stat_df, 'cs_asns_ris')
    print("Added lagged features to country stat data (cs_asns_ris)")

# Create rolling statistics features
def create_rolling_features(df, value_col, window=7, group_col='country'):
    df = df.copy()
    
    df[f'{value_col}_rolling_mean_{window}'] = df.groupby(group_col)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    df[f'{value_col}_rolling_std_{window}'] = df.groupby(group_col)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    
    # Calculate z-score based on rolling statistics
    df[f'{value_col}_zscore_{window}'] = (
        df[value_col] - df[f'{value_col}_rolling_mean_{window}']
    ) / (df[f'{value_col}_rolling_std_{window}'] + 1e-8)  # Add small value to avoid division by zero
    
    return df

# Apply to key metrics
if 'cs_asns_ris' in stat_df.columns:
    stat_df = create_rolling_features(stat_df, 'cs_asns_ris')
    print("Added rolling features to country stat data (cs_asns_ris)")

# Create change features
def create_change_features(df, value_col, group_col='country'):
    df = df.copy()
    
    # Day-over-day change
    df[f'{value_col}_pct_change'] = df.groupby(group_col)[value_col].pct_change()
    df[f'{value_col}_diff'] = df.groupby(group_col)[value_col].diff()
    
    return df

# Apply to key metrics
if 'cs_asns_ris' in stat_df.columns:
    stat_df = create_change_features(stat_df, 'cs_asns_ris')
    print("Added change features to country stat data (cs_asns_ris)")

# Print basic stats
if 'cs_asns_ris' in stat_df.columns:
    print("\nBasic Statistics for ASN Count (cs_asns_ris):")
    print(stat_df['cs_asns_ris'].describe())

# Countries with most records
country_counts = stat_df['country'].value_counts()
print(f"\nCountries with most records:")
print(country_counts.head(10))

# Merge data sources
print("\nMerging data sources...")

# Merge connectivity and country stats data
final_df = pd.merge(conn_df, stat_df, on=['country', 'date'], how='outer')
print(f"Final merged dataset: {final_df.shape}")

# Create censorship indicators based on connectivity drops
def create_censorship_indicators(df):
    df = df.copy()

    # Create binary indicator for significant drops in connectivity
    # Using foreign_neighbours_share if available, otherwise using ASN counts from country stats

    # If we have foreign_neighbours_share, use that as the primary metric
    if 'foreign_neighbours_share' in df.columns:
        # Calculate rolling median for each country
        df['foreign_conn_rolling_median'] = df.groupby('country')['foreign_neighbours_share'].transform(
            lambda x: x.rolling(window=30, min_periods=7).median()
        )

        # Create indicator for significant drop below baseline
        df['censorship_indicator'] = (
            (df['foreign_neighbours_share'] < df['foreign_conn_rolling_median'] * 0.7) &
            (df['foreign_neighbours_share'].notna())
        ).astype(int)

    # If we have cs_asns_ris from country stats, use that as an alternative metric
    if 'cs_asns_ris' in df.columns:
        # Calculate rolling median for ASN count from country stats
        df['asn_count_rolling_median'] = df.groupby('country')['cs_asns_ris'].transform(
            lambda x: x.rolling(window=30, min_periods=7).median()
        )

        # Create indicator for significant drop in ASN count
        df['asn_censorship_indicator'] = (
            (df['cs_asns_ris'] < df['asn_count_rolling_median'] * 0.7) &
            (df['cs_asns_ris'].notna())
        ).astype(int)

    return df

final_df = create_censorship_indicators(final_df)
print("Created censorship indicators")

# Count how many potential censorship events were detected
if 'censorship_indicator' in final_df.columns:
    print(f"Potential censorship events detected: {final_df['censorship_indicator'].sum()}")
if 'asn_censorship_indicator' in final_df.columns:
    print(f"Potential ASN-based censorship events detected: {final_df['asn_censorship_indicator'].sum()}")

# Select final features for ML
print("\nPreparing final dataset for ML...")

# Select numeric columns for ML
numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Number of numeric features: {len(numeric_cols)}")

# Handle missing values
final_df_ml = final_df.copy()
final_df_ml = final_df_ml.fillna(final_df_ml.median(numeric_only=True))

# Ensure date and country columns are preserved
if 'date' in final_df.columns:
    final_df_ml['date'] = final_df['date']
if 'country' in final_df.columns:
    final_df_ml['country'] = final_df['country']

print(f"Final dataset shape: {final_df_ml.shape}")
print(f"Final dataset columns: {final_df_ml.shape[1]}")

# Look at potential target variables
target_cols = ['censorship_indicator', 'asn_censorship_indicator']
for col in target_cols:
    if col in final_df_ml.columns:
        print(f"{col}: {final_df_ml[col].sum()} positive events ({final_df_ml[col].mean():.4f} ratio)")

# Save the preprocessed data for ML
output_path = 'data/processed/preprocessed_for_ml.csv'
final_df_ml.to_csv(output_path, index=False)
print(f"\nPreprocessed data saved to {output_path}")

# Show sample of preprocessed data
print("\nSample of preprocessed data:")
print(final_df_ml.head(5))

print("\nData preprocessing completed successfully!")