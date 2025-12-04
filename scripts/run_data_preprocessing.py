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

# Load the exported CSVs
print('Loading CSV data...')
asn_data = pd.read_csv('data/raw/asn_data.csv')
connectivity_data = pd.read_csv('data/raw/connectivity_data.csv')
neighbour_data = pd.read_csv('data/raw/neighbour_data.csv')
country_stat_data = pd.read_csv('data/raw/country_stat_data.csv')

print('ASN Data Shape:', asn_data.shape)
print('Connectivity Data Shape:', connectivity_data.shape)
print('Neighbour Data Shape:', neighbour_data.shape)
print('Country Stat Data Shape:', country_stat_data.shape)

print('\nASN Data Columns:', asn_data.columns.tolist())
print('\nConnectivity Data Columns:', connectivity_data.columns.tolist())

# Process ASN data
print('\nProcessing ASN data...')
asn_df = asn_data.copy()
asn_df['a_date'] = pd.to_datetime(asn_df['a_date'])
asn_df = asn_df.sort_values(['a_country_iso2', 'a_date']).reset_index(drop=True)
print(f'ASN data processed: {len(asn_df)} records')
print(f'Countries covered: {asn_df["a_country_iso2"].nunique()}')

# Process connectivity data
print('\nProcessing connectivity data...')
conn_df = connectivity_data.copy()
conn_df['date'] = pd.to_datetime(conn_df['date'])
conn_df = conn_df.sort_values(['asn_country', 'date']).reset_index(drop=True)
print(f'Connectivity data processed: {len(conn_df)} records')
print(f'Countries covered: {conn_df["asn_country"].nunique()}')

# Process country stat data
print('\nProcessing country stat data...')
stat_df = country_stat_data.copy()
stat_df['cs_stats_timestamp'] = pd.to_datetime(stat_df['cs_stats_timestamp'])
stat_df = stat_df.sort_values(['cs_country_iso2', 'cs_stats_timestamp']).reset_index(drop=True)
print(f'Country stat data processed: {len(stat_df)} records')
print(f'Countries covered: {stat_df["cs_country_iso2"].nunique()}')

# Create time-based features
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

# Apply to ASN data
asn_df = add_time_features(asn_df, 'a_date')
print("Added time features to ASN data")

# Apply to connectivity data
conn_df = add_time_features(conn_df, 'date')
print("Added time features to connectivity data")

# Apply to country stat data
stat_df = add_time_features(stat_df, 'cs_stats_timestamp')
print("Added time features to country stat data")

# Create lagged features for ASN data
def create_lagged_features(df, value_col, lags=[1, 7, 14, 30], group_col='a_country_iso2'):
    df = df.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df.groupby(group_col)[value_col].shift(lag)
    
    return df

# Apply to key metrics in ASN data
asn_df = create_lagged_features(asn_df, 'asn_count')
print("Added lagged features to ASN data")

# Create rolling statistics features
def create_rolling_features(df, value_col, window=7, group_col='a_country_iso2'):
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
asn_df = create_rolling_features(asn_df, 'asn_count')
print("Added rolling features to ASN data")

# Create change features
def create_change_features(df, value_col, group_col='a_country_iso2'):
    df = df.copy()
    
    # Day-over-day change
    df[f'{value_col}_pct_change'] = df.groupby(group_col)[value_col].pct_change()
    df[f'{value_col}_diff'] = df.groupby(group_col)[value_col].diff()
    
    return df

# Apply to key metrics
asn_df = create_change_features(asn_df, 'asn_count')
print("Added change features to ASN data")

# Print basic stats
print("\nBasic Statistics for ASN Count:")
print(asn_df['asn_count'].describe())

# Countries with most records
country_counts = asn_df['a_country_iso2'].value_counts()
print(f"\nCountries with most records:")
print(country_counts.head(10))

# Merge data sources
print("\nMerging data sources...")

# Rename columns for clarity
asn_df_renamed = asn_df.rename(columns={'a_date': 'date', 'a_country_iso2': 'country'})
conn_df_renamed = conn_df.rename(columns={'asn_country': 'country'})
stat_df_renamed = stat_df.rename(columns={'cs_country_iso2': 'country', 'cs_stats_timestamp': 'date'})

# Merge ASN and connectivity data
merged_df = pd.merge(asn_df_renamed, conn_df_renamed, on=['country', 'date'], how='outer')
print(f"Merged ASN and connectivity: {merged_df.shape}")

# Merge with country stats
final_df = pd.merge(merged_df, stat_df_renamed, on=['country', 'date'], how='outer')
print(f"Final merged dataset: {final_df.shape}")

# Create censorship indicators based on connectivity drops
def create_censorship_indicators(df):
    df = df.copy()
    
    # Create binary indicator for significant drops in connectivity
    # Using foreign_neighbours_share if available, otherwise using asn_count
    
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
        
    # If we have asn_count, use that as an alternative metric
    if 'asn_count' in df.columns:
        # Calculate rolling median for ASN count
        df['asn_count_rolling_median'] = df.groupby('country')['asn_count'].transform(
            lambda x: x.rolling(window=30, min_periods=7).median()
        )
        
        # Create indicator for significant drop in ASN count
        df['asn_censorship_indicator'] = (
            (df['asn_count'] < df['asn_count_rolling_median'] * 0.7) &
            (df['asn_count'].notna())
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