import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("Loading raw data and cleaning for visualization...")

# Load datasets
country_stat_data = pd.read_csv('data/raw/country_stat_data.csv')
connectivity_data = pd.read_csv('data/raw/connectivity_data.csv')

# Convert date columns
country_stat_data['date'] = pd.to_datetime(country_stat_data['cs_stats_timestamp'])
connectivity_data['date'] = pd.to_datetime(connectivity_data['date'])

# Fix: Drop records where routed ASNs > total ASNs (these are invalid data points)
invalid_mask = (country_stat_data['cs_asns_ris'] > country_stat_data['cs_asns_stats']) & (country_stat_data['cs_asns_stats'] > 0)
print(f"Removing {invalid_mask.sum()} invalid records where routed > total...")

cleaned_country_data = country_stat_data[~invalid_mask].copy()
print(f"After cleaning: {len(cleaned_country_data)} records remain")

# Create normalized/corrected visualizations

# Figure 1: Corrected Time series of ASN counts by top countries - with normalization
plt.figure(figsize=(16, 10))
# Get top countries by latest measured routed ASN count
latest_data = cleaned_country_data.loc[cleaned_country_data.groupby('cs_country_iso2')['date'].idxmax()]
top_countries = latest_data.nlargest(10, 'cs_asns_ris')['cs_country_iso2'].tolist()

for i, country in enumerate(top_countries):
    country_data = cleaned_country_data[cleaned_country_data['cs_country_iso2'] == country].copy()
    country_data = country_data.sort_values('date')
    # Normalize by country-specific maximum for better comparison
    country_max = country_data['cs_asns_ris'].max()
    if country_max > 0:
        normalized_values = country_data['cs_asns_ris'] / country_max
    else:
        normalized_values = country_data['cs_asns_ris']
    
    plt.plot(country_data['date'], normalized_values, 
             label=f'{country} (Max: {country_max:.0f})', linewidth=2, alpha=0.8)

plt.title('Time Series of Routed ASN Counts (Normalized by Country Max)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Normalized Routed ASN Count', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/asn_time_series_normalized.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Normalized ASN Time Series saved")

# Figure 2: Last snapshot comparison - corrected
plt.figure(figsize=(16, 8))
# Get the latest data for each country
latest_data = cleaned_country_data.loc[cleaned_country_data.groupby('cs_country_iso2')['date'].idxmax()].copy()

# Get top 25 countries by latest routed ASN count
top_latest = latest_data.nlargest(25, 'cs_asns_ris')

plt.subplot(1, 2, 1)
bars = plt.bar(range(len(top_latest)), top_latest['cs_asns_ris'], color='lightblue', edgecolor='darkblue')
plt.title('Top 25 Countries by Latest Routed ASN Count', fontsize=14)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Latest Routed ASN Count', fontsize=12)
plt.xticks(range(len(top_latest)), top_latest['cs_country_iso2'], rotation=45)
# Add value labels on bars
for bar, value in zip(bars, top_latest['cs_asns_ris']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01, 
             f'{int(value)}', ha='center', va='bottom', fontsize=8)

plt.subplot(1, 2, 2)
plt.scatter(top_latest['cs_asns_stats'], top_latest['cs_asns_ris'], alpha=0.6)
plt.xlabel('Total ASNs')
plt.ylabel('Routed ASNs')
plt.title('Routed vs Total ASNs (Latest Values)')
plt.plot([0, top_latest['cs_asns_stats'].max()], [0, top_latest['cs_asns_stats'].max()], 'r--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.suptitle('Latest ASN Counts - Snapshot View', fontsize=16)
plt.tight_layout()
plt.savefig('figures/latest_asn_counts.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Latest ASN Counts snapshot saved")

# Figure 3: Foreign Neighbor Share trends (no normalization for share values)
plt.figure(figsize=(16, 10))
top_conn_countries = connectivity_data['asn_country'].value_counts().head(10).index

for i, country in enumerate(top_conn_countries):
    country_data = connectivity_data[connectivity_data['asn_country'] == country].copy()
    country_data = country_data.sort_values('date')
    # Keep raw share values, no normalization for share metrics
    
    plt.plot(country_data['date'], country_data['foreign_neighbours_share'],
             label=country, linewidth=2, alpha=0.8)

plt.title('Foreign Neighbor Share Trends', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Foreign Neighbor Share', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/foreign_share_trends.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Foreign Share Trends saved (no normalization)")

# Figure 4: Distribution of ASN counts (corrected)
plt.figure(figsize=(12, 8))
valid_data = cleaned_country_data[(cleaned_country_data['cs_asns_ris'] > 0) & (cleaned_country_data['cs_asns_ris'] <= cleaned_country_data['cs_asns_stats'])]
plt.hist(valid_data['cs_asns_ris'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(valid_data['cs_asns_ris'].median(), color='red', linestyle='--', label='Median')
plt.axvline(valid_data['cs_asns_ris'].mean(), color='orange', linestyle='--', label='Mean')
plt.title('Distribution of Valid Routed ASN Counts (cs_asns_ris)', fontsize=16)
plt.xlabel('Routed ASN Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/asn_distribution_corrected.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Corrected ASN Distribution saved")

# Figure 5: Monthly trends (no normalization for share values)
plt.figure(figsize=(16, 10))
connectivity_data['year_month'] = connectivity_data['date'].dt.to_period('M')
monthly_conn = connectivity_data.groupby(['asn_country', 'year_month'])['foreign_neighbours_share'].mean().unstack(fill_value=0)

# Get top countries by average foreign neighbor share
top_countries_by_avg = connectivity_data.groupby('asn_country')['foreign_neighbours_share'].mean().sort_values(ascending=False).head(8).index

for country in top_countries_by_avg:
    if country in monthly_conn.index:
        monthly_series = monthly_conn.loc[country]
        monthly_series.index = pd.to_datetime(monthly_series.index.astype(str))

        # Keep raw share values, no normalization for share metrics
        plt.plot(monthly_series.index, monthly_series, label=country, linewidth=2)

plt.title('Monthly Average Foreign Neighbor Share', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Foreign Neighbor Share', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/monthly_foreign_share.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Monthly Foreign Share saved (no normalization)")

# Figure 6: Comparison of routed vs total ASNs
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
valid_data = cleaned_country_data[(cleaned_country_data['cs_asns_ris'] > 0) & (cleaned_country_data['cs_asns_ris'] <= cleaned_country_data['cs_asns_stats'])]
plt.scatter(valid_data['cs_asns_stats'], valid_data['cs_asns_ris'], alpha=0.6)
plt.xlabel('Total ASNs (cs_asns_stats)')
plt.ylabel('Routed ASNs (cs_asns_ris)')
plt.title('Routed vs Total ASNs Comparison (Valid Data Only)')
plt.plot([0, valid_data['cs_asns_stats'].max()], [0, valid_data['cs_asns_stats'].max()], 'r--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(valid_data['cs_asns_stats'] - valid_data['cs_asns_ris'], bins=30, alpha=0.7)
plt.xlabel('Difference (Total - Routed)')
plt.ylabel('Frequency')
plt.title('Difference: Total - Routed ASNs')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.hist(valid_data['cs_asns_ris'], bins=30, alpha=0.6, label='Routed ASNs')
plt.hist(valid_data['cs_asns_stats'], bins=30, alpha=0.6, label='Total ASNs')
plt.xlabel('ASN Count')
plt.ylabel('Frequency')
plt.title('Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
ratio_data = valid_data[(valid_data['cs_asns_stats'] > 0)]
ratios = ratio_data['cs_asns_ris'] / ratio_data['cs_asns_stats']
plt.hist(ratios.dropna(), bins=30, alpha=0.7)
plt.xlabel('Routed/Total Ratio')
plt.ylabel('Frequency')
plt.title('Ratio of Routed to Total ASNs')
plt.grid(True, alpha=0.3)

plt.suptitle('Comparison of Routed vs Total ASNs (Valid Data Only)', fontsize=16)
plt.tight_layout()
plt.savefig('figures/routed_vs_total_asn_comparison_corrected.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Corrected Routed vs Total ASN Comparison saved")

# Figure 7: Connectivity metrics correlation heatmap
plt.figure(figsize=(10, 8))
conn_cols = ['asn_count', 'foreign_neighbour_count', 'local_neighbour_count', 
             'total_neighbour_count', 'foreign_neighbours_share']
conn_numeric = connectivity_data[conn_cols].select_dtypes(include=[np.number]).dropna()
correlation_matrix = conn_numeric.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Connectivity Metrics', fontsize=16)
plt.tight_layout()
plt.savefig('figures/connectivity_correlations.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Connectivity Correlations saved")

# Figure 8: Boxplot of connectivity metrics by country (top 15, no normalization for share values)
plt.figure(figsize=(18, 10))
# Select countries with sufficient data
top_countries = connectivity_data.groupby('asn_country').size().sort_values(ascending=False).head(15).index
filtered_data = connectivity_data[connectivity_data['asn_country'].isin(top_countries)]

sns.boxplot(data=filtered_data, x='asn_country', y='foreign_neighbours_share')
plt.title('Distribution of Foreign Neighbor Share by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Foreign Neighbor Share', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/foreign_share_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Foreign Share Boxplot saved (no normalization)")

# Figure 9: Time series of ratio of routed/total ASNs for selected countries
plt.figure(figsize=(16, 10))
selected_countries = ['US', 'CN', 'RU', 'UA', 'DE', 'FR', 'GB', 'JP', 'IN', 'BR']

for country in selected_countries:
    country_data = cleaned_country_data[cleaned_country_data['cs_country_iso2'] == country].copy()
    country_data = country_data.sort_values('date')
    # Filter to valid data only where total >= routed
    valid_country_data = country_data[country_data['cs_asns_ris'] <= country_data['cs_asns_stats']].copy()
    
    # Calculate ratio of routed/total
    valid_country_data['asn_ratio'] = valid_country_data['cs_asns_ris'] / valid_country_data['cs_asns_stats']
    
    if len(valid_country_data) > 0:
        plt.plot(valid_country_data['date'], valid_country_data['asn_ratio'], label=country, linewidth=2)

plt.title('Ratio of Routed to Total ASNs Over Time - Selected Countries', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Routed/Total ASN Ratio', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/routed_total_ratio_time_series.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Routed/Total ASN Ratio Time Series saved")

# Figure 10: ASN growth rates by country
plt.figure(figsize=(16, 10))
# Calculate growth rates for top countries
latest_data = cleaned_country_data.loc[cleaned_country_data.groupby('cs_country_iso2')['date'].idxmax()]
earliest_data = cleaned_country_data.loc[cleaned_country_data.groupby('cs_country_iso2')['date'].idxmin()]

# Merge to compare earliest and latest
comparison = pd.merge(
    earliest_data[['cs_country_iso2', 'cs_asns_ris', 'date']].rename(columns={'date': 'date_first', 'cs_asns_ris': 'asn_first'}),
    latest_data[['cs_country_iso2', 'cs_asns_ris', 'date']].rename(columns={'date': 'date_last', 'cs_asns_ris': 'asn_last'}),
    on='cs_country_iso2'
)

# Calculate growth
comparison = comparison[comparison['asn_first'] > 0].copy()  # Only countries with positive first values
comparison['growth_rate'] = (comparison['asn_last'] - comparison['asn_first']) / comparison['asn_first']

# Get top 15 countries by growth rate
top_growing = comparison.nlargest(15, 'growth_rate')

plt.barh(range(len(top_growing)), top_growing['growth_rate'], color='lightgreen', edgecolor='darkgreen')
plt.title('Top 15 Countries by ASN Routed Growth Rate', fontsize=16)
plt.xlabel('Growth Rate (Fold Increase)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.yticks(range(len(top_growing)), top_growing['cs_country_iso2'])
# Add growth percentage labels
for i, (idx, row) in enumerate(top_growing.iterrows()):
    pct_val = round(row['growth_rate']*100, 1)
    plt.text(row['growth_rate'] + row['growth_rate']*0.01, i, str(pct_val)+'%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/asn_growth_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ ASN Growth Rate Comparison saved")

print("\\nAll corrected visualizations have been created and saved to the 'figures' directory!")
print("Total of 10 corrected and enhanced visualizations created.")
