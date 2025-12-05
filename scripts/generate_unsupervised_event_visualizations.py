import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("Loading processed data and unsupervised results...")

# Load processed data
processed_data = pd.read_csv('data/processed/preprocessed_for_ml.csv')
processed_data['date'] = pd.to_datetime(processed_data['date'])

# Load unsupervised results
# Since we don't have a specific unsupervised results file, we'll create an anomaly detection based on the processed data
# We'll identify potential anomalies as significant drops in ASNs compared to recent trends

print(f"Loaded {len(processed_data)} processed records")

# Get countries with the most supervised censorship events
supervised_events = processed_data[processed_data['asn_censorship_indicator'] == 1]
top_countries_with_events = supervised_events['country'].value_counts().head(5).index.tolist()
print(f"Top countries with supervised events: {top_countries_with_events}")

# Function to detect anomalies using statistical method (for unsupervised events)
def detect_anomalies_country(country_data, window=7, threshold=2):
    """
    Detect anomalies as significant drops compared to recent rolling mean
    """
    country_data = country_data.copy()
    # Calculate rolling mean and std for ASN counts
    country_data['rolling_mean'] = country_data['cs_asns_ris'].rolling(window=window, center=True).mean()
    country_data['rolling_std'] = country_data['cs_asns_ris'].rolling(window=window, center=True).std()
    
    # Calculate z-score for deviations below the rolling mean
    country_data['z_score'] = (country_data['cs_asns_ris'] - country_data['rolling_mean']) / (country_data['rolling_std'] + 1e-8)
    
    # Anomalies are significant drops (negative z-score beyond threshold)
    anomalies = country_data[country_data['z_score'] < -threshold]
    
    return anomalies

# Create visualizations showing both supervised and unsupervised events
for country in top_countries_with_events:
    country_data = processed_data[processed_data['country'] == country].copy()
    country_data = country_data.sort_values('date')
    
    # Detect unsupervised anomalies
    unsupervised_anomalies = detect_anomalies_country(country_data)

    # Separate supervised and unsupervised event data
    supervised_country_events = country_data[country_data['asn_censorship_indicator'] == 1]

    plt.figure(figsize=(16, 6))

    # Plot ASN time series
    plt.plot(country_data['date'], country_data['cs_asns_ris'], label='Routed ASNs', linewidth=1.5)
    plt.plot(country_data['date'], country_data['cs_asns_stats'], label='Total ASNs', linewidth=1.5, alpha=0.7)

    # Highlight supervised events
    if len(supervised_country_events) > 0:
        plt.scatter(supervised_country_events['date'], supervised_country_events['cs_asns_ris'],
                   color='red', s=50, zorder=5, label='Supervised Event', alpha=0.8)

    # Highlight unsupervised anomalies
    if len(unsupervised_anomalies) > 0:
        plt.scatter(unsupervised_anomalies['date'], unsupervised_anomalies['cs_asns_ris'],
                   color='blue', s=50, zorder=5, label='Unsupervised Anomaly', alpha=0.8, marker='^')

    plt.title(f'ASN Time Series with Supervised and Unsupervised Events - {country}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('ASN Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'figures/unsupervised_vs_supervised_{country}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent display
    print(f"✓ Unsupervised vs Supervised events visualization saved for {country}")

# Create an overall comparison showing top country matches between supervised and unsupervised
plt.figure(figsize=(16, 12))

for idx, country in enumerate(top_countries_with_events):
    if idx >= 3:  # Only show top 3 for clarity
        break

    country_data = processed_data[processed_data['country'] == country].copy()
    country_data = country_data.sort_values('date')

    # Detect unsupervised anomalies for this country
    unsupervised_anomalies = detect_anomalies_country(country_data)

    # Get supervised events for this country
    supervised_country_events = country_data[country_data['asn_censorship_indicator'] == 1]

    plt.subplot(3, 1, idx + 1)

    plt.plot(country_data['date'], country_data['cs_asns_ris'], label='Routed ASNs', linewidth=1.5)

    # Show both types of events
    if len(supervised_country_events) > 0:
        plt.scatter(supervised_country_events['date'], supervised_country_events['cs_asns_ris'],
                   color='red', s=40, zorder=5, label='Supervised Event', alpha=0.8)

    if len(unsupervised_anomalies) > 0:
        plt.scatter(unsupervised_anomalies['date'], unsupervised_anomalies['cs_asns_ris'],
                   color='blue', s=40, zorder=5, label='Unsupervised Anomaly', alpha=0.8, marker='^')

    plt.title(f'ASN Time Series - {country} (Supervised vs Unsupervised Events)', fontsize=14)
    plt.ylabel('Routed ASN Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

plt.suptitle('Comparison of Supervised and Unsupervised Event Detection', fontsize=16)
plt.tight_layout()
plt.savefig('figures/comparison_supervised_unsupervised.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Supervised vs Unsupervised comparison visualization saved (simplified)")

# Calculate overlap between supervised and unsupervised approaches
print("\\nAnalyzing overlap between supervised and unsupervised methods...")
total_supervised = len(supervised_events)
total_unsupervised = 0
overlap_count = 0

for country in top_countries_with_events:
    country_data = processed_data[processed_data['country'] == country]
    supervised_country_events = country_data[country_data['asn_censorship_indicator'] == 1]
    unsupervised_anomalies = detect_anomalies_country(country_data)
    
    total_unsupervised += len(unsupervised_anomalies)
    
    # Find overlap (same date and country)
    supervised_dates = set(supervised_country_events['date'])
    unsupervised_dates = set(unsupervised_anomalies['date'])
    overlap = supervised_dates.intersection(unsupervised_dates)
    overlap_count += len(overlap)

print(f"Total supervised events: {total_supervised}")
print(f"Total unsupervised anomalies: {total_unsupervised}")
print(f"Overlapping events (same date): {overlap_count}")
if total_supervised > 0:
    print(f"Overlap rate: {overlap_count/total_supervised*100:.2f}% of supervised events also detected by unsupervised method")

print("\\nAll unsupervised event visualizations have been created and saved to the 'figures' directory!")
