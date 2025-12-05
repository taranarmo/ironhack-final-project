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

print("Loading processed data to visualize detected events...")

# Load processed data
processed_data = pd.read_csv('data/processed/preprocessed_for_ml.csv')
processed_data['date'] = pd.to_datetime(processed_data['date'])

print(f"Loaded {len(processed_data)} processed records")

# Get countries with the most censorship events
censorship_events = processed_data[processed_data['asn_censorship_indicator'] == 1]
print(f"Detected {len(censorship_events)} censorship events across {censorship_events['country'].nunique()} countries")

# Get top countries with most events
top_countries_with_events = censorship_events['country'].value_counts().head(5).index.tolist()
print(f"Top countries with events: {top_countries_with_events}")

# Find some events to highlight
for country in top_countries_with_events:
    country_data = processed_data[processed_data['country'] == country].copy()
    country_data = country_data.sort_values('date')
    
    # Create time series with highlighted censorship events
    plt.figure(figsize=(16, 8))
    
    # Plot the main metrics
    plt.plot(country_data['date'], country_data['cs_asns_ris'], label='Routed ASNs', linewidth=1.5)
    plt.plot(country_data['date'], country_data['cs_asns_stats'], label='Total ASNs', linewidth=1.5)
    
    # Highlight censorship events
    event_data = country_data[country_data['asn_censorship_indicator'] == 1]
    if len(event_data) > 0:
        plt.scatter(event_data['date'], event_data['cs_asns_ris'], 
                   color='red', s=50, zorder=5, label='Censorship Event', alpha=0.7)
    
    plt.title(f'Time Series with Censorship Events Highlighted - {country}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('ASN Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'figures/censorship_events_ts_{country}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent display
    print(f"✓ Censorship events visualization saved for {country}")

# Figure: Distribution of events over time
plt.figure(figsize=(14, 8))
events_by_date = censorship_events.groupby('date').size()
plt.plot(events_by_date.index, events_by_date.values, linewidth=2, marker='o', markersize=4)
plt.title('Number of Censorship Events Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/censorship_events_over_time.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Events over time visualization saved")

# Figure: Top countries with most censorship events
plt.figure(figsize=(12, 8))
top_countries_events = censorship_events['country'].value_counts().head(15)
bars = plt.bar(range(len(top_countries_events)), top_countries_events.values, color='coral')
plt.title('Top 15 Countries by Number of Censorship Events', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)
plt.xticks(range(len(top_countries_events)), top_countries_events.index, rotation=45)
# Add value labels on bars
for bar, value in zip(bars, top_countries_events.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01, 
             str(value), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('figures/top_countries_censorship_events.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ Top countries by events visualization saved")

# Figure: Censorship events vs connectivity metrics
if 'foreign_neighbours_share' in processed_data.columns:
    plt.figure(figsize=(14, 8))
    
    # Sample data for better visualization (too many points otherwise)
    sample_data = processed_data.sample(min(50000, len(processed_data)))
    
    # Plot normal data points
    normal_data = sample_data[sample_data['asn_censorship_indicator'] == 0]
    plt.scatter(normal_data['foreign_neighbours_share'], normal_data['cs_asns_ris'], 
               alpha=0.4, s=10, label='Normal', color='blue')
    
    # Plot censorship events
    event_data = sample_data[sample_data['asn_censorship_indicator'] == 1]
    plt.scatter(event_data['foreign_neighbours_share'], event_data['cs_asns_ris'], 
               alpha=0.7, s=20, label='Censorship Event', color='red', zorder=5)
    
    plt.title('Censorship Events vs Foreign Neighbor Share and Routed ASNs', fontsize=16)
    plt.xlabel('Foreign Neighbor Share', fontsize=12)
    plt.ylabel('Routed ASN Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/censorship_events_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent display
    print("✓ Censorship events scatter plot saved")

# Figure: Time series of ratio changes for top event countries
plt.figure(figsize=(16, 10))
for country in top_countries_with_events:
    country_data = processed_data[processed_data['country'] == country].copy()
    country_data = country_data.sort_values('date')
    
    # Calculate ratio of routed to total ASNs
    country_data['asn_ratio'] = country_data['cs_asns_ris'] / country_data['cs_asns_stats']
    
    # Find significant ratio drops (potential censorship indicators)
    country_data['ratio_change'] = country_data['asn_ratio'].pct_change()
    
    plt.subplot(3, 2, top_countries_with_events.index(country) + 1)
    plt.plot(country_data['date'], country_data['asn_ratio'], label='ASN Ratio', linewidth=1.5)
    
    # Highlight significant drops (potential censorship events)
    significant_drops = country_data[country_data['ratio_change'] < -0.1]  # More than 10% drop
    if len(significant_drops) > 0:
        plt.scatter(significant_drops['date'], significant_drops['asn_ratio'], 
                   color='red', s=30, zorder=5, label='Significant Drop', alpha=0.7)
    
    plt.title(f'ASN Ratio Trend - {country}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.suptitle('ASN Ratio Changes - Identifying Potential Censorship Events', fontsize=16)
plt.tight_layout()
plt.savefig('figures/asn_ratio_changes.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to prevent display
print("✓ ASN ratio changes visualization saved")

# Figure: Event severity analysis
plt.figure(figsize=(14, 8))
event_severity = []
for country in top_countries_with_events:
    country_events = censorship_events[censorship_events['country'] == country]
    for _, row in country_events.iterrows():
        # Calculate severity based on relative drop in ASNs
        pre_event_data = processed_data[
            (processed_data['country'] == country) & 
            (processed_data['date'] < row['date']) & 
            (processed_data['date'] >= row['date'] - pd.Timedelta(days=7))  # Week before
        ]
        if len(pre_event_data) > 0:
            baseline_asn = pre_event_data['cs_asns_ris'].mean()
            if baseline_asn > 0:
                severity = (baseline_asn - row['cs_asns_ris']) / baseline_asn
                event_severity.append({
                    'country': country,
                    'date': row['date'],
                    'severity': severity
                })

if event_severity:
    severity_df = pd.DataFrame(event_severity)
    for country in top_countries_with_events:
        country_severity = severity_df[severity_df['country'] == country]
        if len(country_severity) > 0:
            plt.scatter(country_severity['date'], country_severity['severity'], 
                       label=country, alpha=0.7, s=30)
    
    plt.title('Severity of Censorship Events Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Event Severity (Normalized Drop)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/censorship_severity.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent display
    print("✓ Censorship severity visualization saved")

print("\\nAll censorship event visualizations have been created and saved to the 'figures' directory!")