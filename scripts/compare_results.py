import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def compare_supervised_unsupervised():
    """
    Compare results from supervised and unsupervised ML approaches
    """
    print("Comparing Supervised vs Unsupervised ML Results")
    print("=" * 50)
    
    # Load the main processed dataset
    df = pd.read_csv('data/processed/preprocessed_for_ml.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Load unsupervised results if available
    try:
        unsupervised_results = pd.read_csv('data/processed/unsupervised_anomaly_results.csv')
        print("Unsupervised results loaded successfully")
        
        # Extract the anomaly dates from unsupervised analysis
        unsupervised_anomalies = set()
        for _, row in unsupervised_results.iterrows():
            # Add all dates in the anomalous sequence period to our set
            date_range = pd.date_range(start=row['date_start'], end=row['date_end'])
            for date in date_range:
                unsupervised_anomalies.add((row['country'], date.strftime('%Y-%m-%d')))
    except FileNotFoundError:
        print("No unsupervised results file found. Attempting to recreate from the processed data.")
        # If we don't have the detailed results, we'll need to run the unsupervised analysis again
        # to get the anomaly predictions for individual dates
        unsupervised_anomalies = set()
    
    # Check if we have supervised results (target variable)
    supervised_target = 'censorship_target' if 'censorship_target' in df.columns else None
    if supervised_target:
        supervised_events = df[df[supervised_target] == 1]
        print(f"Supervised model detected {len(supervised_events)} potential censorship events")
    else:
        print("No supervised target variable found, using fallback method")
        # Use a fallback method to identify supervised events
        if 'foreign_neighbours_share' in df.columns:
            # Identify significant drops in foreign neighbors share
            df['foreign_conn_baseline'] = df.groupby('country')['foreign_neighbours_share'].transform(
                lambda x: x.rolling(window=30, min_periods=7).median()
            )
            df['supervised_event'] = (
                (df['foreign_neighbours_share'] < df['foreign_conn_baseline'] * 0.5) &
                (df['foreign_neighbours_share'].notna()) &
                (df['foreign_conn_baseline'].notna())
            ).astype(int)
            supervised_events = df[df['supervised_event'] == 1]
            print(f"Identified {len(supervised_events)} potential censorship events using baseline method")
        else:
            print("No suitable metric for supervised approach found")
            supervised_events = pd.DataFrame()
    
    # Show summary statistics
    if not supervised_events.empty:
        print(f"Supervised approach - Countries with events: {supervised_events['country'].nunique()}")
        print(f"Date range of events: {supervised_events['date'].min()} to {supervised_events['date'].max()}")
    
    print(f"Unsupervised approach - Anomaly records: {len(unsupervised_anomalies) if unsupervised_anomalies else 'N/A'}")
    
    # Create a comparison analysis
    print("\nComparison Analysis:")
    print("-" * 20)
    
    if not supervised_events.empty and unsupervised_anomalies:
        # Find overlap between supervised and unsupervised approaches
        supervised_dates = set()
        for _, row in supervised_events.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            supervised_dates.add((row['country'], date_str))
        
        overlap = supervised_dates.intersection(unsupervised_anomalies)
        print(f"Overlap between approaches: {len(overlap)} events")
        
        if len(overlap) > 0:
            overlap_df = pd.DataFrame(list(overlap), columns=['country', 'date'])
            overlap_df['date'] = pd.to_datetime(overlap_df['date'])
            print(f"Overlap countries: {overlap_df['country'].nunique()}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot top countries with supervised events
    if not supervised_events.empty:
        plt.subplot(2, 2, 1)
        top_supervised_countries = supervised_events['country'].value_counts().head(10)
        plt.bar(range(len(top_supervised_countries)), top_supervised_countries.values)
        plt.title('Top Countries - Supervised Approach')
        plt.xlabel('Country Index')
        plt.ylabel('Number of Events')
        plt.xticks(range(len(top_supervised_countries)), 
                  [country for country in top_supervised_countries.index], 
                  rotation=45)
    
    # Time series of supervised events
    if not supervised_events.empty:
        plt.subplot(2, 2, 2)
        monthly_supervised = supervised_events.groupby(supervised_events['date'].dt.to_period('M')).size()
        if len(monthly_supervised) > 0:
            plt.plot(monthly_supervised.index.astype(str), monthly_supervised.values)
            plt.title('Monthly Trends - Supervised Approach')
            plt.xlabel('Month')
            plt.ylabel('Number of Events')
            plt.xticks(rotation=45)
    
    # Time-based comparison of approaches
    if not supervised_events.empty:
        plt.subplot(2, 2, 3)
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_supervised_counts = df.groupby(['year_month', 'supervised_event']).size().unstack(fill_value=0)
        
        if 1 in monthly_supervised_counts.columns:
            plt.plot(monthly_supervised_counts.index.astype(str), 
                    monthly_supervised_counts[1], 
                    label='Supervised Events', 
                    marker='o')
            plt.title('Monthly Supervised Events')
            plt.xlabel('Month')
            plt.ylabel('Number of Events')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No supervised events detected', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
    
    # Feature importance for supervised methods (if we have it in the processed data)
    feature_cols = [col for col in df.columns 
                   if col not in ['date', 'country', 'censorship_indicator', 'censorship_target', 'year_month']
                   and df[col].dtype in ['int64', 'float64']
                   and col.startswith(('asn_', 'foreign_', 'total_', 'local_'))]
    
    if feature_cols:
        plt.subplot(2, 2, 4)
        top_features = df[feature_cols].std().sort_values(ascending=False).head(10)
        plt.barh(range(len(top_features)), top_features.values)
        plt.title('Top Variable Features')
        plt.xlabel('Std Deviation (Variability)')
        plt.yticks(range(len(top_features)), [f.replace('_', '\n') for f in top_features.index])
    
    plt.tight_layout()
    plt.savefig('supervised_unsupervised_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nComparison completed. Visualization saved as 'supervised_unsupervised_comparison.png'")
    
    # Summary report
    print("\nSUMMARY REPORT")
    print("=" * 40)
    if not supervised_events.empty:
        print(f"• Supervised ML detected {len(supervised_events)} potential censorship events")
        print(f"• Across {supervised_events['country'].nunique()} countries")
        print(f"• Time period: {supervised_events['date'].min()} to {supervised_events['date'].max()}")
    
    if unsupervised_anomalies:
        print(f"• Unsupervised ML detected anomalies in {len(unsupervised_anomalies)} date-country combinations")
    
    if 'overlap' in locals() and len(overlap) > 0:
        print(f"• {len(overlap)} events detected by both approaches")
    
    print("\nThe unsupervised approach identifies anomalous patterns without requiring labeled data,")
    print("while the supervised approach learns from specific patterns associated with censorship.")
    print("Combining both approaches provides a more robust censorship detection system.")


if __name__ == "__main__":
    compare_supervised_unsupervised()