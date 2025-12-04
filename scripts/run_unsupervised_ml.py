import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def build_autoencoder_model(input_shape):
    """Build the LSTM autoencoder model"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_shape[1]))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def run_unsupervised_anomaly_detection():
    """
    Unsupervised anomaly detection using LSTM autoencoder for time series
    """
    print("Loading data for unsupervised anomaly detection...")
    df = pd.read_csv('data/processed/preprocessed_for_ml.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare features for time series modeling
    feature_cols = [col for col in df.columns 
                    if col not in ['date', 'country'] 
                    and df[col].dtype in ['int64', 'float64']]
    
    # Select a few key features for demonstration
    key_features = [
        'asn_count_x', 'foreign_neighbours_share', 
        'total_neighbour_count', 'asn_count_rolling_mean_7'
    ]
    
    # Filter to only use features that actually exist in the dataset
    available_features = [f for f in key_features if f in df.columns]
    if not available_features:
        # If key features don't exist, use any numeric features
        available_features = [col for col in df.columns 
                             if col not in ['date', 'country'] 
                             and df[col].dtype in ['int64', 'float64']]
        available_features = available_features[:4]  # Take first 4 features
    
    print(f"Using features: {available_features}")
    
    # Handle missing values
    df_clean = df.copy()
    df_clean[available_features] = df_clean[available_features].fillna(df_clean[available_features].median())
    
    # Create time series sequences for each country
    countries = df_clean['country'].unique()
    all_sequences = []
    
    sequence_length = 30  # Use 30 days of data for each sequence
    
    for country in countries:
        country_data = df_clean[df_clean['country'] == country].sort_values('date')
        
        if len(country_data) < sequence_length:
            continue
            
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(country_data[available_features])
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - sequence_length + 1):
            sequences.append(scaled_data[i:i + sequence_length])
        
        if sequences:
            country_sequences = np.array(sequences)
            # Add country and sequence index for tracking
            for idx, seq in enumerate(country_sequences):
                all_sequences.append({
                    'sequence': seq,
                    'country': country,
                    'date_start': country_data['date'].iloc[idx],
                    'date_end': country_data['date'].iloc[idx + sequence_length - 1]
                })
    
    if not all_sequences:
        print("No valid sequences found. Using simpler approach...")
        # Fallback: Use a simpler approach with all available features
        df_clean = df_clean.dropna(subset=available_features)
        if len(df_clean) < sequence_length:
            print("Dataset too small for sequence-based approach. Using simpler anomaly detection.")
            simple_unsupervised_approach(df, available_features)
            return
    
    print(f"Created {len(all_sequences)} sequences for training")
    
    # Prepare data for autoencoder
    X = np.array([seq['sequence'] for seq in all_sequences])
    print(f"Data shape: {X.shape}")

    # Define model path
    model_path = 'models/unsupervised_autoencoder.keras'  # Use .keras format for newer Keras versions
    os.makedirs('models', exist_ok=True)

    # Check if model exists and load it, otherwise build and train
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Building new autoencoder model...")
        model = build_autoencoder_model((X.shape[1], X.shape[2]))

        print("Model summary:")
        print(model.summary())

        # Train the autoencoder
        print("Training autoencoder...")
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        history = model.fit(X, X, epochs=50, batch_size=512, verbose=1, callbacks=[early_stopping])

        # Save the trained model
        print(f"Saving model to {model_path}")
        model.save(model_path)
    
    # Use the trained model for anomaly detection
    print("Performing anomaly detection...")
    reconstructed = model.predict(X)
    
    # Calculate reconstruction errors
    mse = np.mean(np.power(X - reconstructed, 2), axis=(1, 2))
    
    # Define anomalies as points with reconstruction error above threshold
    threshold = np.percentile(mse, 95)  # Top 5% are anomalies
    anomalies = mse > threshold
    
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(mse)} sequences")
    print(f"Anomaly threshold: {threshold:.4f}")
    
    # Create anomaly dataframe for visualization
    anomaly_df = pd.DataFrame({
        'sequence_idx': range(len(mse)),
        'reconstruction_error': mse,
        'is_anomaly': anomalies,
        'country': [seq['country'] for seq in all_sequences],
        'date_start': [seq['date_start'] for seq in all_sequences],
        'date_end': [seq['date_end'] for seq in all_sequences]
    })
    
    print("Anomaly Detection Results:")
    print(f"Total sequences: {len(anomaly_df)}")
    print(f"Anomalous sequences: {anomaly_df['is_anomaly'].sum()}")
    print(f"Anomaly rate: {anomaly_df['is_anomaly'].mean():.4f}")
    
    # Show top anomaly dates
    top_anomalies = anomaly_df[anomaly_df['is_anomaly']].sort_values('reconstruction_error', ascending=False).head(10)
    print(f"\nTop 10 anomalous periods:")
    for _, row in top_anomalies.iterrows():
        print(f"  {row['date_start']} to {row['date_end']} | Country: {row['country']} | Error: {row['reconstruction_error']:.4f}")
    
    # Create anomaly predictions that we can compare with our supervised models
    # We'll use the most recent data points to compare
    # Create a more efficient approach using pandas operations
    df_clean['unsupervised_anomaly'] = 0  # Initialize column

    # For performance, only process if we have anomalies
    if np.any(anomalies):
        # Create a DataFrame with anomaly intervals
        anomaly_data = []
        for i, seq_info in enumerate(all_sequences):
            if anomalies[i]:
                # Add each day in the sequence as a separate entry
                date_range = pd.date_range(start=seq_info['date_start'], end=seq_info['date_end'])
                for date in date_range:
                    anomaly_data.append({
                        'country': seq_info['country'],
                        'date': date,
                        'anomaly': 1
                    })

        # Create a DataFrame of anomalous date-country combinations
        if anomaly_data:
            anomalous_dates = pd.DataFrame(anomaly_data)[['country', 'date', 'anomaly']].drop_duplicates()

            # Create a composite key for merging
            df_clean['_merge_key'] = df_clean['country'].astype(str) + '_' + df_clean['date'].dt.strftime('%Y-%m-%d')
            anomalous_dates['_merge_key'] = anomalous_dates['country'].astype(str) + '_' + anomalous_dates['date'].dt.strftime('%Y-%m-%d')

            # Merge to mark anomalies
            anomaly_map = anomalous_dates.set_index('_merge_key')['anomaly'].to_dict()
            df_clean['unsupervised_anomaly'] = df_clean['_merge_key'].map(anomaly_map).fillna(0).astype(int)

            # Clean up
            df_clean = df_clean.drop('_merge_key', axis=1)
    
    # Save results
    output_path = 'data/processed/unsupervised_anomaly_results.csv'
    anomaly_df.to_csv(output_path, index=False)
    print(f"\nUnsupervised anomaly results saved to {output_path}")
    
    return anomaly_df


def simple_unsupervised_approach(df, features):
    """
    Simple unsupervised approach using isolation forest for comparison
    """
    print("Using simple unsupervised approach...")
    
    from sklearn.ensemble import IsolationForest
    
    # Prepare the data
    df_clean = df.copy()
    df_clean[features] = df_clean[features].fillna(df_clean[features].median())
    
    # Handle any remaining infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features)
    
    if df_clean.empty:
        print("No data after cleaning. Exiting.")
        return
    
    # Fit isolation forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Mark 5% as anomalies
    anomaly_predictions = iso_forest.fit_predict(df_clean[features])
    
    # Convert to binary anomaly indicator (1 for anomaly, 0 for normal)
    df_clean['unsupervised_anomaly'] = (anomaly_predictions == -1).astype(int)
    
    print(f"Detected {df_clean['unsupervised_anomaly'].sum()} anomalies out of {len(df_clean)} records")
    print(f"Anomaly rate: {df_clean['unsupervised_anomaly'].mean():.4f}")
    
    # Show top anomalous dates by country
    top_anomalies = df_clean[df_clean['unsupervised_anomaly'] == 1].sort_values('date').groupby('country').head(5)
    print(f"\nTop anomalous periods by country:")
    for country in top_anomalies['country'].unique()[:5]:
        country_anomalies = top_anomalies[top_anomalies['country'] == country].head(3)
        print(f"\nCountry: {country}")
        for _, row in country_anomalies.iterrows():
            print(f"  Date: {row['date']} | Features: {[f'{f}:{row[f]:.2f}' for f in features[:2]]}")


if __name__ == "__main__":
    print("Running Unsupervised Anomaly Detection for Censorship Detection")
    print("=" * 65)
    
    results = run_unsupervised_anomaly_detection()
    print("\nUnsupervised anomaly detection completed successfully!")
