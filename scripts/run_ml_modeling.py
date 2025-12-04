import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# For plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the preprocessed data
df = pd.read_csv('data/processed/preprocessed_for_ml.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.shape[1]}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Countries: {df['country'].nunique()}")

# Convert date column back to datetime
df['date'] = pd.to_datetime(df['date'])

print(f"\nSample of data:")
print(df.head(3))

# Check for potential target variables
target_cols = [col for col in df.columns if 'censorship_indicator' in col]
print(f"\nPotential target columns: {target_cols}")

for target_col in target_cols:
    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        print(f"\n{target_col} distribution:")
        print(target_counts)
        print(f"Positive ratio: {df[target_col].mean():.6f}")

# If we don't have positive censorship events, we'll need to create a binary classification problem
# based on significant drops in connectivity metrics
if 'censorship_indicator' in df.columns and df['censorship_indicator'].sum() == 0:
    print("\nNo censorship events found, creating target based on significant drops...")
    
    # Create a new target variable based on significant changes
    if 'foreign_neighbours_share' in df.columns:
        # Calculate a rolling median for each country to establish baseline
        df['foreign_conn_baseline'] = df.groupby('country')['foreign_neighbours_share'].transform(
            lambda x: x.rolling(window=30, min_periods=7).median()
        )
        
        # Create binary target for significant drops
        df['censorship_target'] = (
            (df['foreign_neighbours_share'] < df['foreign_conn_baseline'] * 0.5) &
            (df['foreign_neighbours_share'].notna()) &
            (df['foreign_conn_baseline'].notna())
        ).astype(int)
        
        target_col = 'censorship_target'
    elif 'asn_count' in df.columns:
        # Calculate rolling median for ASN count
        df['asn_count_baseline'] = df.groupby('country')['asn_count'].transform(
            lambda x: x.rolling(window=30, min_periods=7).median()
        )
        
        # Create binary target for significant drops
        df['censorship_target'] = (
            (df['asn_count'] < df['asn_count_baseline'] * 0.5) &
            (df['asn_count'].notna()) &
            (df['asn_count_baseline'].notna())
        ).astype(int)
        
        target_col = 'censorship_target'
    else:
        # If no suitable metric is found, use a generic approach
        # Select a numeric column with sufficient variation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove date and potential target columns
        exclude_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter', 'is_weekend']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Use the first suitable column to create a binary classification problem
        for col in feature_cols:
            if df[col].std() > 0 and df[col].notna().sum() > 1000:  # Only use columns with good variance and sufficient data
                median_val = df[col].median()
                df['censorship_target'] = (df[col] < median_val).astype(int)
                target_col = 'censorship_target'
                print(f"Created target from column {col}")
                break

# Define features to exclude
exclude_cols = ['date', 'country', target_col]  # Exclude target column
# Add any other columns that shouldn't be features
exclude_cols.extend([col for col in df.columns if 'baseline' in col])

# Select only numeric features
feature_cols = [col for col in df.columns 
                if col not in exclude_cols 
                and df[col].dtype in ['int64', 'float64']]

print(f"\nNumber of features selected: {len(feature_cols)}")
print(f"Features: {feature_cols[:15]}... ({'truncated' if len(feature_cols) > 15 else 'complete'})")

# Prepare feature matrix X and target vector y
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Positive target ratio: {y.mean():.6f}")

# Handle missing values by filling with median for numerical columns
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if pd.notnull(median_val) else 0)

# Double-check for any remaining NaN values and handle them
X = X.fillna(0)  # Fill any remaining NaNs with 0 as a fallback

# Calculate class weights to handle imbalance
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))

print(f"\nClass distribution:")
for cls in classes:
    count = (y == cls).sum()
    print(f"  Class {cls}: {count} samples ({count/len(y)*100:.2f}%)")

print(f"\nClass weights: {class_weight_dict}")

# Add date back to X for time-based splitting
X_with_date = X.copy()
X_with_date['date'] = df['date']

# Sort by date to ensure chronological order
X_sorted = X_with_date.sort_values('date')
y_sorted = y.loc[X_sorted.index]

# Remove date column from features before modeling
X_final = X_sorted.drop('date', axis=1)

# Use the last 20% for testing (time-based split)
split_idx = int(len(X_final) * 0.8)

X_train = X_final.iloc[:split_idx]
X_test = X_final.iloc[split_idx:]
y_train = y_sorted.iloc[:split_idx]
y_test = y_sorted.iloc[split_idx:]

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Positive cases in training: {y_train.sum()} ({y_train.mean():.4f} ratio)")
print(f"Positive cases in test: {y_test.sum()} ({y_test.mean():.4f} ratio)")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        class_weight=class_weight_dict,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1,
        min_samples_split=10,
        min_samples_leaf=5
    )
}

# Initialize scaler for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled features for Logistic Regression, original for Random Forest
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"{name} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    {cm}")

# Detailed classification reports
for name in models.keys():
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, results[name]['y_pred'], zero_division=0))

# Feature importance for Random Forest
rf_model = models['Random Forest']

# Get feature importances
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_final.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_df.head(15))

print("\nCENSORSHIP DETECTION MODELING RESULTS")
print("=" * 50)
print(f"Dataset shape: {df.shape}")
print(f"Features used: {len(feature_cols)}")
print(f"Target variable: {target_col}")
print(f"Positive cases: {y.sum()} ({y.mean():.4f} ratio)")
print(f"Time range: {df['date'].min()} to {df['date'].max()}")
print(f"Countries: {df['country'].nunique()}")
print()

for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    print(f"  - AUC: {metrics['auc']:.4f}")
    print()

print("\nML modeling completed successfully!")