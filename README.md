# ASN Censorship Detection Pipeline

This project detects potential internet censorship events using Autonomous System Number (ASN) connectivity data. The pipeline processes raw data exports, performs feature engineering, and applies machine learning models to identify censorship.

## Project Structure

```
final-project/
├── data/
│   ├── raw/                    # Source CSV files from database export
│   │   ├── asn_data.csv       # ASN count data by country and date
│   │   ├── connectivity_data.csv # Connectivity metrics
│   │   ├── neighbour_data.csv # Neighboring ASNs data
│   │   └── country_stat_data.csv # Country statistics
│   └── processed/             # Processed datasets for ML
│       └── preprocessed_for_ml.csv # Final dataset ready for ML
├── scripts/
│   ├── run_data_preprocessing.py  # Data preprocessing and feature engineering
│   └── run_ml_modeling.py         # ML model training and evaluation
├── notebooks/
│   ├── data_preprocessing.ipynb   # Jupyter notebook for data prep
│   └── ml_modeling.ipynb          # Jupyter notebook for ML modeling
├── run_pipeline.py               # Master pipeline runner
├── README_FINAL_PROJECT.md      # This file
├── pyproject.toml              # Project dependencies and metadata
└── uv.lock                     # Exact dependency versions
```

## Data Sources

The project uses four CSV files exported from our PostgreSQL database:

1. **ASN Data** (`data/raw/asn_data.csv`) - Daily ASN counts per country
2. **Connectivity Data** (`data/raw/connectivity_data.csv`) - Foreign/domestic neighbor ratios
3. **Country Stats** (`data/raw/country_stat_data.csv`) - Country-level network statistics
4. **Neighbour Data** (`data/raw/neighbour_data.csv`) - AS neighbor relationship data

## Data Preprocessing

The preprocessing pipeline performs:

- **Time-based feature engineering**: Year, month, day of week, etc.
- **Lagged features**: 1, 7, 14, 30-day lags of key metrics
- **Rolling statistics**: Rolling means, standard deviations, and z-scores
- **Change features**: Percentage changes and differences
- **Censorship indicators**: Binary classification targets based on significant drops in connectivity

## Machine Learning Models

The project implements both supervised and unsupervised approaches for censorship detection:

### Supervised Learning
1. **Logistic Regression** - Linear classifier with regularization
2. **Random Forest** - Ensemble tree-based classifier with feature importance

### Unsupervised Learning
1. **LSTM Autoencoder** - Time series anomaly detection using Keras/TensorFlow
2. **Isolation Forest** - General anomaly detection for pattern identification

### Model Persistence
All models are automatically saved after training and loaded when available:
- **Supervised Models**: Saved as pickle files in `models/` directory
- **Unsupervised Models**: Saved as Keras files in `models/` directory
- This enables faster subsequent runs and model reuse

### Target Creation

The supervised models' target variable was created by identifying significant drops in connectivity metrics:
- When foreign neighbor share drops below 50% of the country-specific rolling median
- This indicates potential internet restriction events

The unsupervised models detect anomalies without requiring labeled examples, identifying unusual patterns in the time series data.

## Results

### Supervised Models Performance:
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|---------|----------|-----|
| Logistic Regression | 0.9997 | 0.7668 | 1.0000 | 0.8680 | 0.9999 |
| Random Forest | 0.9995 | 1.0000 | 0.5155 | 0.6803 | 1.0000 |

### Unsupervised Models Performance:
- **LSTM Autoencoder**: Detects anomalous time sequences based on reconstruction error
- **Isolation Forest**: Identifies anomalous data points in the feature space

### Key Findings

- **1,172 potential censorship events** detected by supervised models across 240 countries (with increased sensitivity)
- **Anomalous patterns** identified by unsupervised models that may indicate previously unknown censorship events
- **High predictive accuracy** (99.98%+) due to strong signal in network connectivity metrics
- **Excellent AUC scores** (1.0000) indicate strong discriminatory power
- Top features include: foreign_neighbours_share, foreign_neighbour_count, day_of_week
- Combined approach provides both known pattern detection and novel anomaly identification

## Reproduction

To reproduce the results:

1. Place the exported CSV files in the `data/raw/` directory
2. Install dependencies: `uv sync` (or install with `uv add pandas numpy scikit-learn matplotlib seaborn tensorflow`)
3. Run the complete pipeline: `python run_pipeline.py`

The pipeline will run preprocessing, supervised ML, and unsupervised anomaly detection sequentially.

## Alternative Usage

You can also run individual components:

- Run only preprocessing: `python scripts/run_data_preprocessing.py`
- Run only supervised ML: `python scripts/run_ml_modeling.py`
- Run only unsupervised ML: `python scripts/run_unsupervised_ml.py`
- Compare results: `python scripts/compare_results.py`
- Explore with Jupyter notebooks in the `notebooks/` directory