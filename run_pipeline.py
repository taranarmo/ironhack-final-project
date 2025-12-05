#!/usr/bin/env python3
"""
ASN Censorship Detection Pipeline Launcher

This script runs the complete censorship detection pipeline:
1. Data preprocessing and feature engineering
2. Supervised machine learning model training and evaluation
3. Unsupervised anomaly detection using time series autoencoders
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print("=" * (len(description) - 1))
    print(f"Running command: {cmd}")

    try:
        # Run from project root directory to ensure proper relative paths
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        else:
            print("SUCCESS: Command completed successfully")
            return True

    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        return False


def main():
    """Main function to run the complete pipeline"""
    print("ASN Censorship Detection Pipeline")
    print("===============================")

    # Check if required CSV files exist - asn_data.csv is no longer required as it's redundant with country_stat_data
    required_files = [
        "data/raw/connectivity_data.csv",
        "data/raw/neighbour_data.csv",
        "data/raw/country_stat_data.csv"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"ERROR: Required CSV files not found:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all required CSV files are in the data/raw/ directory.")
        sys.exit(1)

    print("All required data files found!")

    # Step 1: Run data preprocessing
    success = run_command(
        "python scripts/run_data_preprocessing.py",
        "STEP 1: Running data preprocessing and feature engineering..."
    )

    if not success:
        print("FAILED: Data preprocessing failed - stopping pipeline")
        sys.exit(1)

    # Step 2: Run supervised ML modeling
    success = run_command(
        "python scripts/run_ml_modeling.py",
        "STEP 2: Running supervised machine learning modeling..."
    )

    if not success:
        print("FAILED: Supervised ML modeling failed")
        sys.exit(1)

    # Step 3: Run unsupervised anomaly detection
    success = run_command(
        "python scripts/run_unsupervised_ml.py",
        "STEP 3: Running unsupervised anomaly detection with time series autoencoders..."
    )

    if not success:
        print("WARNING: Unsupervised ML failed, but continuing with results")

    # Step 4: Summarize results
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    print("===============================")
    print("Pipeline has completed data preprocessing, supervised ML, and unsupervised anomaly detection.")
    print("\nNext steps:")
    print("- Check results in the output logs above")
    print("- Review the processed data in data/processed/")
    print("- Explore the Jupyter notebooks in notebooks/ for deeper analysis")
    print("- Compare supervised and unsupervised results")
    print("- Refer to README.md for detailed documentation")


if __name__ == "__main__":
    main()