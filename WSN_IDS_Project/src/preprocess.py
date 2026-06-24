import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import sys

def preprocess_data(data_path="data/WSN_Dataset.csv", output_path="data/preprocessed_dataset.csv", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    
    print("="*50)
    print("PHASE 3: DATA PREPROCESSING")
    print("="*50)

    # 1. Load Dataset
    try:
        df = pd.read_csv(data_path)
        print(f"[+] Loaded dataset from {data_path}")
    except FileNotFoundError:
        print(f"[-] ERROR: Dataset not found at {data_path}.")
        sys.exit(1)

    print("\n--- BEFORE PREPROCESSING ---")
    print(f"Shape: {df.shape}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # 2. Handle missing values
    # For numerical columns, we fill with the median. For categorical, we fill with the mode.
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 3. Remove duplicates
    df.drop_duplicates(inplace=True)

    # Automatically identify target column or default to the last one
    target_candidates = ['label', 'class', 'attack', 'target', 'is_intrusion', 'optimization_algorithm']
    target_col = None
    for col in df.columns:
        if col.lower() in target_candidates:
            target_col = col
            break
            
    if not target_col:
        target_col = df.columns[-1]

    print(f"\n[+] Identified Target Column: '{target_col}'")
    if df[target_col].nunique() > 20:
        print(f"\n[!] WARNING: Target column '{target_col}' has {df[target_col].nunique()} unique values.")
        print("    This looks like a continuous variable (regression), but you are building a classification model.")
        print("    If this is incorrect, please open src/preprocess.py and manually set the target_col variable.")

    # Optional: Drop irrelevant columns like Node_ID or Timestamp to prevent the model from learning noise
    cols_to_drop = [col for col in df.columns if col.lower() in ['node_id', 'timestamp', 'id']]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"[+] Dropped irrelevant identifier/datetime columns: {cols_to_drop}")

    # 4. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save the label encoder
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, le_path)
    print(f"[+] Saved LabelEncoder to {le_path}")
    print(f"[+] Encoded Classes: {list(label_encoder.classes_)}")

    # Ensure all features are numerical before scaling. Apply one-hot encoding if categorical features exist.
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"[+] One-Hot Encoding categorical features: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 6. Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved StandardScaler to {scaler_path}")

    # Convert scaled features back to DataFrame to save them properly
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Re-attach the encoded target column
    preprocessed_df = X_scaled_df.copy()
    preprocessed_df['Target'] = y_encoded  # standardize target name to 'Target' for the next phases

    # Save the completely preprocessed dataset
    preprocessed_df.to_csv(output_path, index=False)
    print(f"[+] Saved fully preprocessed dataset to {output_path}")

    print("\n--- AFTER PREPROCESSING ---")
    print(f"Shape: {preprocessed_df.shape}")
    print(f"Missing Values: {preprocessed_df.isnull().sum().sum()}")
    print(f"Duplicates: {preprocessed_df.duplicated().sum()}")
    print("\nPreprocessing Completed successfully.")

if __name__ == "__main__":
    preprocess_data()
