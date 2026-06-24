import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def perform_eda(data_path="data/wsn_dataset.csv", results_dir="results"):
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*50)
    print("PHASE 2: DATASET EXPLORATION AND UNDERSTANDING")
    print("="*50)

    # 1. Load WSN dataset
    try:
        df = pd.read_csv(data_path)
        print(f"[+] Successfully loaded dataset from {data_path}")
    except FileNotFoundError:
        print(f"[-] ERROR: Dataset not found at {data_path}.")
        print("Please place your dataset in the 'data/' folder and name it 'wsn_dataset.csv', or update the path.")
        sys.exit(1)
        
    # Open a text file to save the summary
    summary_file_path = os.path.join(results_dir, "dataset_summary.txt")
    with open(summary_file_path, "w") as f:
        
        # Helper function to print and write to file
        def log(text):
            print(text)
            f.write(text + "\n")
            
        log("\n--- DATASET SHAPE ---")
        log(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        log("\n--- COLUMN NAMES ---")
        log(str(list(df.columns)))
        
        log("\n--- DATA TYPES ---")
        log(str(df.dtypes))
        
        log("\n--- MISSING VALUES ---")
        missing_values = df.isnull().sum()
        log(str(missing_values[missing_values > 0]) if missing_values.sum() > 0 else "No missing values found.")
        
        log("\n--- CLASS DISTRIBUTION ---")
        # We assume the target column is the last column or named 'Label', 'Class', 'Attack', 'target'
        # Let's try to identify the target column
        target_cols = ['label', 'class', 'attack', 'target', 'is_intrusion']
        target_col = None
        for col in df.columns:
            if col.lower() in target_cols:
                target_col = col
                break
                
        # If no standard name is found, assume the last column is the target
        if not target_col:
            target_col = df.columns[-1]
            
        log(f"Assumed Target Column: '{target_col}'")
        class_dist = df[target_col].value_counts()
        log(str(class_dist))
        
        # 2. Generate Exploratory Visualizations
        print("\n[+] Generating Exploratory Visualizations...")
        
        # Class distribution chart
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=target_col, palette='viridis')
        plt.title('Class Distribution (Normal vs. Intrusion)')
        plt.xlabel('Class Label')
        plt.ylabel('Count')
        plt.tight_layout()
        class_dist_path = os.path.join(results_dir, "class_distribution.png")
        plt.savefig(class_dist_path)
        plt.close()
        print(f"[+] Saved class distribution chart to {class_dist_path}")
        
        # Correlation heatmap
        # Select only numerical columns for correlation
        numerical_df = df.select_dtypes(include=[np.number])
        if not numerical_df.empty:
            plt.figure(figsize=(12, 10))
            corr = numerical_df.corr()
            # Plot only a subset of features if there are too many (to avoid clutter)
            if len(numerical_df.columns) > 20:
                print("[!] Many features detected. Plotting correlation heatmap for the first 20 features.")
                corr = numerical_df.iloc[:, :20].corr()
                
            sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            heatmap_path = os.path.join(results_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            print(f"[+] Saved correlation heatmap to {heatmap_path}")
        else:
            print("[-] No numerical features found for correlation heatmap.")

    print(f"\n[+] Dataset summary saved to {summary_file_path}")
    print("\nEDA Completed successfully.")

if __name__ == "__main__":
    perform_eda()
