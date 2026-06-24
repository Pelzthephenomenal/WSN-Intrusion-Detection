import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def evaluate_models(data_path="data/preprocessed_dataset.csv", models_dir="models", results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    print("="*50)
    print("PHASE 6: EVALUATION AND VISUALIZATION (CONFUSION MATRICES)")
    print("="*50)

    # 1. Load Preprocessed Data and Split
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[-] ERROR: Dataset not found at {data_path}.")
        sys.exit(1)

    X = df.drop(columns=['Target'])
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Load Models
    model_files = {
        "Decision Tree": "decision_tree.pkl",
        "Gaussian Naive Bayes": "naive_bayes.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Ensemble (Hard Voting)": "ensemble.pkl"
    }

    print("[+] Generating Confusion Matrices for all models...")
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"[-] WARNING: {filename} not found. Skipping {model_name}...")
            continue
            
        # Predict
        y_pred = model.predict(X_test)
        
        # Generate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal (0)', 'Attack (1)'], 
                    yticklabels=['Normal (0)', 'Attack (1)'])
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        cm_path = os.path.join(results_dir, f"{safe_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        print(f"[+] Saved {model_name} confusion matrix to {cm_path}")

    # Note: Accuracy, Precision, Recall, and F1 comparison charts were generated in Phase 5.
    print(f"\n[+] Note: Performance comparison charts were already generated in Phase 5.")
    print(f"[+] All visualizations are saved in the {results_dir}/ directory.")
    print("\nPhase 6 Completed successfully.")

if __name__ == "__main__":
    evaluate_models()
