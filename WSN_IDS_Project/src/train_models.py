import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_individual_models(data_path="data/preprocessed_dataset.csv", models_dir="models", results_dir="results"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("="*50)
    print("PHASE 4: TRAIN INDIVIDUAL LIGHTWEIGHT MODELS")
    print("="*50)

    # 1. Load Preprocessed Data
    try:
        df = pd.read_csv(data_path)
        print(f"[+] Successfully loaded preprocessed dataset from {data_path}")
    except FileNotFoundError:
        print(f"[-] ERROR: Dataset not found at {data_path}. Please complete Phase 3 first.")
        sys.exit(1)

    # The target column is standardized as 'Target' from Phase 3
    if 'Target' not in df.columns:
        print("[-] ERROR: 'Target' column not found in dataset. Ensure Phase 3 preprocessing is correct.")
        sys.exit(1)

    X = df.drop(columns=['Target'])
    y = df['Target']

    # 2. Train-Test Split (80% Train, 20% Test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n[+] Dataset split successfully:")
    print(f"    Training instances: {X_train.shape[0]}")
    print(f"    Testing instances:  {X_test.shape[0]}")

    # Initialize Models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    results_list = []
    
    # Text file for detailed reports
    report_file = os.path.join(results_dir, "phase4_training_reports.txt")
    with open(report_file, "w") as f:
        f.write("=== PHASE 4 MODEL TRAINING REPORTS ===\n\n")

        print("\n[+] Training and Evaluating Models...\n")
        print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        print("-" * 75)

        for name, model in models.items():
            # 3. Train Model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # 4. Calculate Metrics (using macro avg for multi-class/binary resilience)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Save results for comparison table
            results_list.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

            # Print to terminal table
            print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

            # 5. Generate Classification Report and write to file
            report = classification_report(y_test, y_pred, zero_division=0)
            f.write(f"--- {name} ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Classification Report:\n{report}\n\n")

            # 6. Save the trained model
            model_filename = name.lower().replace(" ", "_").replace("-", "") + ".pkl"
            if name == "Gaussian Naive Bayes":
                model_filename = "naive_bayes.pkl" # Specific requested naming
                
            model_path = os.path.join(models_dir, model_filename)
            joblib.dump(model, model_path)

    print("-" * 75)
    print(f"\n[+] Detailed classification reports saved to {report_file}")
    
    # Save the comparison table to CSV
    results_df = pd.DataFrame(results_list)
    comparison_path = os.path.join(results_dir, "phase4_model_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    print(f"[+] Model comparison metrics saved to {comparison_path}")
    print("\nPhase 4 Completed successfully.")

if __name__ == "__main__":
    train_individual_models()
