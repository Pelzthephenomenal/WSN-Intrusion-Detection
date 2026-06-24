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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_ensemble_model(data_path="data/preprocessed_dataset.csv", models_dir="models", results_dir="results"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("="*50)
    print("PHASE 5: BUILD ENSEMBLE MODEL")
    print("="*50)

    # 1. Load Preprocessed Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[-] ERROR: Dataset not found at {data_path}.")
        sys.exit(1)

    X = df.drop(columns=['Target'])
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build VotingClassifier
    print("[+] Initializing Ensemble Voting Classifier (Hard Voting)...")
    clf1 = DecisionTreeClassifier(random_state=42)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=5)

    ensemble = VotingClassifier(
        estimators=[
            ('dt', clf1),
            ('nb', clf2),
            ('knn', clf3)
        ],
        voting='hard'
    )

    # 3. Train Ensemble
    print("[+] Training Ensemble Model (This may take a moment)...")
    ensemble.fit(X_train, y_train)

    # 4. Evaluate Ensemble
    y_pred = ensemble.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n[+] Ensemble Model Performance:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")

    # 5. Save the Ensemble Model
    ensemble_path = os.path.join(models_dir, "ensemble.pkl")
    joblib.dump(ensemble, ensemble_path)
    print(f"\n[+] Saved Ensemble model to {ensemble_path}")

    # 6. Load Phase 4 metrics and compare
    comparison_path = os.path.join(results_dir, "phase4_model_comparison.csv")
    try:
        results_df = pd.read_csv(comparison_path)
    except FileNotFoundError:
        print("[-] WARNING: Phase 4 comparison metrics not found. Please run Phase 4 first.")
        sys.exit(1)

    # Add ensemble results to dataframe
    ensemble_df = pd.DataFrame([{
        "Model": "Ensemble (Hard Voting)",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }])
    
    combined_results = pd.concat([results_df, ensemble_df], ignore_index=True)
    
    # Save the updated combined table
    final_comparison_path = os.path.join(results_dir, "final_model_comparison.csv")
    combined_results.to_csv(final_comparison_path, index=False)
    print(f"[+] Saved final combined comparison table to {final_comparison_path}")

    # 7. Create Comparison Charts
    print("[+] Generating Performance Comparison Charts...")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        # Ensure 'Ensemble' stands out by using a different color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
        sns.barplot(data=combined_results, x='Model', y=metric, palette=colors)
        plt.title(f'Model Comparison: {metric}')
        plt.ylim(0, 1.1)
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        chart_path = os.path.join(results_dir, f"{metric.lower().replace(' ', '_')}_comparison.png")
        plt.savefig(chart_path)
        plt.close()

    print(f"[+] Comparison charts saved to {results_dir}/")
    print("\nPhase 5 Completed successfully.")

if __name__ == "__main__":
    train_ensemble_model()
