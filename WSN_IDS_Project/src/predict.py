import pandas as pd
import numpy as np
import joblib
import os
import sys

def load_system_artifacts(model_name="ensemble.pkl", models_dir="models"):
    """Loads the trained model, scaler, and label encoder."""
    try:
        model = joblib.load(os.path.join(models_dir, model_name))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
        return model, scaler, encoder
    except FileNotFoundError as e:
        print(f"[-] ERROR loading artifacts: {e}")
        print("    Ensure you have run Phases 3, 4, and 5 to generate these files.")
        sys.exit(1)

def predict_network_traffic(data_dict, model, scaler, encoder):
    """
    Accepts new network traffic data (as a dictionary),
    applies scaling, and returns the prediction and confidence score.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Scale the features
    scaled_features = scaler.transform(df)
    
    # Get the final prediction
    prediction_encoded = model.predict(scaled_features)[0]
    prediction_label = encoder.inverse_transform([prediction_encoded])[0]
    
    # Calculate Confidence Score
    confidence = 0.0
    
    # Since we used Hard Voting for the ensemble, `predict_proba` is not natively supported.
    # To get a confidence score, we manually tally the votes from the 3 underlying models!
    if hasattr(model, 'estimators_'):
        votes = [clf.predict(scaled_features)[0] for clf in model.estimators_]
        majority_vote_count = votes.count(prediction_encoded)
        confidence = (majority_vote_count / len(votes)) * 100.0
    # For individual models that support predict_proba (like Decision Tree or KNN)
    elif hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = np.max(probabilities) * 100.0
    else:
        # Fallback if no probability is available
        confidence = 100.0

    return prediction_label, confidence

if __name__ == "__main__":
    print("="*50)
    print("PHASE 7: PREDICTION MODULE TEST")
    print("="*50)
    
    # 1. Load the best model (Ensemble)
    print("[+] Loading model and artifacts...")
    model, scaler, encoder = load_system_artifacts("ensemble.pkl")
    
    # 2. Define a sample piece of network traffic (dummy data matching the 15 feature columns)
    # This simulates real-time data arriving from the Wireless Sensor Network
    sample_traffic = {
        'X_Coordinate': 37.45,
        'Y_Coordinate': 37.36,
        'Z_Coordinate': 36.49,
        'Initial_Energy': 11.38,
        'Residual_Energy': 2.98,
        'Transmission_Power': 2.13,
        'Signal_Strength': -45.50,
        'Noise_Level': 21.27,
        'Energy_Consumption': 1.64,
        'Packet_Loss_Rate': 17.54,
        'Network_Lifetime': 311.35,
        'Adaptive_Learning_Rate': 0.018,
        'Temperature': -3.85,
        'Humidity': 73.25,
        'Detection_Accuracy': 87.51
    }
    
    print("\n[+] Received New Network Traffic Data:")
    for k, v in sample_traffic.items():
        print(f"    {k}: {v}")
        
    print("\n[+] Processing and Predicting...")
    predicted_class, confidence = predict_network_traffic(sample_traffic, model, scaler, encoder)
    
    # Format the output professionally
    status = "MALICIOUS ATTACK DETECTED" if predicted_class == 1 else "NORMAL TRAFFIC"
    
    print("-" * 50)
    print(f"Prediction Result : {status} (Class {predicted_class})")
    print(f"Confidence Score  : {confidence:.2f}%")
    print("-" * 50)
    
    print("\nPhase 7 Completed successfully.")
