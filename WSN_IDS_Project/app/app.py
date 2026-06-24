import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_system_artifacts, predict_network_traffic

# Set Page Config
st.set_page_config(page_title="WSN IDS Dashboard", page_icon="🛡️", layout="wide")

def main():
    st.sidebar.title("🛡️ WSN IDS Dashboard")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate to:", 
                            ["🏠 Home", 
                             "📊 Dataset Analysis", 
                             "⚙️ Model Training", 
                             "📈 Model Comparison", 
                             "🔮 Live Prediction"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Final Year Project\nIntrusion Detection in WSNs using Lightweight ML.")

    if page == "🏠 Home":
        st.title("Intrusion Detection in Wireless Sensor Networks")
        st.markdown("### Using Ensemble Lightweight Machine Learning Models")
        st.markdown("---")
        
        st.markdown("""
        ### 📖 Project Overview
        Wireless Sensor Networks (WSNs) are highly vulnerable to various cyber attacks due to their distributed nature and resource constraints. This project implements a robust **Intrusion Detection System (IDS)** designed specifically for WSN environments.
        
        Instead of relying on heavy deep learning models that drain sensor batteries, this system utilizes a highly efficient **Ensemble of Lightweight Machine Learning Models**:
        - 🌳 **Decision Tree**
        - 📊 **Gaussian Naive Bayes**
        - 📍 **K-Nearest Neighbors (KNN)**
        
        By combining these models using a **Hard Voting Classifier**, the system achieves maximum accuracy while maintaining minimal computational overhead.
        
        👈 **Use the sidebar to navigate through the project phases, view data analysis, compare models, and test live predictions!**
        """)

    elif page == "📊 Dataset Analysis":
        st.title("📊 Exploratory Data Analysis (EDA)")
        st.markdown("Understanding the underlying patterns in the network traffic data.")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            try:
                img1 = Image.open(os.path.join("results", "class_distribution.png"))
                st.image(img1, caption="Normal Traffic vs. Malicious Attacks", use_container_width=True)
            except FileNotFoundError:
                st.warning("Chart not found. Run Phase 2 first.")
                
        with col2:
            st.subheader("Feature Correlation")
            try:
                img2 = Image.open(os.path.join("results", "correlation_heatmap.png"))
                st.image(img2, caption="Correlation between network features", use_container_width=True)
            except FileNotFoundError:
                st.warning("Chart not found. Run Phase 2 first.")
                
        st.markdown("---")
        st.subheader("Dataset Summary Text")
        try:
            with open(os.path.join("results", "dataset_summary.txt"), "r") as f:
                st.text(f.read())
        except FileNotFoundError:
            st.warning("Summary not found.")

    elif page == "⚙️ Model Training":
        st.title("⚙️ Model Training Results")
        st.markdown("Detailed classification reports for all independent lightweight models.")
        st.markdown("---")
        
        try:
            with open(os.path.join("results", "phase4_training_reports.txt"), "r") as f:
                report_text = f.read()
                st.text(report_text)
        except FileNotFoundError:
            st.warning("Training report not found. Run Phase 4 first.")

    elif page == "📈 Model Comparison":
        st.title("📈 Model Comparison")
        st.markdown("Comparing the independent models against the Hard Voting Ensemble.")
        st.markdown("---")
        
        # Display CSV table
        st.subheader("Performance Metrics Table")
        try:
            comparison_df = pd.read_csv(os.path.join("results", "final_model_comparison.csv"))
            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='lightgreen'))
        except FileNotFoundError:
            st.warning("Metrics table not found. Run Phase 5 first.")
            
        st.markdown("---")
        st.subheader("Visual Metric Comparisons")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        columns = [c1, c2, c3, c4]
        
        for metric, col in zip(metrics, columns):
            with col:
                try:
                    img = Image.open(os.path.join("results", f"{metric}_comparison.png"))
                    st.image(img, caption=f"{metric.capitalize()} Comparison", use_container_width=True)
                except FileNotFoundError:
                    st.warning(f"{metric} chart not found.")
                    
        st.markdown("---")
        st.subheader("Confusion Matrices (Phase 6)")
        cm1, cm2 = st.columns(2)
        cm3, cm4 = st.columns(2)
        
        models = [("Decision Tree", "decision_tree", cm1), 
                  ("Naive Bayes", "gaussian_naive_bayes", cm2), 
                  ("KNN", "knearest_neighbors", cm3), 
                  ("Ensemble", "ensemble_hard_voting", cm4)]
                  
        for name, filename_prefix, col in models:
            with col:
                try:
                    cm_img = Image.open(os.path.join("results", f"{filename_prefix}_confusion_matrix.png"))
                    st.image(cm_img, caption=f"{name} Confusion Matrix", use_container_width=True)
                except FileNotFoundError:
                    st.warning(f"Confusion matrix for {name} not found.")


    elif page == "🔮 Live Prediction":
        st.title("🔮 Real-Time Intrusion Detection Module")
        st.markdown("Input raw WSN features below to predict if the traffic is a malicious attack.")
        st.markdown("---")
        
        # Try loading models safely
        try:
            model, scaler, encoder = load_system_artifacts("ensemble.pkl", models_dir="models")
            system_ready = True
        except SystemExit:
            st.error("System Artifacts (Models/Scalers) are missing. Please complete Phases 3-5.")
            system_ready = False
            
        if system_ready:
            with st.form("prediction_form"):
                st.subheader("Network Traffic Feature Input")
                
                # Create 3 columns for neat input layout
                f1, f2, f3 = st.columns(3)
                
                # Match these exactly to the 15 features dropped during EDA
                with f1:
                    x_coord = st.number_input("X_Coordinate", value=37.45)
                    initial_energy = st.number_input("Initial_Energy", value=11.38)
                    signal_strength = st.number_input("Signal_Strength", value=-45.50)
                    packet_loss = st.number_input("Packet_Loss_Rate", value=17.54)
                    temp = st.number_input("Temperature", value=-3.85)

                with f2:
                    y_coord = st.number_input("Y_Coordinate", value=37.36)
                    residual_energy = st.number_input("Residual_Energy", value=2.98)
                    noise_level = st.number_input("Noise_Level", value=21.27)
                    net_lifetime = st.number_input("Network_Lifetime", value=311.35)
                    humidity = st.number_input("Humidity", value=73.25)

                with f3:
                    z_coord = st.number_input("Z_Coordinate", value=36.49)
                    tx_power = st.number_input("Transmission_Power", value=2.13)
                    energy_cons = st.number_input("Energy_Consumption", value=1.64)
                    learning_rate = st.number_input("Adaptive_Learning_Rate", value=0.018)
                    det_accuracy = st.number_input("Detection_Accuracy", value=87.51)

                submit_button = st.form_submit_button(label="🔍 Predict Traffic Status")
                
            if submit_button:
                # Build dictionary for the prediction script
                input_data = {
                    'X_Coordinate': x_coord,
                    'Y_Coordinate': y_coord,
                    'Z_Coordinate': z_coord,
                    'Initial_Energy': initial_energy,
                    'Residual_Energy': residual_energy,
                    'Transmission_Power': tx_power,
                    'Signal_Strength': signal_strength,
                    'Noise_Level': noise_level,
                    'Energy_Consumption': energy_cons,
                    'Packet_Loss_Rate': packet_loss,
                    'Network_Lifetime': net_lifetime,
                    'Adaptive_Learning_Rate': learning_rate,
                    'Temperature': temp,
                    'Humidity': humidity,
                    'Detection_Accuracy': det_accuracy
                }
                
                with st.spinner("Analyzing Network Traffic..."):
                    pred_class, confidence = predict_network_traffic(input_data, model, scaler, encoder)
                    
                st.markdown("---")
                if pred_class == 1:
                    st.error(f"🚨 **MALICIOUS ATTACK DETECTED** 🚨")
                    st.error(f"Confidence Score: **{confidence:.2f}%**")
                else:
                    st.success(f"✅ **NORMAL TRAFFIC** ✅")
                    st.success(f"Confidence Score: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
