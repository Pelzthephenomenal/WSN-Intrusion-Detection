# Intrusion Detection in Wireless Sensor Networks Using Ensemble Lightweight Machine Learning Models

## Project Overview
This project aims to detect normal and malicious traffic in a Wireless Sensor Network (WSN) using lightweight machine learning models (Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors) and an ensemble Voting Classifier.

## Folder Structure

The project is structured into the following directories:

- **`data/`**: Stores the raw and preprocessed datasets. All WSN network traffic data should be placed here.
- **`src/`**: Contains the core Python scripts for data preprocessing, model training, prediction, and other utilities. This is the main codebase for the machine learning pipeline.
- **`models/`**: Used to save the trained machine learning models (`.pkl` files) such as the individual classifiers, the ensemble model, and preprocessing artifacts like scalers or label encoders.
- **`results/`**: Stores the outputs of our experiments, including visualizations, correlation heatmaps, class distribution charts, and confusion matrices.
- **`app/`**: Contains the Streamlit application code. This provides a professional dashboard for exploring data, training models, and predicting network intrusions.
- **`notebooks/`**: For Jupyter notebooks used in preliminary exploratory data analysis (EDA), scratchpad testing, and interactive prototyping.

## Virtual Environment Setup Instructions

To ensure dependencies are isolated and do not interfere with system-wide packages, it is recommended to use a Python virtual environment. Follow these steps to set it up:

1. **Open your terminal or command prompt.**
2. **Navigate to the project root directory (`WSN_IDS_Project/`).**
3. **Create the virtual environment** (named `venv`):
   ```bash
   python -m venv venv
   ```
4. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
5. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Flow Through the System

The workflow of the data through our intrusion detection system follows these key stages:

1. **Data Ingestion (`data/`)**: The raw WSN traffic dataset is loaded into the system. This dataset contains features representing network activity and labels indicating whether the traffic is 'normal' or a specific type of 'attack'.
2. **Data Preprocessing (`src/`)**: 
   - Missing values are handled.
   - Duplicates are removed.
   - Target labels are encoded into numerical formats (e.g., normal=0, attack=1).
   - Numerical features are scaled to ensure lightweight models (like KNN) perform optimally.
3. **Model Training (`src/` -> `models/`)**: 
   - The preprocessed dataset is split into training and testing sets.
   - Individual lightweight models (Decision Tree, Naive Bayes, KNN) learn patterns from the training set.
   - An ensemble Voting Classifier combines these models for a robust final prediction.
   - Trained models and preprocessing components (scalers, encoders) are saved as artifacts in the `models/` directory.
4. **Evaluation and Visualization (`src/` -> `results/`)**: 
   - The models are evaluated on the testing set.
   - Metrics (Accuracy, Precision, Recall, F1 Score) and plots (Confusion Matrices, Comparison Charts) are generated and saved to the `results/` directory.
5. **Prediction and Dashboard (`app/` & `src/predict.py`)**: 
   - New, unseen network traffic data can be fed into the system via the Streamlit dashboard or the prediction module.
   - The system loads the saved models/scalers and processes the new data.
   - It outputs a prediction (Normal vs. Intrusion) along with a confidence score to the user interface.
