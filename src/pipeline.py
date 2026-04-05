import pandas as pd
import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, roc_auc_score

def load_data(filepath="./data/df_churn.csv"):
    """
    Loads dataset from filepath. 
    Handles relative path fallback for use in notebooks vs scripts.
    """
    if not os.path.exists(filepath):
        # Fallback for relative path challenges
        alt_path = os.path.join("..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"DataFile Not Found : {filepath}")
            
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Processes the dataframe:
    - Numeric conversion for TotalCharges
    - Imputes missing values
    - Drops customerID
    - Encodes categorical features
    - Scales numerical features
    """
    df_clean = df.copy()
    
    # TotalCharges: coerce to numeric and fill NaN with mean
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].mean())
    
    # Drop customerID if present
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    # Binary encode target : Churn (Yes/No -> 1/0)
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
    # Encode categorical features using LabelEncoder
    encoders = {}
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
        
    # Final pass for any remaining NaN values
    df_clean = df_clean.fillna(0)
    
    # Scaling: MinMaxScaler for all features
    target = None
    if 'Churn' in df_clean.columns:
        target = df_clean['Churn']
        df_clean = df_clean.drop(columns=['Churn'])
        
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
    
    if target is not None:
        df_scaled['Churn'] = target.reset_index(drop=True)
        
    print(f"Data preprocessed successfully.")
    return df_scaled, encoders, scaler

def split_data(df, target_col='Churn', test_size=0.2):
    """
    Splits the dataframe into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Trains multiple models (Logistic Regression, RandomForest, SVC).
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Model Trained: {name}")
        
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluates models and returns a dictionary of metrics.
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except (AttributeError, ValueError):
            metrics["roc_auc"] = "N/A"
            
        results[name] = metrics
        print(f"\n--- {name} Results ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print(classification_report(y_test, y_pred))
            
    return results

def save_trained_models(models, directory='models'):
    """
    Saves trained models to the specified directory using joblib.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
            
    for name, model in models.items():
        path = os.path.join(directory, f"{name.lower()}_model.joblib")
        joblib.dump(model, path)
        print(f"Saved {name} model to {path}")

def run_pipeline():
    """
    Executes the full machine learning flow.
    """
    print("Starting Machine Learning Pipeline...")
    try:
        raw_df = load_data()
        processed_df, encoders, scaler = preprocess_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(processed_df)
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        models = train_models(X_train, y_train)
        evaluate_models(models, X_test, y_test)
        save_trained_models(models)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"ERROR during pipeline execution: {e}")

if __name__ == "__main__":
    run_pipeline()