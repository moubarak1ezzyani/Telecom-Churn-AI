import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, roc_auc_score

# --- loading data & handle paths
def load_data(filepath="./data/raw/ChurnDataFile.csv"):
    if not os.path.exists(filepath):
        filepath = os.path.join("..", filepath)
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DataFile Not Found : {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. : {df.shape}")
    return df

# --- clean & encode/normalize data.
def preprocess_data(df):
    df = df.copy() 
    
    # -> [TotalCharges] --> numeric 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # -> Drop irrelevant cols
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # -> Binary encode target : Churn
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # -> Encode categorica features - LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()     # production : OneHotEncoder 
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # -> MinMax scaling
    target = None
    if 'Churn' in df.columns:
        target = df['Churn']
        df = df.drop(columns=['Churn'])
        
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    if target is not None:
        df_scaled['Churn'] = target.reset_index(drop=True)      # Restore target (if present)
        
    return df_scaled

# --- splitting data
def split_data(df, target_col='Churn', test_size=0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --- training data
def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Model Trained : {name}")
        
    return trained_models

# --- Evaluate model & performance metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    
    # Handle ROC AUC - proba scores
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except:
        metrics["roc_auc"] = "N/A"
        
    return metrics

# =======================
# Main execution block
# =======================
if __name__ == "__main__":      # pyhton pipeline.py
    try:
        # -> loading data
        df = load_data()
        
        # -> cleaning 
        df_clean = preprocess_data(df)
        
        # -> Split
        X_train, X_test, y_train, y_test = split_data(df_clean)
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # -> training 
        models = train_models(X_train, y_train)
        
        # -> Evaluation & Selection
        print("\nResults :")
        for name, model in models.items():
            res = evaluate_model(model, X_test, y_test)
            print(f"--- {name} ---")
            print(res)
            
    except Exception as e:
        print(f"ERROR : {e}")