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

class ChurnPipeline:
    """
    A class-based machine learning pipeline for Telecom Churn Prediction.
    Inspired by eda.ipynb and modularized for production-like usage.
    """
    def __init__(self, filepath="./data/raw/ChurnDataFile.csv"):
        self.filepath = filepath
        self.df = None
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.encoders = {}

    def load_data(self):
        """
        Loads the dataset from the specified filepath.
        """
        if not os.path.exists(self.filepath):
            # Fallback for relative path issues in notebooks vs scripts
            alt_path = os.path.join("..", self.filepath)
            if os.path.exists(alt_path):
                self.filepath = alt_path
            else:
                raise FileNotFoundError(f"DataFile Not Found : {self.filepath}")
                
        self.df = pd.read_csv(self.filepath)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def preprocess(self):
        """
        Processes the raw dataframe:
        - Converts TotalCharges to numeric
        - Imputes missing values
        - Drops irrelevant columns
        - Encodes categorical features
        - Scales numerical features
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.df.copy()
        
        # Handle 'TotalCharges' - conversion to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing with mean as a default (preserving existing logic)
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
        
        # Drop irrelevant columns
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        # Binary encode target : Churn
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            
        # Encode categorical features - LabelEncoder (as per original logic)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
            
        # Final pass for any remaining NaN values
        df = df.fillna(0)
        
        # MinMax scaling for features
        target = None
        if 'Churn' in df.columns:
            target = df['Churn']
            df = df.drop(columns=['Churn'])
            
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        self.scalers['minmax'] = scaler
        
        if target is not None:
            df_scaled['Churn'] = target.reset_index(drop=True)
            
        self.df = df_scaled
        return self.df

    def split(self, target_col='Churn', test_size=0.2):
        """
        Splits the processed dataframe into training and testing sets.
        """
        if self.df is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Trains multiple models (Logistic Regression, RandomForest, SVC).
        """
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVC': SVC(probability=True, random_state=42)
        }
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"Model Trained: {name}")
            
        return self.models

    def evaluate(self, X_test, y_test):
        """
        Evaluates trained models and returns metrics.
        """
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
            }
            
            # Handle ROC AUC - requires probability scores
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except (AttributeError, ValueError):
                metrics["roc_auc"] = "N/A"
            
            self.results[name] = metrics
            print(f"\n--- {name} Results ---")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            print(classification_report(y_test, y_pred))
            
        return self.results

    def save_models(self, directory='models'):
        """
        Saves all trained models to the specified directory using joblib.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name.lower()}_model.joblib")
            joblib.dump(model, path)
            print(f"Saved {name} model to {path}")

    def run(self):
        """
        Executes the full pipeline.
        """
        print("Starting Churn Prediction Pipeline...")
        self.load_data()
        self.preprocess()
        X_train, X_test, y_train, y_test = self.split()
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)
        self.save_models()
        print("Pipeline execution complete.")

if __name__ == "__main__":
    pipeline = ChurnPipeline()
    try:
        pipeline.run()
    except Exception as e:
        print(f"ERROR during pipeline execution: {e}")