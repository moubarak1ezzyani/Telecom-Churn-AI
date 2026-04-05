import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import ChurnPipeline

@pytest.fixture
def sample_data():
    """Generates dummy data for testing."""
    data = {
        'customerID': [str(i) for i in range(100)],
        'gender': ['Male', 'Female'] * 50,
        'SeniorCitizen': [0, 1] * 50,
        'Partner': ['Yes', 'No'] * 50,
        'Dependents': ['No', 'Yes'] * 50,
        'tenure': np.random.randint(1, 72, 100),
        'PhoneService': ['Yes', 'No'] * 50,
        'MultipleLines': ['No', 'Yes'] * 50,
        'InternetService': ['DSL', 'Fiber optic'] * 50,
        'OnlineSecurity': ['No', 'Yes'] * 50,
        'OnlineBackup': ['Yes', 'No'] * 50,
        'DeviceProtection': ['No', 'Yes'] * 50,
        'TechSupport': ['No', 'Yes'] * 50,
        'StreamingTV': ['Yes', 'No'] * 50,
        'StreamingMovies': ['No', 'Yes'] * 50,
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'] * 25,
        'PaperlessBilling': ['Yes', 'No'] * 50,
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'] * 25,
        'MonthlyCharges': np.random.uniform(18, 118, 100),
        'TotalCharges': [str(x) for x in np.random.uniform(18, 5000, 100)],
        'Churn': ['Yes', 'No'] * 50
    }
    # Sprinkle some problematic values in TotalCharges
    data['TotalCharges'] = list(data['TotalCharges']) # ensure list for indexing
    data['TotalCharges'][5] = ' '
    data['TotalCharges'][10] = ''
    
    return pd.DataFrame(data)

def test_pipeline_load_data(tmp_path, sample_data):
    """Test if the pipeline can load data from a CSV file."""
    csv_file = tmp_path / "test_churn.csv"
    sample_data.to_csv(csv_file, index=False)
    
    pipeline = ChurnPipeline(filepath=str(csv_file))
    df = pipeline.load_data()
    assert df.shape == (100, 21)
    assert not df.empty

def test_pipeline_preprocess(sample_data):
    """Test if the preprocessing steps work as expected."""
    pipeline = ChurnPipeline()
    pipeline.df = sample_data.copy()
    
    processed_df = pipeline.preprocess()
    
    # Check if 'customerID' is dropped
    assert 'customerID' not in processed_df.columns
    
    # Check if 'TotalCharges' is numeric
    assert pd.api.types.is_numeric_dtype(processed_df['TotalCharges'])
    
    # Check if 'Churn' is encoded to 0 and 1
    assert set(processed_df['Churn'].unique()).issubset({0, 1})
    
    # Check for missing values
    assert processed_df.isnull().sum().sum() == 0

def test_pipeline_split(sample_data):
    """Test the data splitting functionality."""
    pipeline = ChurnPipeline()
    pipeline.df = sample_data.copy()
    pipeline.preprocess()
    
    X_train, X_test, y_train, y_test = pipeline.split(test_size=0.2)
    
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_pipeline_run_integration(tmp_path, sample_data):
    """Run an integration test of the full pipeline on dummy data."""
    # Setup temp file
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    
    models_dir = tmp_path / "test_models"
    
    pipeline = ChurnPipeline(filepath=str(csv_file))
    
    # Call methods in sequence
    pipeline.load_data()
    pipeline.preprocess()
    X_train, X_test, y_train, y_test = pipeline.split()
    pipeline.train(X_train, y_train)
    results = pipeline.evaluate(X_test, y_test)
    pipeline.save_models(directory=str(models_dir))
    
    # Broad check for results
    assert 'LogisticRegression' in results
    assert 'RandomForest' in results
    assert 'SVC' in results
    
    # Check if model files were saved
    assert os.path.exists(os.path.join(str(models_dir), "logisticregression_model.joblib"))
    assert os.path.exists(os.path.join(str(models_dir), "randomforest_model.joblib"))
    assert os.path.exists(os.path.join(str(models_dir), "svc_model.joblib"))
