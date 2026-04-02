import pytest
import pandas as pd
import numpy as np
from src.pipeline import preprocess_data, split_data

# Fixture : dummy data --> testing
@pytest.fixture

def dummy_data():
    # liste(5)*4 -> 20 lines => test_test_split
    data = {
        'customerID': [str(i) for i in range(20)],
        'TotalCharges': ['100', '200', ' ', '400', '500'] * 4, 
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'] * 4,
        'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes'] * 4
    }
    return pd.DataFrame(data)

def test_preprocess_data(dummy_data):

    df_clean = preprocess_data(dummy_data)  # cleaning data
    
    # No CustomerID
    assert 'customerID' not in df_clean.columns
    
    # Churn=[0,1]
    assert df_clean['Churn'].isin([0, 1]).all()
    
    # No missed values
    assert df_clean.isnull().sum().sum() == 0
    
    # TotalCharges --> numeric
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])

# --- split <=> dimensions
def test_split_dimensions(dummy_data):

    df_clean = preprocess_data(dummy_data)
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2)
    
    # features <=> target
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # split daat - total length 
    assert len(X_train) + len(X_test) == len(df_clean)