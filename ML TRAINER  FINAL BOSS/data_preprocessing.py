import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_and_preprocess_data():
    """
    Load and preprocess the heart disease dataset.
    
    Returns:
        tuple: (data, X, y, feature_names, target_name)
    """
    # Load the heart disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names based on documentation
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    data = pd.read_csv(url, names=column_names, na_values='?')
    
    # Check for missing values and handle them
    if data.isnull().sum().sum() > 0:
        # Fill missing values with the median for numerical columns
        for col in data.columns:
            if data[col].dtype != 'object':
               data[col] = data[col].fillna(data[col].median())

    
    # Clean and prepare the data
    
    # Convert the target to binary (0: no disease, 1: disease)
    data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)
    
    # Feature preprocessing
    # 1. Sex: 0 = female, 1 = male
    # 2. CP (chest pain type): 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
    # 3. FBS (fasting blood sugar > 120 mg/dl): 0 = false, 1 = true
    # 4. RestECG: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy
    # 5. ExAng (exercise induced angina): 0 = no, 1 = yes
    # 6. Slope: 1 = upsloping, 2 = flat, 3 = downsloping
    # 7. CA: number of major vessels (0-3) colored by flourosopy
    # 8. Thal: 3 = normal, 6 = fixed defect, 7 = reversable defect
    
    # Convert categorical variables to numeric if needed
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    # Drop any rows that still have missing values after conversion
    data = data.dropna()
    
    # Prepare features and target
    X = data.drop('target', axis=1).values
    y = data['target'].values
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return data, X, y, list(data.drop('target', axis=1).columns), 'target'

def display_data_statistics(data):
    """
    Display statistics for the dataset
    
    Args:
        data (pd.DataFrame): The dataset
    """
    # Display basic statistics
    st.write("### Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Display missing values
    st.write("### Missing Values")
    missing_data = pd.DataFrame({
        'Column': data.columns,
        'Missing Values': data.isnull().sum().values,
        'Percentage': (data.isnull().sum().values / len(data) * 100)
    })
    st.dataframe(missing_data)
    
    # Display data types
    st.write("### Data Types")
    data_types = pd.DataFrame({
        'Column': data.columns,
        'Data Type': [str(data[col].dtype) for col in data.columns]
    })
    st.dataframe(data_types)
    
    # Count of target variable values
    st.write("### Target Distribution")
    target_counts = pd.DataFrame(data['target'].value_counts()).reset_index()
    target_counts.columns = ['Target Value', 'Count']
    target_counts['Description'] = target_counts['Target Value'].apply(
        lambda x: 'No Heart Disease' if x == 0 else 'Heart Disease'
    )
    st.dataframe(target_counts)
