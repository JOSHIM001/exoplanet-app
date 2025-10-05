import pandas as pd
import numpy as np
from config import MODEL_FEATURES

def preprocess_data(df):
    """
    Preprocesses the input DataFrame to match the format used for model training.
    - Handles missing values
    - Creates the 'koi_period_log' feature
    - Ensures columns are in the correct order
    """
    # Handle missing values by filling with the median of the column
    for col in df.select_dtypes(include=np.number).columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Feature Engineering: Create the log-transformed period
    if 'koi_period' in df.columns:
        df['koi_period_log'] = np.log1p(df['koi_period'])
        df.drop(columns=['koi_period'], inplace=True)
    
    # Ensure all required model features are present, fill missing ones with 0
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
            
    # Return the DataFrame with columns in the correct order for the model
    return df[MODEL_FEATURES]
