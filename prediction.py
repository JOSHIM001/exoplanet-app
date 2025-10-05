# In your prediction.py file

import pickle
import pandas as pd
import numpy as np

def make_prediction_with_proba(input_data):
   
    with open("best_exoplanet_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

  
    input_df = pd.DataFrame(input_data)
    
   
    if 'koi_period' in input_df.columns:
        input_df['koi_period_log'] = np.log1p(input_df['koi_period'])
        input_df.drop(columns=['koi_period'], inplace=True)
    

    model_features = [
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'koi_duration', 'koi_depth', 'koi_impact', 'koi_prad', 'koi_teq',
        'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_model_snr',
        'koi_period_log'
    ]
    input_df = input_df[model_features]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)

  
    probabilities = model.predict_proba(input_scaled)
    prediction_labels = model.predict(input_scaled)

   
    result_df = pd.DataFrame(input_data)
    result_df['prediction_label'] = prediction_labels

   
    class_mapping = {
        0: 'CANDIDATE',
        1: 'CONFIRMED',
        2: 'FALSE POSITIVE'
    }
    result_df['prediction'] = result_df['prediction_label'].map(class_mapping)

    
    return result_df, probabilities