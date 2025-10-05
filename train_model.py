import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model():
    """
    This function trains the best Random Forest model on the full dataset
    and saves the model, scaler, and label encoder to .pkl files.
    """
    try:
      
        print("Loading and preparing the data...")
        header_row = 0
        with open('KOI.csv', 'r') as f:
            for i, line in enumerate(f):
                if 'rowid,kepid,kepoi_name' in line:
                    header_row = i
                    break
        df = pd.read_csv('KOI.csv', skiprows=header_row)

        recommended_columns = [
            'koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_fpflag_ec', 'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_model_snr'
        ]
        exo_df = df[recommended_columns].copy()

        df_processed = exo_df.copy()
        df_processed.dropna(subset=['koi_disposition'], inplace=True)
        
        for col in df_processed.select_dtypes(include=np.number).columns:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)

        df_processed['koi_period_log'] = np.log1p(df_processed['koi_period'])
        df_processed.drop(columns=['koi_period'], inplace=True)

        X = df_processed.drop('koi_disposition', axis=1)
        y = df_processed['koi_disposition']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

       
        print("Training the final, best model...")
        best_rf_model = RandomForestClassifier(
            n_estimators=300, 
            min_samples_split=2, 
            min_samples_leaf=2,
            max_features='sqrt', 
            max_depth=None, 
            random_state=42
        )
        best_rf_model.fit(X_scaled, y_encoded)

  
        print("\nSaving the model, scaler, and label encoder as .pkl files...")
        
        with open('best_exoplanet_model.pkl', 'wb') as model_file:
            pickle.dump(best_rf_model, model_file)

        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
            
        with open('label_encoder.pkl', 'wb') as le_file:
            pickle.dump(le, le_file)

        print("✅ Training complete. Files saved successfully!")
        print("Files created: 'best_exoplanet_model.pkl', 'scaler.pkl', 'label_encoder.pkl'")

    except FileNotFoundError:
        print("Could not find 'KOI.csv'. Please ensure the file is in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    train_and_save_model()
