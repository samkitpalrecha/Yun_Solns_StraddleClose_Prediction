from flask import Flask, request, jsonify
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        index = df['index'].iloc[0]
        dte = df['DaysToExpiry'].iloc[0]

        with open(f"Scalers/scaler{index}_dte{dte}.pkl", "rb") as f:
            scaler = pickle.load(f)

        df_feature = df.drop(columns=['index', 'DaysToExpiry', 'date'])

        X = df_feature.to_numpy()
        X_ = scaler.transform(X)

        df_feature_scaled = pd.DataFrame(X_, columns=df_feature.columns)

        with open(f"K_Means/kmeans_{index}_dte{dte}.pkl", "rb") as f:
            k_means_model = joblib.load(f)

        p_c = k_means_model.predict(df_feature_scaled)
        p_c = p_c[0]

        model_nn = load_model(f'Neural_Nets/Neural_Net_{index}_dte{dte}_{p_c}.keras')

        x = df_feature_scaled[f'open_dte{dte}']
        X = x.to_numpy()
        X = X.reshape(-1,1)

        prediction = model_nn.predict(X)
        prediction = prediction.flatten()
        prediction = round(float(prediction[0]), 3)

        return jsonify({"predicted close": prediction, "Index": data['index'], "Date": data['date']})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)