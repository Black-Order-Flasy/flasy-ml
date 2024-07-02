# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
# Load the saved centroids (assuming they are pickled)
rf_model = joblib.load('random_forest_classifier/model/flood_prediction_rf_model.pkl')


def predict_flood_probability_rf(rainfall, forest_ratio, streamflow):
    prob = rf_model.predict_proba([[rainfall, forest_ratio, streamflow]])[0][1]
    return prob


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    rainfall = float(data['rainfall'])
    # forest_ratio = float(data['forest_ratio'])
    forest_ratio = 0
    streamflow = float(data['streamflow'])

    predicted_data = predict_flood_probability_rf(rainfall, forest_ratio, streamflow)

    category = ""
    if (predicted_data >= 0.00 and predicted_data <= 0.25) :
        category = "Aman"
    elif (predicted_data >= 0.251 and predicted_data <= 0.5) :
        category = "Siaga"
    elif (predicted_data >= 0.51 and predicted_data <= 0.75) :
        category = "Waspada"
    elif (predicted_data >= 0.751 and predicted_data <= 1) :
        category = "Awas"
        


    return jsonify({
        'predicted_data': int(predicted_data),
        'category': category
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# def predict_cluster(elevation, rainfall, centroids):
#     """
#     Predicts the cluster label for a new data point using the saved centroids.

#     Args:
#         elevation: A float value representing the elevation.
#         rainfall: A float value representing the rainfall.
#         centroids: A numpy array of shape (k, 2) containing the cluster centers.

#     Returns:
#         An integer representing the predicted cluster label.
#     """
#     new_data = np.array([[elevation, rainfall]])  # Reshape for prediction
#     distances = np.linalg.norm(centroids - new_data, axis=1)  # Euclidean distances
#     predicted_cluster = np.argmin(distances)  # Index of the closest centroid
#     return predicted_cluster

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     elevation = float(data['elevation'])
#     rainfall = float(data['rainfall'])

#     predicted_cluster = predict_cluster(elevation, rainfall, centroids)

#     category = ""
#     if int(predicted_cluster) == 0:
#         category = "Aman"
#     elif int(predicted_cluster) == 1:
#         category = "Waspada"
#     elif int(predicted_cluster) == 2:
#         category = "Awas"
#     elif int(predicted_cluster) == 3:
#         category = "Siaga"

#     return jsonify({
#         'predicted_cluster': int(predicted_cluster),
#         'category': category
#     })
