# app.py

from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved centroids (assuming they are pickled)
with open('model/kmeans_centroids.pkl', 'rb') as f:
    centroids = pickle.load(f)

# Define the number of clusters (same as used for training)
k = 3

def predict_cluster(elevation, rainfall, centroids):
    """
    Predicts the cluster label for a new data point using the saved centroids.

    Args:
        elevation: A float value representing the elevation.
        rainfall: A float value representing the rainfall.
        centroids: A numpy array of shape (k, 2) containing the cluster centers.

    Returns:
        An integer representing the predicted cluster label.
    """
    new_data = np.array([[elevation, rainfall]])  # Reshape for prediction
    distances = np.linalg.norm(centroids - new_data, axis=1)  # Euclidean distances
    predicted_cluster = np.argmin(distances)  # Index of the closest centroid
    return predicted_cluster

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    elevation = float(data['elevation'])
    rainfall = float(data['rainfall'])

    predicted_cluster = predict_cluster(elevation, rainfall, centroids)

    category = ""
    if int(predicted_cluster) == 0:
        category = "Aman"
    elif int(predicted_cluster) == 1:
        category = "Waspada"
    elif int(predicted_cluster) == 2:
        category = "Awas"
    elif int(predicted_cluster) == 3:
        category = "Siaga"

    return jsonify({
        'predicted_cluster': int(predicted_cluster),
        'category': category
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
