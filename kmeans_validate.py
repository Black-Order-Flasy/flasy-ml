
import pickle
import numpy as np

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

# Example usage: predict cluster labels for new data points
# elevation1 = 45
elevation = float(input("Masukkan elevasi: "))
# rainfall = 3.2
rainfall = float(input("Masukkan rainfall: "))
# elevation2 = 10
# rainfall2 = 0.8

# predicted_cluster1 = predict_cluster(elevation1, rainfall1, centroids)
predicted_cluster = predict_cluster(elevation, rainfall, centroids)

# print("Predicted cluster for (", elevation1, ",", rainfall1, "):", predicted_cluster1)
category = ""
if (int(predicted_cluster) == 0) :
  category = "Waspada"
elif (int(predicted_cluster) == 1) :
  category = "Awas"
elif (int(predicted_cluster) == 2) :
  category = "Siaga"
print("Predicted cluster for (", elevation, ",", rainfall, "):", predicted_cluster, category)
