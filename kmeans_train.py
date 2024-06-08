from sklearn.cluster import KMeans
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load your data from the CSV file
data = pd.read_csv("dataset/ready_dataset.csv")  # Replace "your_data.csv" with your actual file name

# Extract features (assuming the first two columns contain features)
features = data.iloc[:, :2]

# Define the number of clusters (k)
k = 3

# Create the KMeans model
kmeans = KMeans(n_clusters=k, random_state=0)  # Set random_state for reproducibility

# Fit the model to the data
kmeans.fit(features)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Plot the data points colored by their cluster labels
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=cluster_labels)

# Add labels for the axes
plt.xlabel("Elevation")
plt.ylabel("Rainfall")

# Add a title
plt.title("K-Means Clustering ("+ str(k) +" Clusters)")

# Show the plot
plt.show()

# The KMeans model itself doesn't persist a separate model file. 
# However, you can save the cluster centers (centroids) for later use.
centroids = kmeans.cluster_centers_

# You can save the centroids to a file (e.g., pickle) for later use
with open('model/kmeans_centroids.pkl', 'wb') as f:
  pickle.dump(centroids, f)

print("Cluster labels:", cluster_labels)
