from sklearn.cluster import KMeans
import pandas as pd
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/ready_dataset.csv")
features = data.iloc[:, :2]

k = 20
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(features)

cluster_labels = kmeans.labels_

plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=cluster_labels)
plt.xlabel("Elevation")
plt.ylabel("Rainfall")
plt.title("K-Means Clustering ("+ str(k) +" Clusters)")
plt.show()

centroids = kmeans.cluster_centers_
with open('model/kmeans_centroids.pkl', 'wb') as f:
  pickle.dump(centroids, f)

print("Cluster labels:", cluster_labels)
