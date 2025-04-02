import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Sample dataset
data = np.array([
    [1.0, 23.0], [2.5, 19.8], [3.2, 25.6], [0.9, 18.4], [4.1, 22.7],
    [2.3, 21.9], [3.8, 24.3], [1.5, 20.2], [2.9, 26.5], [4.0, 23.9]
])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(data)

# Apply Expectation-Maximization (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(data)

# Evaluate performance
silhouette_kmeans = silhouette_score(data, kmeans_labels)
silhouette_gmm = silhouette_score(data, gmm_labels)

print(f"Silhouette Score - K-Means: {silhouette_kmeans:.4f}")
print(f"Silhouette Score - GMM: {silhouette_gmm:.4f}")

# Plot results
def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_clusters(data, kmeans_labels, "K-Means Clustering")
plot_clusters(data, gmm_labels, "GMM Clustering")

