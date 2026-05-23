import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

class kmeans:
    def __init__(self, k, max_iters, tolerance):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    # centroid initialization
    def centroid_init(self, X):
        np.random.seed(42)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=self.k, replace=False)
        return X[indices].copy()
    
    # assign each sample to a centroid
    def assignment(self, X, centroids):
        # add new dimension for calcuations
        reshaped_X = X[:, np.newaxis, :]
        reshaped_centroids = centroids[np.newaxis, :, :]
        
        distance = np.sqrt(np.sum(((reshaped_X - reshaped_centroids) ** 2), axis=2))
        labels = np.argmin(distance, axis=1)
        return labels
    
    # update centroids by the points of each cluster
    def update_centroids(self, X, labels):
        new_centroids = []

        for i in range(self.k):
            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                new_center = cluster_points.mean(axis=0)
                new_centroids.append(new_center)
            else:
                random_center = self.centroid_init(X)
                new_centroids.append(random_center[0])

        return np.array(new_centroids)

# read dataset
df = pd.read_csv('Mall_Customers.csv')

# drop CustomerID column
df = df.drop('CustomerID', axis=1)

# pairplot including gender
sns.pairplot(df, hue='Gender')
plt.suptitle("Pairplot segmented by Gender", y=1.02)
plt.show()

# drop Gender & Age column because it has no influence on clustering
df = df.drop('Gender', axis=1)
df = df.drop('Age', axis=1)

# normalize dataset
scaler = StandardScaler()
X_standardize = scaler.fit_transform(df)