import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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

    # calculate inertia for each k value
    def calculate_inertia(self, X, centroids, labels, k):
        inertia = 0
        
        for i in range(k):
            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)

        return inertia

    # fit dataset
    def fit(self, X):
        centroids = self.centroid_init(X)
        total_iters = 0
        converged = False

        for i in range(self.max_iters):
            labels = self.assignment(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))

            # check convergence
            if shift < self.tolerance:
                converged = True
                break

            centroids = new_centroids
            total_iters = i
        
        return labels, new_centroids, total_iters, converged

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

# max number of clusters
k = int(np.sqrt(X_standardize.shape[0] / 2))

inertias = []
k_values = []

# create model
for i in range(2, k+1):
    my_model = kmeans(i, 300, 0.0001)
    labels, centroids, total_iters, converged = my_model.fit(X_standardize)
    k_values.append(i)
    inertia = my_model.calculate_inertia(X_standardize, centroids, labels, i)
    inertias.append(inertia)

    # evaluation by silhouette score
    score = silhouette_score(X_standardize, labels)
    print(f'k: {i}, inertia: {inertia:.2f}, sil_score: {score:.2f}, total_iters: {total_iters}, converged: {converged}')

# evaluation by elbow method
plt.figure(figsize=(7, 5))
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.show()

# figure of final and best model by k=5
final_model = kmeans(5, 300, 0.0001)
labels, centroids, _, _ = final_model.fit(X_standardize)
df['Cluster'] = labels

plt.figure(figsize=(7, 5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', data=df)
plt.scatter(scaler.inverse_transform(centroids)[:, 0],
            scaler.inverse_transform(centroids)[:, 1],
            label='Centroids')
plt.title('Customer Segmentation (K=5)')
plt.legend()
plt.show()