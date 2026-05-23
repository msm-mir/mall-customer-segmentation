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