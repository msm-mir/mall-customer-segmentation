# Mall Customer Segmentation

This project contains the implementation of the **K-Means Clustering** algorithm built completely from scratch (without using pre-built clustering libraries like `scikit-learn`) to segment mall customers based on their demographic and purchasing behavior.

---

## Project Overview
The goal of this project is to apply unsupervised learning techniques to group customers of a shopping mall. Segmenting customers helps businesses design more targeted and effective marketing strategies.

### Dataset Information
- Gender
- Age
- Annual Income
- Spending Score

---

## Features
- **Centroid Initialization:** Random selection of initial cluster centers.
- **Euclidean Distance:** Calculating the distance between data points and centroids.
- **Iterative Update:** Reassigning points to the nearest centroid and updating centroids until convergence.
- **Model Evaluation:** Finding the optimal number of clusters (k) using:
  - Elbow Method
  - Silhouette Score

---

## Exploratory Data Analysis & Preprocessing
1. **Feature Selection:** To optimize clustering performance, irrelevant or low-impact features were removed. Feature selection was conducted based on how each attribute influences the separation of the resulting clusters.
2. **Feature Scaling:** The data is normalized using `StandardScaler`.

---

## Project Structure

```
├── src/
│   └── src.py
├── visualizations/
│   |── customer_segmentation.png
│   |── elbow_method.png
│   └── pairplot_by_gender.png
├── .gitignore
└── README.md
```
