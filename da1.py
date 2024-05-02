
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


data = pd.read_csv('customer_data.csv')

# Data cleaning and preprocessing

data.dropna(inplace=True)

# Perform feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Customer segmentation using K-means clustering
# Determine optimal number of clusters using silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Add 2 to account for range start at 2
print("Optimal number of clusters:", optimal_num_clusters)

# Perform K-means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to original dataset
data['Cluster'] = cluster_labels

# Customer segmentation visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Feature1', y='Feature2', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Market basket analysis
# Implement market basket analysis techniques to identify frequently co-purchased products and associations between different product categories
# Perform time-series analysis to identify trends in customer behavior over time

# Insights and recommendations
# Summarize key findings and insights from the analysis
# Provide actionable recommendations for marketing strategies tailored to each customer segment

# Save the updated dataset with cluster labels
data.to_csv('segmented_customer_data.csv', index=False)
