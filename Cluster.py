import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering

csv_file = "csvfile.csv"
df = pd.read_csv(csv_file, sep=';')
X = df[['probability_mean']].values

# Способ 1: Кластеризация с DBSCAN с уменьшенным eps.
eps_value = 0.0010  # Пример: попробуйте уменьшить значение eps
dbscan = DBSCAN(eps=eps_value, min_samples=1)
df['cluster_dbscan'] = dbscan.fit_predict(X)

cluster_stats_dbscan = df.groupby('cluster_dbscan')['probability_mean'].agg(['median', 'count']).reset_index()
print("DBSCAN кластеризация (eps = {:.3f}):".format(eps_value))
print(cluster_stats_dbscan)

unique_clusters_dbscan = sorted(df['cluster_dbscan'].unique())
colors_dbscan = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_clusters_dbscan)))

plt.figure(figsize=(10, 6))
for color, cluster in zip(colors_dbscan, unique_clusters_dbscan):
    cluster_data = df[df['cluster_dbscan'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['probability_mean'],
                color=color, label=f'Кластер {cluster}', s=50, alpha=0.7)
plt.xlabel('Индекс записи')
plt.ylabel('Probability Mean')
plt.title(f'Кластеризация DBSCAN (eps = {eps_value})')
plt.legend()
plt.grid(True)
plt.show()

# Способ 2: Агломеративная кластеризация с порогом расстояния.
distance_threshold_value = 0.05  # Можно также подобрать экспериментально
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold_value)
df['cluster_agg'] = agg.fit_predict(X)

cluster_stats_agg = df.groupby('cluster_agg')['probability_mean'].agg(['median', 'count']).reset_index()
print("Агломеративная кластеризация (distance_threshold = {:.3f}):".format(distance_threshold_value))
print(cluster_stats_agg)

unique_clusters_agg = sorted(df['cluster_agg'].unique())
colors_agg = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_clusters_agg)))

plt.figure(figsize=(10, 6))
for color, cluster in zip(colors_agg, unique_clusters_agg):
    cluster_data = df[df['cluster_agg'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['probability_mean'],
                color=color, label=f'Кластер {cluster}', s=50, alpha=0.7)
plt.xlabel('Индекс записи')
plt.ylabel('Probability Mean')
plt.title(f'Агломеративная кластеризация (distance_threshold = {distance_threshold_value})')
plt.legend()
plt.grid(True)
plt.show()