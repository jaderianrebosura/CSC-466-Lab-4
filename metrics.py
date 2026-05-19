import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, rand_score
from helpers import euclidean_distance, cluster_radius

def compute_silhouette(data, labels):
    return silhouette_score(data, labels)

def compute_cluster_silhouettes(data, labels):
    sil_samples = silhouette_samples(data, labels)
    cluster_scores = {}

    for cluster_id in set(labels):
        cluster_scores[cluster_id] = np.mean(sil_samples[labels == cluster_id])

    return cluster_scores

def intercluster_distance(cluster1, cluster2):
    min_dist = float("inf")
    for p1 in cluster1:
        for p2 in cluster2:
            d = euclidean_distance(p1, p2)
            if d < min_dist:
                min_dist = d
    return min_dist

def radius_distance_ratio(clusters):
    ratios = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            r1 = cluster_radius(clusters[i])
            r2 = cluster_radius(clusters[j])
            dist = intercluster_distance(clusters[i], clusters[j])

            if dist > 0:
                ratios.append((r1 + r2) / dist)

    return np.mean(ratios) if ratios else 0

def compute_rand_index(true_labels, pred_labels):
    return rand_score(true_labels, pred_labels)