import pandas as pd
import numpy as np

def load_clustering_data(filename):
    rows = []

    with open(filename, "r") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]

            if len(parts) > 0 and parts[-1] == "":
                parts = parts[:-1]

            if len(parts) > 0:
                rows.append(parts)

    restriction = rows[0]
    data_rows = rows[1:]

    use_cols = [i for i, val in enumerate(restriction) if int(val) == 1]

    data = []
    for row in data_rows:
        selected = [float(row[i]) for i in use_cols]
        data.append(selected)

    return np.array(data)

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def compute_distance_matrix(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(data[i], data[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix

def compute_centroid(points):
    return np.mean(points, axis=0)

def cluster_radius(cluster_points):
    centroid = compute_centroid(cluster_points)
    return max(euclidean_distance(p, centroid) for p in cluster_points)

def intercluster_distance(cluster1, cluster2):
    min_dist = float("inf")
    for p1 in cluster1:
        for p2 in cluster2:
            dist = euclidean_distance(p1, p2)
            if dist < min_dist:
                min_dist = dist
    return min_dist
