import sys
import numpy as np
from helpers import load_clustering_data, euclidean_distance, cluster_radius
from metrics import compute_silhouette, compute_cluster_silhouettes, radius_distance_ratio

def epsilon_neighborhood(data, idx, eps):
    neighbors = []
    for j in range(len(data)):
        if j != idx and euclidean_distance(data[idx], data[j]) <= eps:
            neighbors.append(j)
    return neighbors

def identify_core_pts(data, eps, min_pts):
    neighborhoods = {}
    core = set()

    for i in range(len(data)):
        neighbors = epsilon_neighborhood(data, i, eps)
        neighborhoods[i] = neighbors

        if len(neighbors) + 1 >= min_pts:
            core.add(i)

    return neighborhoods, core

def expand_clusters(data, neighborhoods, core):
    n = len(data)
    labels = np.full(n, -1)
    cluster_id = 0

    def density_connected(current_point, current_cluster):
        for neighbor in neighborhoods[current_point]:
            if labels[neighbor] == -1:
                labels[neighbor] = current_cluster
                if neighbor in core:
                    density_connected(neighbor, current_cluster)

    for point in core:
        if labels[point] == -1:
            cluster_id += 1
            labels[point] = cluster_id
            density_connected(point, cluster_id)

    return labels


def dbscan(data, eps, num_points):
    neighborhoods, core = identify_core_pts(data, eps, num_points)
    labels = expand_clusters(data, neighborhoods, core)
    return labels, core

def print_clusters(data, labels, core, eps, min_pts):
    print("DBSCAN clustering result")
    print(f"Epsilon: {eps}, MinPts: {min_pts}")

    unique_labels = set(labels)

    clusters = [data[labels == c] for c in unique_labels if c != -1]
    noise = data[labels == -1]

    if len(clusters) > 1:
        silhouette = compute_silhouette(data, labels)
        ratio = radius_distance_ratio(clusters)

        print(f"Overall Silhouette Score: {silhouette:.4f}")
        print(f"Radius / Intercluster Distance Ratio: {ratio:.4f}")

    else:
        print("Overall Silhouette Score: N/A")
        print("Radius / Intercluster Distance Ratio: N/A")

    print(f"Noise Points: {len(noise)}\n")

    cluster_silhouette = compute_cluster_silhouettes(data, labels) if len(clusters) > 1 else {}

    for c in unique_labels:
        if c == -1:
            continue
        cluster_points = data[labels == c]
        print(f"Cluster {c}:")
        print(f"Size: {len(cluster_points)}")

        centroid = np.mean(cluster_points, axis=0)
        print(f"Centroid: {centroid.tolist()}")
        print(f"Radius: {cluster_radius(cluster_points):.4f}")

        if c in cluster_silhouette:
            print(f"Silhouette Score: {cluster_silhouette[c]:.4f}")

        core_count = 0
        for i in range(len(data)):
            if labels[i] == c and i in core:
                core_count += 1

        print(f"Core Points: {core_count}")

        print("Points:")
        for p in cluster_points[:3]:
            print(p.tolist())

        print()

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 dbscan.py <Filename> <epsilon> <NumPoints>")
        return

    filename = sys.argv[1]
    eps = float(sys.argv[2])
    min_pts = int(sys.argv[3])

    data = load_clustering_data(filename)

    labels, core = dbscan(data, eps, min_pts)
    print_clusters(data, labels, core, eps, min_pts)

if __name__ == '__main__':
    main()
