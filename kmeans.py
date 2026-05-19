import sys
import numpy as np
from helpers import load_clustering_data, euclidean_distance, compute_centroid, cluster_radius
from metrics import compute_silhouette, compute_cluster_silhouettes, radius_distance_ratio

def initialize_centroids(data, k, seed=1):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=k, replace=False)
    return data[indices].astype(float)

def assign_clusters(data, centroids):
    labels = []

    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        labels.append(np.argmin(distances))

    return np.array(labels)

def recompute_centroids(data, labels, k, old_centroids):
    new_centroids = []

    for cluster_id in range(k):
        cluster_points = data[labels == cluster_id]

        if len(cluster_points) == 0:
            new_centroids.append(old_centroids[cluster_id])
        else:
            new_centroids.append(compute_centroid(cluster_points))

    return np.array(new_centroids)

def kmeans(data, k, max_iters=100, tol=0.001, seed=1):
    centroids = initialize_centroids(data, k, seed)

    for iteration in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = recompute_centroids(data, labels, k, centroids)

        centroid_shift = np.linalg.norm(new_centroids - centroids)

        if centroid_shift < tol:
            return labels, new_centroids, iteration + 1

        centroids = new_centroids

    return labels, centroids, max_iters

def print_clusters(data, labels, centroids, k):
    print("Overall clustering result")
    print(f"Number of clusters: {k}")

    unique_labels = set(labels)

    if len(unique_labels) > 1:
        overall_silhouette = compute_silhouette(data, labels)
        cluster_silhouettes = compute_cluster_silhouettes(data, labels)

        clusters = [data[labels == cluster_id] for cluster_id in range(k)]
        ratio = radius_distance_ratio(clusters)

        print(f"Overall Silhouette Score: {overall_silhouette:.4f}")
        print(f"Radius / Intercluster Distance Ratio: {ratio:.4f}")
    else:
        cluster_silhouettes = {}
        print("Overall Silhouette Score: N/A")
        print("Radius / Intercluster Distance Ratio: N/A")

    print()

    for cluster_id in range(k):
        cluster_points = data[labels == cluster_id]

        print(f"Cluster {cluster_id}")
        print(f"Size: {len(cluster_points)}")
        print(f"Centroid: {centroids[cluster_id].tolist()}")

        if len(cluster_points) > 0:
            print(f"Radius: {cluster_radius(cluster_points):.4f}")

            if cluster_id in cluster_silhouettes:
                print(f"Silhouette Score: {cluster_silhouettes[cluster_id]:.4f}")
            else:
                print("Silhouette Score: N/A")

        print("Points:")

        if len(cluster_points) <= 10:
            for point in cluster_points:
                print(point.tolist())
        else:
            for point in cluster_points[:3]:
                print(point.tolist())
            print("...")

        print()


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 kmeans.py <Filename> <k>")
        return

    filename = sys.argv[1]
    k = int(sys.argv[2])

    data = load_clustering_data(filename)

    labels, centroids, iterations = kmeans(data, k)

    print(f"K-means completed in {iterations} iterations")
    print_clusters(data, labels, centroids, k)


if __name__ == "__main__":
    main()