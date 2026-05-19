import sys
import json
import numpy as np

from helpers import load_clustering_data, compute_distance_matrix, compute_centroid, cluster_radius, euclidean_distance
from metrics import compute_silhouette, compute_cluster_silhouettes, radius_distance_ratio

def cluster_distance(cluster1, cluster2, dist_matrix, linkage="single"):
    distances = []

    for i in cluster1["indices"]:
        for j in cluster2["indices"]:
            distances.append(dist_matrix[i][j])

    if linkage == "single":
        return min(distances)
    elif linkage == "complete":
        return max(distances)
    elif linkage == "average":
        return sum(distances) / len(distances)
    else:
        raise ValueError("Invalid linkage method")

def find_closest_clusters(clusters, dist_matrix, linkage="single"):
    best_i = None
    best_j = None
    best_dist = float("inf")

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = cluster_distance(clusters[i], clusters[j], dist_matrix, linkage)

            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_j = j

    return best_i, best_j, best_dist

def agglomerative_clustering(data, linkage="single"):
    dist_matrix = compute_distance_matrix(data)

    clusters = []

    for i, point in enumerate(data):
        clusters.append({
            "indices": [i],
            "tree": {
                "type": "leaf",
                "index": i,
                "data": point.tolist()
            }
        })

    while len(clusters) > 1:
        i, j, height = find_closest_clusters(clusters, dist_matrix, linkage)

        c1 = clusters[i]
        c2 = clusters[j]

        merged = {
            "indices": c1["indices"] + c2["indices"],
            "tree": {
                "type": "node",
                "height": float(height),
                "nodes": [
                    c1["tree"],
                    c2["tree"]
                ]
            }
        }

        for idx in sorted([i, j], reverse=True):
            clusters.pop(idx)

        clusters.append(merged)

    root = clusters[0]["tree"]
    root["type"] = "root"

    return root

def collect_indices(node):
    if node["type"] == "leaf":
        return [node["index"]]

    indices = []
    for child in node["nodes"]:
        indices.extend(collect_indices(child))

    return indices


def cut_tree(tree, threshold):
    clusters = []

    def helper(node):
        if node["type"] == "leaf":
            clusters.append([node["index"]])
        elif node["height"] <= threshold:
            clusters.append(collect_indices(node))
        else:
            for child in node["nodes"]:
                helper(child)

    helper(tree)
    return clusters


def clusters_to_labels(n_points, clusters):
    labels = np.full(n_points, -1)

    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            labels[idx] = cluster_id

    return labels

def print_threshold_clusters(data, clusters):
    labels = clusters_to_labels(len(data), clusters)

    print()
    print("Clusters after threshold cut")
    print(f"Number of clusters: {len(clusters)}")

    if len(set(labels)) > 1:
        overall_sil = compute_silhouette(data, labels)
        cluster_sils = compute_cluster_silhouettes(data, labels)
        cluster_points_list = [data[idxs] for idxs in clusters]
        ratio = radius_distance_ratio(cluster_points_list)

        print(f"Overall Silhouette Score: {overall_sil:.4f}")
        print(f"Radius / Intercluster Distance Ratio: {ratio:.4f}")
    else:
        cluster_sils = {}
        print("Overall Silhouette Score: N/A")
        print("Radius / Intercluster Distance Ratio: N/A")

    print()

    for cluster_id, indices in enumerate(clusters):
        cluster_points = data[indices]

        print(f"Cluster {cluster_id}")
        print(f"Size: {len(cluster_points)}")
        print(f"Centroid: {compute_centroid(cluster_points).tolist()}")
        print(f"Radius: {cluster_radius(cluster_points):.4f}")
        print(f"Silhouette Score: {cluster_sils.get(cluster_id, 'N/A')}")

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
    if len(sys.argv) < 2:
        print("Usage: python3 hclustering.py <Filename> [threshold] [linkage]")
        return

    filename = sys.argv[1]

    threshold = None
    if len(sys.argv) >= 3:
        threshold = float(sys.argv[2])

    linkage = "single"
    if len(sys.argv) >= 4:
        linkage = sys.argv[3]

    data = load_clustering_data(filename)

    tree = agglomerative_clustering(data, linkage=linkage)

    print(json.dumps(tree, indent=2))

    if threshold is not None:
        clusters = cut_tree(tree, threshold)
        print_threshold_clusters(data, clusters)


if __name__ == "__main__":
    main()