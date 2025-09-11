import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.distance import cdist
from collections import deque

def knn(x, n_neighbors):
    """
    Compute k-nearest neighbors for each point.

    Args:
        x (np.ndarray): Input data of shape (num_points, num_features).
        n_neighbors (int): Number of neighbors to find.

    Returns:
        np.ndarray: Indices of k nearest neighbors for each point,
                    shape (num_points, n_neighbors).
    """
    # Compute distances from points to all other points
    distances = cdist(x, x)
    # Get indices of k nearest neighbors for each point
    # np.argsort sorts in ascending order, so we take the first k+1
    # We skip the first one (index 0) because it's the point itself
    knn_indices = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]
    return knn_indices

def cluster_points(x, n_neighbors):
    """
    Cluster points based on k-nearest neighbors.

    Args:
        x (np.ndarray): Input data of shape (num_points, num_features).
        n_neighbors (int): Number of neighbors to use for clustering.

    Returns:
        tuple:
            - list: Sizes of each cluster.
            - np.ndarray: Cluster assignment for each point, shape (num_points,).
    """
    num_points = x.shape[0]
    l = knn(x, n_neighbors)
    cluster_sizes = []
    point_clusters = np.zeros(num_points, dtype=np.int32)
    cluster_idx = 0
    to_visit = set(range(l.shape[0])) # Using l.shape[0] which is num_points
    stack = deque()

    while len(to_visit) > 0:
        # Efficiently get an arbitrary element from the set
        idx = to_visit.pop()
        cluster_size = 0
        stack.append(idx)
        # Mark as visited by putting it back temporarily if needed, or handle in logic
        # In this DFS-like approach, once it's in the stack and processed, it's "visited"
        # with respect to the current cluster formation.

        current_cluster_nodes = {idx} # Keep track of nodes added to stack for this cluster

        while len(stack) > 0:
            curr_i = stack.popleft()
            # Check if already processed for a cluster (if it was popped from to_visit before)
            # This check is implicitly handled as we only add from 'to_visit'

            cluster_size += 1
            point_clusters[curr_i] = cluster_idx

            # Get unique neighbor indices for the current point
            # l[curr_i] should already be unique if n_neighbors < number of unique points
            # but torch.unique was used, so we replicate with np.unique
            unique_indices = np.unique(l[curr_i])

            for i_neighbor in unique_indices:
                # i_neighbor is an integer index
                if i_neighbor in to_visit:
                    to_visit.remove(i_neighbor)
                    stack.append(i_neighbor)
                    current_cluster_nodes.add(i_neighbor)
                # If already visited and assigned to a different cluster, this creates a link
                # The original algorithm implies a connected components approach.

        if cluster_size > 0: # Only append if the cluster actually formed
            cluster_sizes.append(cluster_size)
            cluster_idx += 1
        # Ensure all nodes popped from to_visit for this cluster are truly removed
        # This is handled by the to_visit.remove(i_neighbor) line.

    return cluster_sizes, point_clusters

def enforce_min_cluster_size(x, labels, min_cluster_size):
    """
    Enforce a minimum cluster size by merging small clusters with their nearest neighbors.
    
    Args:
        x (np.ndarray): Input data of shape (num_points, num_features).
        labels (np.ndarray): Cluster labels for each point.
        min_cluster_size (int): Minimum allowed cluster size.
        
    Returns:
        np.ndarray: Updated cluster labels after merging small clusters.
    """
    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create a new array to hold updated labels
    new_labels = labels.copy()
    
    # Identify small clusters
    small_clusters = unique_labels[counts < min_cluster_size]
    large_clusters = unique_labels[counts >= min_cluster_size]
    
    # If there are no large clusters, keep the largest small cluster and merge the rest
    if len(large_clusters) == 0:
        # Find the largest small cluster
        largest_small_idx = np.argmax(counts)
        largest_small_label = unique_labels[largest_small_idx]
        # Merge all other small clusters into the largest one
        for cluster in unique_labels:
            if cluster != largest_small_label:
                new_labels[labels == cluster] = largest_small_label
        return new_labels
    
    # Calculate cluster centroids
    centroids = {}
    for label in unique_labels:
        cluster_points = x[labels == label]
        centroids[label] = np.mean(cluster_points, axis=0)
    
    # For each small cluster, find the nearest large cluster and merge
    for small_cluster in small_clusters:
        # Get centroid of small cluster
        small_centroid = centroids[small_cluster]
        
        # Find nearest large cluster
        min_distance = float('inf')
        nearest_large_cluster = None
        
        for large_cluster in large_clusters:
            distance = np.linalg.norm(small_centroid - centroids[large_cluster])
            if distance < min_distance:
                min_distance = distance
                nearest_large_cluster = large_cluster
        
        # Merge small cluster into nearest large cluster
        new_labels[labels == small_cluster] = nearest_large_cluster
    
    # Relabel to have consecutive integers
    final_labels = np.zeros_like(new_labels)
    for i, label in enumerate(np.unique(new_labels)):
        final_labels[new_labels == label] = i
    
    return final_labels

def get_knn_clusters(x, max_components=None):
    """
    Determine the number of clusters and labels by iterating through n_neighbors.
    Enforces a minimum occupancy of twice the number of features per cluster.

    Args:
        x (np.ndarray): Input data of shape (num_points, num_features).
        max_components (int, optional): Maximum number of neighbors to consider.
                                     Defaults to number of points divided by twice the number of features.

    Returns:
        tuple:
            - int: Number of clusters found.
            - np.ndarray: Cluster labels for each point.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a NumPy array.")

    num_points = x.shape[0]
    num_features = x.shape[1]
    min_cluster_size = 2 * num_features  # Minimum cluster size
    
    max_components = num_points // (2 * num_features) if max_components is None else max_components
    # Ensure ns doesn't exceed potential neighbors (num_points - 1 for knn)
    ns = [i for i in range(min(max_components, num_points))] # n_neighbors cannot be num_points

    sizes_prev = None
    labels_to_return = np.zeros(num_points, dtype=np.int32) # Initialize

    if num_points == 0:
        return 0, labels_to_return
    if num_points == 1:
        return 1, np.array([0], dtype=np.int32)


    for n_neighbors in ns:
        if n_neighbors == 0: # KNN with 0 neighbors means each point is its own cluster
            sizes = [1] * num_points
            labels = np.arange(num_points, dtype=np.int32)
        elif n_neighbors >= num_points: # Cannot have more neighbors than other points
             # This case should ideally be caught by knn if k > N-1
             # For simplicity, let's assume knn handles k >= N appropriately
             # or we can break/continue
            sizes, labels = cluster_points(x, n_neighbors=num_points-1)
        else:
            sizes, labels = cluster_points(x, n_neighbors=n_neighbors)

        # Condition for stopping:
        # 1. Cluster sizes are the same as the previous iteration
        # 2. All clusters are larger than the minimum size
        if sizes_prev is not None and \
           sorted(sizes) == sorted(sizes_prev) and \
           all(size >= min_cluster_size for size in sizes):
            return len(sizes_prev), labels_prev # Return previous state's results

        sizes_prev = sizes
        labels_prev = labels # Store current labels for potential return in next iteration
        labels_to_return = labels # Update labels to return if loop completes

    # If we reach this point, it means we've gone through all the potential neighbors
    # and still have small clusters. Now we need to merge them.
    if sizes_prev is not None:
        # Get the counts for each cluster
        unique_labels, counts = np.unique(labels_to_return, return_counts=True)
        
        # Check if there are any small clusters
        if np.any(counts < min_cluster_size):
            # Enforce minimum cluster size by merging small clusters
            labels_to_return = enforce_min_cluster_size(x, labels_to_return, min_cluster_size)
            
            # Recompute unique labels after merging
            unique_labels = np.unique(labels_to_return)
            new_cluster_count = len(unique_labels)
            
            return new_cluster_count, labels_to_return
        
        return len(sizes_prev), labels_to_return
    
    # Handle edge cases
    if num_points > 0:
        return num_points, np.arange(num_points, dtype=np.int32) # Each point is a cluster
    else:
        return 0, np.array([], dtype=np.int32)
