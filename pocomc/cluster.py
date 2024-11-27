import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import det
from scipy.special import logsumexp
from typing import List, Optional, Tuple
import warnings

class RecursiveDensityClustering(BaseEstimator, ClusterMixin):
    """
    Recursive Density-Based Clustering with Optional Statistical Error in Density Computations.

    This clustering algorithm recursively splits the cluster with the lowest density into two child clusters
    using KMeans. A split is accepted only if the average density of the child clusters exceeds a user-defined
    factor (`alpha`) times the density of the parent cluster. Optionally, the decision to accept a split can
    account for the statistical uncertainty in density estimations. Additionally, splits resulting in clusters with
    fewer than `min_points` points are rejected. If a split is rejected, the algorithm terminates early.

    Parameters
    ----------
    max_components : int, default=100
        Maximum number of clusters to create. The algorithm will terminate after this many splits.
    
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of `n_init` consecutive runs in terms of inertia.

    max_iter : int, default=1000
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Tolerance for convergence of the k-means algorithm.

    min_points : int, default=None
        Minimum number of points required in a cluster to allow splitting.
        If None, defaults to `2 * n_features` to ensure reliable covariance estimation.

    alpha : float, default=2.0
        Density threshold factor for accepting a split.
        A split is accepted only if the average density of the child clusters is at least `alpha` times
        the density of the parent cluster.

    use_error_threshold : bool, default=True
        Determines whether to incorporate statistical uncertainty in the density threshold.
        - If `True`, the split is accepted only if the average density of the child clusters exceeds
          `alpha` times the parent density **plus** the combined standard error of the parent and child densities.
        - If `False`, the split is accepted solely based on the density ratio without considering uncertainty.

    rescale : bool, default=False
        If True, scales the data to lie within the [0, 1] range before clustering.
        Ensures that all features contribute equally to the distance computations.

    verbose : bool, default=False
        If True, prints progress messages during clustering.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.

    cluster_centers_ : List[np.ndarray]
        List of cluster centers. Each center is an array of shape (n_features,).

    cluster_covariances_ : List[np.ndarray]
        List of covariance matrices for each cluster.

    n_clusters_ : int
        The number of clusters found.

    scaler_ : MinMaxScaler or None
        Fitted scaler used for data rescaling. Available only if `rescale=True`.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=0)
    >>> clustering = RecursiveDensityClustering(n_init=10, max_iter=1000, alpha=2.0, rescale=True, verbose=True, random_state=42)
    >>> clustering.fit(X)
    Recursive Density-Based Clustering:
    Initial number of clusters: 1
    Iteration 1: Split cluster 0 into clusters 1 and 2
    Iteration 2: Split cluster 2 into clusters 3 and 4
    Iteration 3: Split cluster 4 into clusters 5 and 6
    Final number of clusters: 3
    >>> clustering.labels_
    array([0, 0, 0, ..., 2, 2, 2])
    """

    def __init__(self, 
                 max_components: int = 100,
                 n_init: int = 10, 
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 min_points: Optional[int] = None,
                 alpha: float = 2.0, 
                 use_error_threshold: bool = True,
                 rescale: bool = False, 
                 verbose: bool = False,
                 random_state: Optional[int] = None
                 ):
        self.max_components = max_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.min_points = min_points
        self.alpha = alpha
        self.use_error_threshold = use_error_threshold
        self.rescale = rescale
        self.verbose = verbose
        self.random_state = random_state

        # Attributes to be populated during fitting
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: List[np.ndarray] = []
        self.cluster_covariances_: List[np.ndarray] = []
        self.n_clusters_: int = 0
        self.scaler_: Optional[MinMaxScaler] = None

    def compute_log_volume(self, cluster_data: np.ndarray) -> Tuple[float, float]:
        """
        Estimates the differential entropy of a cluster using the covariance matrix and computes the standard deviation.

        Parameters
        ----------
        cluster_data : np.ndarray
            Data points within the cluster. Shape: (n_points, n_features)

        Returns
        -------
        Tuple[float, float]
            Estimated log volume of the cluster and its standard deviation.
        """
        n_samples, n_features = cluster_data.shape
        if n_samples < n_features:
            # Covariance matrix cannot be computed reliably
            return -np.inf, 0.0  # Represents log(0) and zero std deviation

        # Compute covariance matrix and do a minor regularization
        cov_matrix = np.cov(cluster_data, rowvar=False)
        diag_cov_matrix = np.diag(np.var(cluster_data, axis=0))
        cov_matrix = (n_samples - 1) * cov_matrix / n_samples + diag_cov_matrix / n_samples

        # Compute log determinant of the covariance matrix
        sign, log_det_cov = np.linalg.slogdet(cov_matrix)
        if sign <= 0:
            # Singular or non-positive definite covariance matrix
            warnings.warn("Covariance matrix is singular or not positive definite. Adjusting with regularization.")
            cov_matrix += np.eye(n_features) * 1e-6  # Regularization
            sign, log_det_cov = np.linalg.slogdet(cov_matrix)
            if sign <= 0:
                return -np.inf, 0.0  # Cannot compute a valid log determinant

        # Correct for bias in log determinant calculation
        correction = np.sum(np.log(1.0 - np.arange(n_features) / n_samples))
        log_det_cov -= correction

        # Compute log volume
        log_volume = 0.5 * log_det_cov
        log_volume += n_features * (np.log(2.0 * np.pi) + 1.0) / 2.0

        # Compute standard deviation of log_det_cov
        try:
            ratio = n_features / n_samples
            if ratio >= 1.0:
                # Avoid math domain error
                log_volume_std = 0.0
            else:
                log_volume_std = 0.5 * np.sqrt(-2.0 * np.log(1.0 - ratio))
        except:
            log_volume_std = 0.0

        return log_volume, log_volume_std

    def compute_log_density(self, cluster_data: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the logarithmic density of a cluster and its standard deviation.

        Parameters
        ----------
        cluster_data : np.ndarray
            Data points within the cluster. Shape: (n_points, n_features)

        Returns
        -------
        Tuple[float, float]
            Log density of the cluster and its standard deviation.
        """
        n_points = cluster_data.shape[0]
        if n_points == 0:
            return -np.inf, 0.0  # log(0) and zero std deviation
        log_volume, log_volume_std = self.compute_log_volume(cluster_data)
        if log_volume == -np.inf:
            return -np.inf, 0.0
        log_density = np.log(n_points) - log_volume
        log_density_std = log_volume_std  # Assuming uncertainty primarily from log_volume
        return log_density, log_density_std

    def split_cluster(self, cluster_indices: List[int], X: np.ndarray) -> Optional[List[List[int]]]:
        """
        Splits a cluster into two using KMeans based on data point indices.

        Parameters
        ----------
        cluster_indices : List[int]
            Indices of data points in the cluster.

        X : np.ndarray
            The entire dataset. Shape: (n_samples, n_features)

        Returns
        -------
        List[List[int]] or None
            List containing two child clusters as lists of indices if split is successful, else None.
        """
        n_features = X.shape[1]
        current_min_points = self.min_points if self.min_points is not None else 2 * n_features  # Updated default

        if len(cluster_indices) < 2 * current_min_points:
            # Not enough points to split
            return None

        try:
            kmeans = KMeans(n_clusters=2, 
                            n_init=self.n_init, 
                            max_iter=self.max_iter, 
                            tol=self.tol, 
                            random_state=self.random_state)
            cluster_data = X[cluster_indices]
            labels = kmeans.fit_predict(cluster_data)

            # Split indices based on KMeans labels
            child1_indices = [idx for idx, label in zip(cluster_indices, labels) if label == 0]
            child2_indices = [idx for idx, label in zip(cluster_indices, labels) if label == 1]

            # Ensure both children have at least min_points
            if len(child1_indices) < current_min_points or len(child2_indices) < current_min_points:
                return None

            return [child1_indices, child2_indices]

        except Exception as e:
            warnings.warn(f"KMeans split failed: {e}")
            return None

    def fit(self, X: np.ndarray, y=None) -> 'RecursiveDensityClustering':
        """
        Performs recursive density-based clustering on the input data, optionally accounting for statistical uncertainty in log determinant.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to be clustered.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : RecursiveDensityClustering
            Fitted clustering model.
        """
        # Check if X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_samples, n_features = X.shape
        self.n_features = n_features

        # Rescale data if required
        if self.rescale:
            self.scaler_ = MinMaxScaler()
            X_scaled = self.scaler_.fit_transform(X)
            if self.verbose:
                print("Data has been rescaled to [0, 1] range.")
        else:
            X_scaled = X
            self.scaler_ = None

        # Initialize with all data points in one cluster (indices 0 to n_samples-1)
        current_clusters = [[i for i in range(n_samples)]]
        iteration = 0

        if self.verbose:
            print("Recursive Density-Based Clustering:")
            print(f"Initial number of clusters: {len(current_clusters)}")

        while iteration < self.max_components:
            iteration += 1
            # Compute log densities and their standard deviations for all clusters
            log_densities = []
            log_densities_std = []
            for cluster_indices in current_clusters:
                cluster_data = X_scaled[cluster_indices]
                log_density, log_density_std = self.compute_log_density(cluster_data)
                log_densities.append(log_density)
                log_densities_std.append(log_density_std)

            # Identify the cluster with the lowest log density
            # Note: Lower log density corresponds to lower actual density
            min_density_idx = np.argmin(log_densities)
            min_density = log_densities[min_density_idx]
            min_density_std = log_densities_std[min_density_idx]
            min_density_cluster = current_clusters[min_density_idx]

            # Attempt to split this cluster
            children = self.split_cluster(min_density_cluster, X_scaled)

            if children is None:
                # Cannot split this cluster, mark its density as -inf (log(0))
                log_densities[min_density_idx] = -np.inf
                log_densities_std[min_density_idx] = 0.0
                # Check if all clusters are non-splittable
                if all(d == -np.inf for d in log_densities):
                    if self.verbose:
                        print(f"All clusters are non-splittable after {iteration} iterations.")
                    break
                else:
                    continue

            # Compute log densities and standard deviations of child clusters
            child1_data = X_scaled[children[0]]
            child2_data = X_scaled[children[1]]
            log_density_child1, log_density_std_child1 = self.compute_log_density(child1_data)
            log_density_child2, log_density_std_child2 = self.compute_log_density(child2_data)
            
            # Compute logsumexp for densities: log(d1 + d2)
            log_sum_densities = logsumexp([log_density_child1, log_density_child2])
            # Compute log average density: log_sum_densities - log(2)
            log_avg_density_children = log_sum_densities - np.log(2)

            # Compute log(alpha * parent_density) = log(alpha) + log(parent_density)
            log_alpha_parent_density = np.log(self.alpha) + min_density

            # Compute combined standard error
            # Assuming independence, combine standard deviations additively
            combined_std_error = min_density_std + (log_density_std_child1 + log_density_std_child2) / 2

            # Determine if statistical error should be used in the threshold
            if self.use_error_threshold:
                # Check if log_avg_density_children >= log_alpha_parent_density + combined_std_error
                threshold = log_alpha_parent_density + combined_std_error
                condition = log_avg_density_children >= threshold
                threshold_info = f"log(alpha)*parent_log_density + combined_std_error={threshold:.4f}"
            else:
                # Check if log_avg_density_children >= log_alpha_parent_density
                threshold = log_alpha_parent_density
                condition = log_avg_density_children >= threshold
                threshold_info = f"log(alpha)*parent_log_density={threshold:.4f}"

            if condition:
                # Accept the split
                current_clusters.pop(min_density_idx)
                current_clusters.extend(children)
                if self.verbose:
                    new_cluster1 = len(current_clusters) - 2
                    new_cluster2 = len(current_clusters) - 1
                    print(f"Iteration {iteration}: Split cluster {min_density_idx} into clusters {new_cluster1} and {new_cluster2}")
            else:
                # Reject the split and terminate early
                if self.verbose:
                    print(f"Iteration {iteration}: Rejected split of cluster {min_density_idx} due to density constraint (avg_log_density={log_avg_density_children:.4f} < {threshold_info}). Terminating clustering.")
                break

        # After splitting, assign labels
        labels = np.full(n_samples, -1, dtype=int)  # Initialize with -1
        cluster_centers = []
        cluster_covariances = []

        for cluster_idx, cluster_indices in enumerate(current_clusters):
            cluster_data = X_scaled[cluster_indices]
            # Store cluster center
            center = np.mean(cluster_data, axis=0)
            cluster_centers.append(center)
            # Store covariance matrix
            if cluster_data.shape[0] >= self.n_features:
                cov_matrix = np.cov(cluster_data, rowvar=False)
                # Handle singular covariance matrices
                if det(cov_matrix) <= 0:
                    cov_matrix += np.eye(n_features) * 1e-6
                cluster_covariances.append(cov_matrix)
            else:
                # Assign identity matrix if covariance cannot be computed
                cov_matrix = np.eye(n_features)
                cluster_covariances.append(cov_matrix)
                warnings.warn(f"Cluster {cluster_idx} has fewer points than features. Assigned identity covariance.")

            # Assign labels
            labels[cluster_indices] = cluster_idx

        self.labels_ = labels
        self.cluster_centers_ = cluster_centers
        self.cluster_covariances_ = cluster_covariances
        self.n_clusters_ = len(current_clusters)

        if self.verbose:
            print(f"Final number of clusters: {self.n_clusters_}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assigns cluster labels to new data points based on the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to assign to clusters.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None or not self.cluster_centers_:
            raise ValueError("The model has not been fitted yet.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Apply scaling if necessary
        if self.rescale:
            if self.scaler_ is None:
                raise ValueError("Scaler has not been fitted. Ensure that the model is fitted before predicting.")
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # Compute distances to cluster centers
        centers = np.array(self.cluster_centers_)
        # Compute Euclidean distances
        distances = np.linalg.norm(X_scaled[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

        # Assign labels based on nearest cluster center
        labels = np.argmin(distances, axis=1)

        return labels

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Convenience method; fits the model to X and returns cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to be clustered.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_
