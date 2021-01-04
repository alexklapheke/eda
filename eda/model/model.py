"""
File: model.py
Author: Alex Klapheke
Email: alexklapheke@gmail.com
Github: https://github.com/alexklapheke
Description: Unsupervised modeling tools for EDA

Copyright © 2020 Alex Klapheke

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import silhouette_score


class DBSCAN(BaseEstimator, ClusterMixin):
    """Memory-light implementation of DBSCAN. Unlike sklearn, does not
    precompute a distance matrix, trading off memory for time. The value
    -1 is assigned to points that don't fit into a cluster. Example usage:

        dbscan = DBSCAN()
        df["cluster"] = dbscan.fit_predict(df)

    You can pass the following options to the instance declaration:

        eps:         The maximum distance to another point for it to be
                     considered part of the same cluster. Default: 0.5
        min_samples: The minimum number of points to be considered a cluster.
                     Default: 5
        p:           The p-norm to use for distance metric. 1 is equivalent to
                     taxicab distance. 2 is equivalent to Euclidean distance.
                     Default: 2

    The .fit() method adds the following properties:

        .is_fit:      This is set to True
        .n_clusters_: The number of clusters, not including the "noise cluster"
        .labels_:     A list of cluster labels corresponding to each value in
                      the data passed to .fit()
        .silhouette_: The silhouette score of the clustering, from -1 (worst)
                      to 1 (best)"""

    def __init__(self, eps=0.5, min_samples=5, p=2):
        # User-set
        self.eps = eps
        self.min_samples = min_samples
        self.p = p

        # Built-in
        self.noise_label = -1

        # Initialize
        self._labels = dict()
        self.is_fit = False

    def _check_fit(self):
        if not self.is_fit:
            raise NotFittedError("You must fit the model to data first!")

    @staticmethod
    def _metric(x1, x2, p=2):
        """Distance metric of order p"""
        return (np.sum(np.abs((x1-x2)**p), axis=(x1.ndim-1)))**(1/p)

    @staticmethod
    def _k(key):
        """Use a numpy array as a dictionary key"""
        return hash(key.data.tobytes())

    def fit(self, X, y=None):
        cluster = -1
        X = np.array(X)

        for p in X:

            # If already classified, don't bother
            if self._k(p) in self._labels:
                continue

            # Find neighbors
            neighbors = self._metric(X, p, self.p) < self.eps

            # Assign to current cluster, or to "noise cluster"
            if X[neighbors].shape[0] >= self.min_samples:
                cluster += 1
                self._labels[self._k(p)] = cluster
            else:
                self._labels[self._k(p)] = self.noise_label
                continue

            while X[neighbors].shape[0] > 0:
                new_neighbors = np.repeat(False, neighbors.shape)

                # Assign new neighbors to current cluster
                for q in X[neighbors]:
                    if self._k(q) not in self._labels or self._k(q) == -1:
                        self._labels[self._k(q)] = cluster
                        new_neighbors |= self._metric(X, q, self.p) < self.eps

                # Add to running list and remove from "new" list
                neighbors = new_neighbors.copy()

        self.is_fit = True
        self.n_clusters_ = cluster + 1
        self.labels_ = self.predict(X)
        try:
            self.silhouette_ = silhouette_score(X, self.labels_)
        except ValueError:  # Only one cluster
            self.silhouette_ = np.nan

    def predict(self, X):
        self._check_fit()
        return np.array([self._labels[self._k(p)] for p in np.array(X)])

    def score(self):
        self._check_fit()
        return self.silhouette_
