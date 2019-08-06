"""
This module defines model tree estimator classes for classification and regression.

References
----------
.. [1] Broelemann, K. and Kasneci, G.,
   "A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees",
   Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2019
"""

#  Copyright 2019 SCHUFA Holding AG
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from sklearn.base import MetaEstimatorMixin, BaseEstimator, clone
from abc import ABCMeta, abstractmethod

import numpy as np

from ._nodes import TreeNode, Split
from ._gradients import get_default_gradient_function


class BaseModelTree(BaseEstimator, MetaEstimatorMixin, metaclass=ABCMeta):
    """
    Base class for all model tree classes.

    Do not use this class directly, but instantiate derived classes.
    """
    root_: TreeNode
    n_features_: int
    n_outputs_: int

    _required_parameters = []

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 max_depth=3,
                 min_samples_split=10,
                 gradient_function=None):
        """

        Parameters
        ----------
        base_estimator:
            Base estimator to be used in the nodes
        max_depth : int (default = 3)
            Maximal depth of the tree
        min_samples_split : int (default = 10)
            Minimal number of samples that go to each split
        gradient_function
            A function that computes the gradient of a model at a given point.
            The gradient_function gets 3 parameters: a fitted model (see base_estimator),
            the input matrix X and the target vector y.
        """
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.gradient_function = gradient_function

    def fit(self, X, y):
        """
        Train a model tree on the provided training data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.

        Returns
        -------
        self : object
        """

        # Check model parameters
        self._validate_parameters()

        # Check input and store dimensions
        self._validate_training_data(X, y)

        # Create the three structure
        self.root_ = self._create_tree_structure(X, y)

        return self

    def _validate_parameters(self):
        """
        Validates the provided model parameters.

        Returns
        -------

        """
        # Check gradient function and try to get default
        if self.gradient_function is None:
            self.gf_ = get_default_gradient_function(self.base_estimator)
        else:
            self.gf_ = self.gradient_function

    def _validate_training_data(self, X, y):
        """
        Validates the training data and stores the input and output dimension

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.

        Returns
        -------

        """
        # TODO: check input type and shape
        self.n_features_ = np.shape(X)[1]
        if np.ndim(y) < 2:
            self.n_outputs_ = None
        else:
            self.n_outputs_ = np.shape(y)[1]

    def _create_tree_structure(self, X, y, depth=0):
        """
        Recursively creates the model (sub-)tree structure with respect to the provided training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        depth: int
            Zero-Based depth of the node
        Returns
        -------
        root : TreeNode
            Root node of the created (sub-)tree

        """

        # Create and train base estimator
        estimator = clone(self.base_estimator)
        estimator.fit(X, y)

        # Only split, if the maximal depth is not reached, yet
        if depth < self.max_depth:
            # Find best split
            try:
                split, gain = self._find_split(estimator, X, y)

                # Split trainings data and create child nodes from the
                child_data = split._apply_split(X, y)

                # (Recursively) create child nodes
                children = [self._create_tree_structure(cX, cy, depth=depth + 1) for cX, cy in child_data]

                return TreeNode(
                    depth=depth,
                    estimator=estimator,
                    children=children,
                    split=split
                )
            except:
                # In Case of errors: Create Leave node
                return TreeNode(depth=depth, estimator=estimator)

        else:
            # Create Leave node if maximal depth is reached
            return TreeNode(depth=depth, estimator=estimator)

    def _find_split(self, model, X, y):
        """
        Finds the optimal split point for a tree node based on the training data.

        Parameters
        ----------
        model
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.

        Returns
        -------
        split: Split
            The new split. This can be used to grow the model tree
        gain:
            The approximated gain by using the split.

        """
        # Size of the sample set
        n_samples, n_features = np.shape(X)

        # Compute gradient for all samples
        g = self.gf_(model, X, y)

        # Compute the sum of the gradients (for a perfectly trained model this should be 0)
        g_sum = g.sum(axis=0)

        # Compute split criterion on all possible splits
        best_split = None
        max_gain = -np.inf
        for i in range(n_features):
            # Sort along feature i
            s_idx = np.argsort(X[:, i])
            Xi = X[s_idx, i]

            # Find unique values along one column.
            #   u_Xi   : the sorted unique feature values for feature / column i
            #   splits : zero-base index of the first occurrence the unique elements in Xi (works, because Xi is sorted)
            #            This also indicates potential splits, since these are the points where in the sorted Xi the
            #            value changes.
            #            A potential split (with index j) maps samples s_idx[:splits[j]] to the left and
            #            samples s_idx[splits[j]:] to the right child node
            u_Xi, splits = np.unique(Xi, return_index=True)

            if len(u_Xi) <= 1:
                # No split possible if there is only one value
                continue

            # splits[0] = 0 is no real split
            splits = splits[1:]

            # Compute for each split the number of samples in the left and right subtree
            n_left = splits
            n_right = n_samples - n_left

            # Ignore all splits where one child has less than `min_samples_split` training samples
            filter = np.minimum(n_left, n_right) >= self.min_samples_split
            splits = splits[filter]
            n_left = n_left[filter]
            n_right = n_right[filter]

            # Compute the sum of gradients for the left side
            g_sum_left = g[s_idx, :].cumsum(axis=0)
            g_sum_left = g_sum_left[splits - 1, :]

            # Compute the sum of gradients for the right side
            g_sum_right = g_sum - g_sum_left

            # Compute the Gain (see Eq. (6) in [1])
            gain = np.power(g_sum_left, 2).sum(axis=1) / n_left + np.power(g_sum_right, 2).sum(axis=1) / n_right

            # Find maximal gain, if a split is done by this feature
            best_idx = np.argmax(gain)
            gain = gain[best_idx]

            # Compare with previously found gains
            if gain > max_gain:
                max_gain = gain
                best_split = Split(
                    split_feature=i,
                    split_threshold=(Xi[splits[best_idx] - 1] + Xi[splits[best_idx]]) / 2
                )

        return best_split, max_gain

