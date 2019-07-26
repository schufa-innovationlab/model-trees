"""
This module defines model tree estimator classes for classification and regression.
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
                 min_samples_split=10):
        """

        Parameters
        ----------
        base_estimator:
            Base estimator to be used in the nodes
        max_depth : int (default = 3)
            Maximal depth of the tree
        min_samples_split : int (default = 10)
            Minimal number of samples that go to each split
        """
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

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

        # Check input and store dimensions
        self._validate_training_data(X, y)

        # Create the three structure
        self.root_ = self._create_tree_structure(X, y)

        return self

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
            split = self._find_split(X, y)

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

        else:
            # Create Leave node if maximal depth is reached
            return TreeNode(depth=depth, estimator=estimator)

    def _find_split(self, X, y):
        """
        Finds the optimal split point for a tree node based on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.

        Returns
        -------
        split: Split
            The new split. This can be used to create a new

        """
        # TODO: implement me
        # Dummy implementation: Random split
        feat_idx = np.random.randint(self.n_features_)
        sample_idx = np.random.randint(np.shape(X)[0])

        return Split(feat_idx, X[sample_idx, feat_idx])
