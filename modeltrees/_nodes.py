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

import numpy as np


class TreeNode:
    """
    A helper class to store the tree structure of a model tree.

    Do not instantiate this class directly, but used the model tree classes

    Parameters
    ----------
    parent_node : TreeNode or None
        Parent node or None if this is a root node
    estimator : object
        Base estimator of the node.
        This estimator is used in leaf nodes for predictions, but can also be stored in other nodes.
    children : list or None
        List of child nodes. Should have 2 or 0 elements or be None.
    split : Split
        Defines, how samples are split (and mapped) to the child nodes.

    Attributes
    ----------
    parent_node : TreeNode or None
        Parent node or None if this is a root node
    depth : int, (default=0)
        Zero-based depth of the node in the tree
    estimator : object
        Base estimator of the node.
        This estimator is used in leaf nodes for predictions, but can also be stored in other nodes.
    children : list or None
        List of child nodes. Should have 2 or 0 elements or be None.
    split : Split
        Defines, how samples are split (and mapped) to the child nodes.

    See Also
    --------
    modeltrees.tree.BaseModelTree : Base Model Tree implementation
    Split : Class that defines how split / mapping to the child nodes

    Notes
    -----
    This is not a sklearn estimator class, but a helper class

    """

    def __init__(self, parent_node=None, estimator=None, children=None, split=None):
        self.parent_node = parent_node
        self.estimator = estimator
        self.children = children
        self.split = split

        if parent_node is None:
            self.depth = 0
        else:
            self.depth = parent_node.depth + 1

    def is_leaf(self):
        """
        Checks, if the node is a leaf node, i.e. no split is set.

        Returns
        -------
        True, if the node is a leaf node.
        """
        return self.split is None

    def is_root(self):
        """
        Checks, if the node is a root node, i.e. no parent_node is set.

        Returns
        -------
        True, if the node is a root node.
        """
        return self.parent_node is None

    def map_to_leaf(self, X):
        """
        Maps input samples to leaf nodes by using split rules and the subtree structure

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the samples

        Returns
        -------
        leaf_idx: array-like, shape = [n_samples]
            For each sample an index of the corresponding leaf node.
        leafs: list
            A list of leaf nodes. Positions correspond to the indices in `leaf_idx`

        """

        if self.is_leaf():
            return np.zeros(np.shape(X)[0], dtype=int), [self]
        else:
            child_idx = self.split.map_to_children(X)
            leaf_idx = -np.ones(child_idx.shape, dtype=int)

            leafs = []

            # Iterate over children
            for c in range(len(self.children)):
                # Get sample subset for child c
                idx = child_idx == c

                if np.any(idx):
                    # Recursively map to leafs
                    leaf_idx_, leafs_ = self.children[c].map_to_leaf(X[idx])

                    # Include results into output leaf_idx
                    # Note that we need to shift the index to avoid return the same leaf index for different leafs.
                    shift = len(leafs)
                    leaf_idx[idx] = leaf_idx_ + shift

                    # Append the new found leafs
                    leafs = leafs + leafs_

            # Return results
            return leaf_idx, leafs

    def get_path(self):
        """
        Gets the path from the root to this node

        Returns
        -------
        path : list of TreeNode
            The list of TreeNodes along the path from the root to this node
        """
        path = []

        node = self
        while node is not None:
            path.insert(0, node)
            node = node.parent_node

        return path


class Split:
    """
    Defines a splitting of a decision / model tree node, i.e. the mapping of samples to the child node.

    This class supports splits based on one feature and threshold.
    All samples with a feature value (in the given feature) less or equal to the threshold are mapped to child 0.
    All others are mapped to child 1.

    Parameters
    ----------
    split_feature : int
        Index of the feature that is used for the split
    split_threshold : float
        Threshold for the split.

    Attributes
    ----------
    split_feature : int
        Index of the feature that is used for the split
    split_threshold : float
        Threshold for the split.
    """
    def __init__(self, split_feature, split_threshold):
        self.split_feature = split_feature
        self.split_threshold = split_threshold

    def _apply_split(self, X, y = None):
        """
        Splits a set samples according to the defines split rule in split.


        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the samples
        y : array-like, shape = [n_samples] or [n_samples, n_outputs], optional
            Target variable.

        Returns
        -------
        subsets: list
            A list of Subsets. If `y` is `None`, each element `i` is an array with [n_samples[i], n_features].
            Otherwise each element is a pair of input features and target variable.

        """
        # Check for left subtree
        split_filter = X[:, self.split_feature] <= self.split_threshold

        # Output depending in input
        if y is None:
            return [X[split_filter], X[~split_filter]]
        else:
            return [
                (X[split_filter], y[split_filter]),  # Samples for the left subtree
                (X[~split_filter], y[~split_filter])  # Samples for the right subtree
            ]

    def map_to_children(self, X):
        """
        Maps samples to child nodes. This is done based on the split feature and threshold

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the samples

        Returns
        -------
        child_idx: array-like, shape = [n_samples]
            For each sample an index (0 for left child, 1 for right child).
        """
        child_idx = 1 - (X[:, self.split_feature] <= self.split_threshold)
        return child_idx
