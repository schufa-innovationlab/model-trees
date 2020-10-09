"""
This module defines classes for different split criteria.
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

from abc import ABCMeta, abstractmethod
from ._nodes import Split

import warnings
import numpy as np


# Algorithm constants
_EPS = 0.001


class BaseSplitCriterion(metaclass=ABCMeta):
    """
    Base Class for split criteria.
    """
    def __call__(self, X, y, model, mt):
        """
        Finds the best split point for a tree node based on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        model
            Weak Model trained on all samples of the current node
        mt
            Model Tree for which the split criterion is used
        Returns
        -------
        split: Split
            The new split. This can be used to grow the model tree
        gain:
            The approximated gain by using the split.

        Warnings
        --------
        The class and this method are still under development and they might undergo heavy change in the future.

        """
        return self.find_best_split(X, y, model, mt)

    @abstractmethod
    def find_best_split(self, X, y, model, mt):
        """
        Finds the best split point for a tree node based on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        model
            Weak Model trained on all samples of the current node
        mt
            Model Tree for which the split criterion is used
        Returns
        -------
        split: Split
            The new split. This can be used to grow the model tree
        gain:
            The approximated gain by using the split.

        Warnings
        --------
        The class and this method are still under development and they might undergo heavy change in the future.

        """
        pass


class GradientSplitCriterion(BaseSplitCriterion):
    """
    A gradient-based split criterion for model trees.

    References
    ----------
    .. [1] Broelemann, K. and Kasneci, G.,
       "A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees",
       Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2019
    """

    def __init__(self, is_renormalizing):
        """
        Parameters
        ----------
        is_renormalizing: bool
            A flag that indicates whether a renormalization shall be performed

        """
        self.is_renormalizing = is_renormalizing

    def find_best_split(self, X, y, model, mt):
        """
        Finds the best split point for a tree node based on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        model
            Weak Model trained on all samples of the current node
        mt
            Model Tree for which the split criterion is used
        Returns
        -------
        split: Split
            The new split. This can be used to grow the model tree
        gain:
            The approximated gain by using the split.

        Warnings
        --------
        The class and this method are still under development and they might undergo heavy change in the future.

        """
        # Size of the sample set
        n_samples, n_features = np.shape(X)

        # Compute gradient for all samples
        g = mt.gf_(model, X, y)

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
            filter = np.minimum(n_left, n_right) >= mt.min_samples_split
            splits = splits[filter]
            n_left = n_left[filter]
            n_right = n_right[filter]

            if len(splits) < 1:
                # No split found after filtering
                continue

            # Compute the sum of gradients for the left side
            g_sum_left = g[s_idx, :].cumsum(axis=0)
            g_sum_left = g_sum_left[splits - 1, :]

            # Compute the sum of gradients for the right side
            g_sum_right = g_sum - g_sum_left

            # Additional code needed for renormalization
            if self.is_renormalizing:
                # Renormalization with z-transform:
                #   compute mean and variance for each potential left and right child (i.e. for each split)

                # For computational stability.
                # See:
                #   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data
                #   with K = mu

                # Total mean as row vector (for shifting the input)
                mu = np.reshape(np.mean(X, axis=0), (1, -1))

                # reshape for broadcasting purposes
                n_l = np.reshape(n_left, (-1, 1))
                n_r = np.reshape(n_right, (-1, 1))

                # Reorder and shift X to efficiently compute variance and mean
                Xrs = X[s_idx] - mu

                # Compute cumulative sum
                #   (i.e. for each split the sum of features of samples that go to the left side)
                Xcs_l = Xrs.cumsum(axis=0)

                # Compute the corresponding sums for the right side:
                Xcs_r = Xcs_l[-1:, :] - Xcs_l  # here Xcs_l[-1:,:] is the sum of all samples

                # Compute mean of left and right side for all splits
                mu_l = Xcs_l[splits - 1, :] / n_l
                mu_r = Xcs_r[splits - 1, :] / n_r

                # Equivalently, compute the sum of the squares (as preparation for the variance computation)
                X2cs_l = np.power(Xrs, 2).cumsum(axis=0)
                X2cs_r = X2cs_l[-1:, :] - X2cs_l

                # Compute standard deviation of left and right side for all splits
                sigma_l = np.sqrt(np.maximum(X2cs_l[splits - 1, :] / (n_l - 1) - np.power(mu_l, 2), _EPS ** 2))
                sigma_r = np.sqrt(np.maximum(X2cs_r[splits - 1, :] / (n_r - 1) - np.power(mu_r, 2), _EPS ** 2))

                # Correct for previous shift (it was only done on X, not on the gradients)
                # NOTE: This needs to be done AFTER computing sigma with shifted values.
                mu_l = mu_l + mu
                mu_r = mu_r + mu

                # Renormalize gradients
                g_sum_left = mt.rnf_(model, g_sum_left, 1 / sigma_l, -mu_l / sigma_l)
                g_sum_right = mt.rnf_(model, g_sum_right, 1 / sigma_r, -mu_r / sigma_r)

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
