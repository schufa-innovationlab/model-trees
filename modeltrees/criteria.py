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
from scipy.stats import entropy
from sklearn.base import BaseEstimator


# Algorithm constants
_EPS = 0.001


class WithParamsMixin:
    """
    A mixin that that adds scikit-learn parameters to a non-estimator class.
    This is helpful for nested objects, e.g. split criteria with hyper-parameters.
    """
    def _get_param_names(self):
        return []

    def get_params(self, deep=True):
        """
        List of parameters of the object.

        Parameters
        ----------
        deep: bool
            If true, the parameters of nested objects will also be added.

        Returns
        -------
        dict
            Dictionary of parameter-value pairs.
        """
        return {name:getattr(self,name) for name in self._get_param_names()}

    # Copying the set_parameter behaviour of sklearn estimators.
    set_params = BaseEstimator.set_params


class BaseSplitCriterion(WithParamsMixin,metaclass=ABCMeta):
    """
    Base Class for split criteria.
    """
    def __call__(self, X, y, estimator, mt):
        """
        Finds the best split point for a tree node based on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        estimator
            Weak Estimator trained on all samples of the current node
        mt
            Model Tree for which the split criterion is used
        Returns
        -------
        split: Split
            The new split. This can be used to grow the model tree
        gain: float
            The approximated gain by using the split.

        Warnings
        --------
        The class and this method are still under development and they might undergo heavy change in the future.

        """
        return self.find_best_split(X, y, estimator, mt)

    def find_best_split(self, X, y, estimator, mt):
        """
        Finds the best split point for a tree node based on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        estimator
            Weak Estimator trained on all samples of the current node
        mt
            Model Tree for which the split criterion is used
        Returns
        -------
        split: Split
            The new split. This can be used to grow the model tree
        gain: float
            The approximated gain by using the split.

        Warnings
        --------
        The class and this method are still under development and they might undergo heavy change in the future.

        """
        # Size of the sample set
        n_features = np.shape(X)[1]

        split_info = self._precompute_splits(X, y, estimator, mt)

        # Compute split criterion on all possible splits
        best_split = None
        max_gain = -np.inf
        for i in range(n_features):
            # Compute split thresholds and gain
            gain, thresh = self.compute_split_gain_by_feature(X, y, i, estimator, mt, split_info)

            # Check if splits could be found
            if len(gain) < 1:
                continue

            # Find maximal gain, if a split is done by this feature
            best_idx = np.argmax(gain)
            gain = gain[best_idx]

            # Compare with previously found gains
            if gain > max_gain:
                max_gain = gain
                best_split = Split(
                    split_feature=i,
                    split_threshold=thresh[best_idx]
                )

        return best_split, max_gain

    def compute_split_gain_by_feature(self, X, y, feature_id, estimator, mt, split_info = None):
        """
        This method identifies split candidates along one feature axis
        and computes the gain for each of these potential splits.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        estimator
            Weak Model trained on all samples of the current node
        mt
            Model Tree for which the split criterion is used
        split_info: list
            Precomputed values.

        Returns
        -------
        gain: array-like
            The gain for all split candidates
        thresh:
            The split-threshold for all candidates

        """
        # Sort along feature i
        sort_idx = np.argsort(X[:, feature_id])

        # Get Split Candidates
        n_left, n_right, splits = self._get_split_candidates(X, y, feature_id, mt, sort_idx)

        if len(splits) < 1:
            # No split candidate found
            return [], []

        if split_info is None:
            # No precomputed values are provided
            split_info = self._precompute_splits(X, y, estimator, mt)

        # Compute the split gain along an axis
        gain = self._compute_split_gain_by_feature(X, y, feature_id, estimator, mt, n_left, n_right, sort_idx, splits, *split_info)

        # Compute thresholds
        thresh = self._get_threshold(X, feature_id, sort_idx, splits)

        return gain, thresh

    def _get_threshold(self, X, feature_id, sort_idx, splits):
        """
        Get threshold inbetween the smallest right and larges left sample.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        sort_idx: array-like
            Permutation to order X and y according to the currently analysed axis
        splits: array-like
            Zero-based index of the split candidates.
            A split candidate (with index j) maps samples ``sort_idx[:splits[j]]`` to the left and
            samples ``sort_idx[splits[j]:]`` to the right child node

        Returns
        -------

        """
        upper = X[sort_idx[splits], feature_id]
        lower = X[sort_idx[splits - 1], feature_id]
        thresh = (upper + lower) / 2
        return thresh

    def _precompute_splits(self, X, y, estimator, mt):
        """
        This method allows to precompute some information that is the same for each axis.
        An example are gradients on for the ``GradientSplitCriterion``

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        estimator
            The trained weak estimator in a node.
        mt
            A model tree instance

        Returns
        -------
        precomputed_data: list
            A list of precomputed values. Which values these are is defined in the concrete split criterion sub class.

        Warnings
        --------
        The class and this method are still under development and they might undergo heavy change in the future.
        """
        return []

    def _get_split_candidates(self, X, y, feature_id, mt, sort_idx):
        """
        This methods identifies split points candidates along one axis.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        mt:
            The calling model tree object
        sort_idx: array-like
            Permutation to order X and y according to the currently analysed axis

        Returns
        -------
        n_left: array-like
            For each split the number of elements on the left side
        n_right: array-like
            For each split the number of elements on the right side
        splits: array-like
            Zero-based index of the split candidate.
            A split candidate (with index j) maps samples ``xi[:splits[j]]`` to the left and
            samples ``xi[splits[j]:]`` to the right child node

        Warnings
        --------
        The class and this method are still under development and it might undergo heavy change in the future.
        """

        # Find unique values along one column.
        #   splits : zero-base index of the first occurrence the unique elements in Xi (works, because Xi is sorted)
        #            This also indicates potential splits, since these are the points where in the sorted Xi the
        #            value changes.
        #            A potential split (with index j) maps samples s_idx[:splits[j]] to the left and
        #            samples s_idx[splits[j]:] to the right child node

        _, splits = np.unique(X[sort_idx, feature_id], return_index=True)

        if len(splits) <= 1:
            # No split possible if there is only one value
            return [], [], []

        # Compute for each split the number of samples in the left and right subtree
        n_samples = X.shape[0]
        n_left = splits
        n_right = n_samples - n_left

        # Ignore all splits where one child has less than `min_samples_split` training samples
        filter = np.minimum(n_left, n_right) >= mt.min_samples_split
        splits = splits[filter]
        n_left = n_left[filter]
        n_right = n_right[filter]

        return n_left, n_right, splits

    @abstractmethod
    def _compute_split_gain_by_feature(self, X, y, feature_id, estimator, mt, n_left, n_right, sort_idx, splits, *args):
        """
        Core method of the split criterion.
        This method computes an approximation of the gain for each split point candidate along one axis.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        estimator
            Weak estimator trained on `X` and `y`
        mt
            Model Tree Object
        n_left: array-like
            Contains each split candidate the number of samples for the left child
        n_right: array-like
            Contains each split candidate the number of samples for the right child
        sort_idx: array-like
            Permutation to order X and y according to the currently analysed axis
        splits: array-like
            Zero-based index of the split candidates.
            A split candidate (with index j) maps samples ``sort_idx[:splits[j]]`` to the left and
            samples ``sort_idx[splits[j]:]`` to the right child node
        args: list
            Precomputed values.

        Returns
        -------
        gain: array-like
            The approximate gain per split candidate

        Warnings
        --------
        The class and this method are still under development and it might undergo heavy change in the future.
        """
        pass

    def _get_param_names(self):
        return []


class GradientSplitCriterion(BaseSplitCriterion):
    """
    A gradient-based split criterion for model trees.

    References
    ----------
    .. [1] Broelemann, K. and Kasneci, G.,
       "A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees",
       Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2019
    """

    def _precompute_splits(self, X, y, estimator, mt):
        """
        Precomputes the gradients for each sample

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        estimator
            The trained weak estimator in a node.
        mt
            A model tree instance

        Returns
        -------
        g: array-like, shape = [n_samples, n_parameters]
            Gradients for the input samples
        g_sum: array-like, shape=[n_parameters]
            Sum of gradients.

        Warnings
        --------
        The class and this method are still under development and it might undergo heavy change in the future.
        """
        # Compute gradient for all samples
        g = mt.gf_(estimator, X, y)

        # Compute the sum of the gradients (for a perfectly trained model this should be 0)
        g_sum = g.sum(axis=0)

        return g, g_sum

    def _compute_split_gain_by_feature(self, X, y, feature_id, estimator, mt, n_left, n_right, sort_idx, splits, g, g_sum):
        """
        This method computes an approximation of the gain using the gradient split criterion.
        The computation is done for each split point candidate along one feature-axis.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        estimator
            Weak estimator trained on `X` and `y`
        mt
            Model Tree Object
        n_left: array-like
            Contains each split candidate the number of samples for the left child
        n_right: array-like
            Contains each split candidate the number of samples for the right child
        sort_idx: array-like
            Permutation to order X and y according to the currently analysed axis
        splits: array-like
            Zero-based index of the split candidates.
            A split candidate (with index j) maps samples ``sort_idx[:splits[j]]`` to the left and
            samples ``sort_idx[splits[j]:]`` to the right child node
        g: array-like, shape = [n_samples, n_parameters]
            Gradients for the input samples
        g_sum: array-like, shape=[n_parameters]
            Sum of gradients.

        Returns
        -------
        gain: array-like
            The approximate gain per split candidate

        Warnings
        --------
        The class and this method are still under development and it might undergo heavy change in the future.
        """
        # Compute the sum of gradients for the left side
        g_sum_left = g[sort_idx, :].cumsum(axis=0)
        g_sum_left = g_sum_left[splits - 1, :]

        # Compute the sum of gradients for the right side
        g_sum_right = g_sum - g_sum_left

        # Renormalize Gradients (if renormalization is implemented)
        g_sum_left, g_sum_right = self._renormalize_gradients(X, estimator, g_sum_left, g_sum_right, mt, n_left, n_right, sort_idx, splits)

        # Compute the Gain (see Eq. (6) in [1])
        gain = np.power(g_sum_left, 2).sum(axis=1) / n_left + np.power(g_sum_right, 2).sum(axis=1) / n_right

        return gain

    def _renormalize_gradients(self, X, estimator, g_sum_left, g_sum_right, mt, n_left, n_right, sort_idx, splits):
        return g_sum_left, g_sum_right


class ZRenormGradientSplitCriterion(GradientSplitCriterion):
    def _renormalize_gradients(self, X, estimator, g_sum_left, g_sum_right, mt, n_left, n_right, sort_idx, splits):
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
        Xrs = X[sort_idx] - mu

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
        sigma_l = np.sqrt(np.maximum( (X2cs_l[splits - 1, :] - np.power(mu_l, 2) * n_l) / (n_l - 1), _EPS ** 2))
        sigma_r = np.sqrt(np.maximum( (X2cs_r[splits - 1, :] - np.power(mu_r, 2) * n_r) / (n_r - 1), _EPS ** 2))

        # Correct for previous shift (it was only done on X, not on the gradients)
        # NOTE: This needs to be done AFTER computing sigma with shifted values.
        mu_l = mu_l + mu
        mu_r = mu_r + mu

        # Renormalize gradients
        g_sum_left = mt.rnf_(estimator, g_sum_left, 1 / sigma_l, -mu_l / sigma_l)
        g_sum_right = mt.rnf_(estimator, g_sum_right, 1 / sigma_r, -mu_r / sigma_r)

        return g_sum_left, g_sum_right


class SumOfSquareErrorSplitCriterion(BaseSplitCriterion):
    """
    Implementation of the typical regression tree split criterion.
    For each split candidate, the reduction of the sum of square errors is considers.

    Notes
    -----
    The sum of square error criterion is meant to be used with regression trees.
    """
    def _compute_split_gain_by_feature(self, X, y, feature_id, estimator, mt, n_left, n_right, sort_idx, splits, *args):
        """
        This method computes an approximation of the gain using the gradient split criterion.
        The computation is done for each split point candidate along one feature-axis.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        estimator
            Weak estimator trained on `X` and `y`
        mt
            Model Tree Object
        n_left: array-like
            Contains each split candidate the number of samples for the left child
        n_right: array-like
            Contains each split candidate the number of samples for the right child
        sort_idx: array-like
            Permutation to order X and y according to the currently analysed axis
        splits: array-like
            Zero-based index of the split candidates.
            A split candidate (with index j) maps samples ``sort_idx[:splits[j]]`` to the left and
            samples ``sort_idx[splits[j]:]`` to the right child node

        Returns
        -------
        gain: array-like
            The approximate gain per split candidate

        Warnings
        --------
        The class and this method are still under development and it might undergo heavy change in the future.
        """
        # Reshape y, if necessary
        y = np.reshape(y, (-1))
        y = y - np.mean(y)

        n = len(y)

        # Total sum of y
        ys = y.sum()

        # Cumulative sum of y
        ycs_l = y[sort_idx].cumsum()
        ycs_r = ys - ycs_l

        # Cumulative sum of square y
        y2cs_l = np.power(y[sort_idx],2).cumsum()
        y2cs_r = y2cs_l[-1] - y2cs_l

        # Compute sum of square error without split
        sse = y2cs_l[-1] - n * np.power(ycs_l[-1]/n,2)

        # Compute sum of square error on the children
        sse_l = y2cs_l[splits] - n_left * np.power(ycs_l[splits] / n_left, 2)
        sse_r = y2cs_r[splits] - n_right * np.power(ycs_r[splits] / n_right, 2)

        # And the gain
        gain = sse - sse_l - sse_r
        return gain


class CrossEntropySplitCriterion(BaseSplitCriterion):
    """
    Implementation of the entropy-based split criterion for decision tree for classification tasks.
    For each split candidate, the reduction of the sum of square errors is considers.

    Notes
    -----
    This Criterion is only suitable for classification tasks
    """
    def _compute_split_gain_by_feature(self, X, y, feature_id, estimator, mt, n_left, n_right, sort_idx, splits, *args):
        """
        This method computes an approximation of the gain using the gradient split criterion.
        The computation is done for each split point candidate along one feature-axis.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.
        feature_id : int
            ID of the feature, i.e. splitting along ``X[:,feature_id]``
        estimator
            Weak estimator trained on `X` and `y`
        mt
            Model Tree Object
        n_left: array-like
            Contains each split candidate the number of samples for the left child
        n_right: array-like
            Contains each split candidate the number of samples for the right child
        sort_idx: array-like
            Permutation to order X and y according to the currently analysed axis
        splits: array-like
            Zero-based index of the split candidates.
            A split candidate (with index j) maps samples ``sort_idx[:splits[j]]`` to the left and
            samples ``sort_idx[splits[j]:]`` to the right child node

        Returns
        -------
        gain: array-like
            The approximate gain per split candidate

        Warnings
        --------
        The class and this method are still under development and it might undergo heavy change in the future.
        """
        # Target Vector vs. Target Matrix
        target_1d = len(np.shape(y)) == 1

        # Number of samples
        n = np.shape(y)[0]

        # Total sum of y
        ys = y.sum(axis=0)

        # Cumulative sum of y
        ycs_l = y[sort_idx].cumsum(axis=0)
        ycs_r = ys - ycs_l

        # Compute probabilities
        p_l = ycs_l[splits] / n_left
        p_r = ycs_r[splits] / n_right
        p = ys / n

        # Compute Gain
        if target_1d:
            # Binary classification; just class 1 probability is stored in p, p_l, p_r
            gain = entropy([p, 1-p]) - entropy([p_l, 1-p_l], axis=0) - entropy([p_r, 1-p_r], axis=0)
        else:
            # Propbabilities for all classes are stored in p, p_l, p_r along axis 1
            gain = entropy(p) - entropy(p_l, axis=1) - entropy(p_r, axis=1)

        return gain
