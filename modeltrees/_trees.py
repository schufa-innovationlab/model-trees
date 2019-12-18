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

from sklearn.base import MetaEstimatorMixin, BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np

from ._nodes import TreeNode, Split
from ._gradients import get_default_gradient_function, get_default_renormalization_function

# Parameter Constants
_CRITERION_GRADIENT = "gradient"
_CRITERION_GRADIENT_RENORM = "gradient-renorm-z"

_SUPPORTED_CRITERIA = {_CRITERION_GRADIENT, _CRITERION_GRADIENT_RENORM}

# Algorithm constants
_EPS = 0.001


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
                 criterion="gradient",
                 max_depth=3,
                 min_samples_split=10,
                 gradient_function=None,
                 renorm_function=None):
        """

        Parameters
        ----------
        base_estimator:
            Base estimator to be used in the nodes
        criterion : str
            Split Criterion. Supported values are: `"gradient"` and `"gradient-renorm-z"`
        max_depth : int (default = 3)
            Maximal depth of the tree
        min_samples_split : int (default = 10)
            Minimal number of samples that go to each split
        gradient_function : callable
            A function that computes the gradient of a model with respect to the model parameters at given points.
            The gradient_function gets 3 parameters: a fitted model (see base_estimator),
            the input matrix X and the target vector y.
        renorm_function : callable
            A function that allows to renormalize gradients of a model.
            The renorm_function gets 4 parameters: a fitted model (see base_estimator),
            the gradient matrix g, a scale factor and a shift vector.
            *The renormalization function must be set based on the `gradient_function`. Unsuitable functions might
            result in bad results, exceptions and unexpected outcomes.*
            **Note: Only set this parameter if you use a custom gradient function and know what you are doing.**
        """
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.gradient_function = gradient_function
        self.renorm_function = renorm_function

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
        # Check criterion
        if self.criterion not in _SUPPORTED_CRITERIA:
            msg = f"Invalid Split Criterion. Got '{self.criterion}'. Valid values are {_SUPPORTED_CRITERIA}"
            raise ValueError(msg)

        # Check gradient function and try to get default
        if self.gradient_function is None:
            self.gf_ = get_default_gradient_function(self.base_estimator)
        else:
            self.gf_ = self.gradient_function

        # Check renormalization function and try to get default
        if self.renorm_function is None:
            self.rnf_ = get_default_renormalization_function(self.base_estimator)
        else:
            if self.gradient_function is None:
                warnings.warn(
                    "Using a custom renormalization function with a standard gradient function will likely result in "
                    "undesired outcomes.",
                    UserWarning
                )
            self.rnf_ = self.renorm_function

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
        estimator = self._create_and_fit_estimator(X, y)

        # Only split, if the maximal depth is not reached, yet
        if depth < self.max_depth:
            # Find best split
            split, gain = self._find_split(estimator, X, y)

            # Did the algorithm find a split?
            if split is not None:
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
                # If not split was found, return es leaf node
                return TreeNode(depth=depth, estimator=estimator)
        else:
            # Create leaf node if maximal depth is reached
            return TreeNode(depth=depth, estimator=estimator)

    def _create_and_fit_estimator(self, X, y):
        """
        Creates a new estimator for a node and trains it with the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.

        Returns
        -------
        estimator
            The trained estimator
        """
        estimator = clone(self.base_estimator)
        estimator.fit(X, y)
        return estimator

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

            if len(splits) < 1:
                # No split found after filtering
                continue

            # Compute the sum of gradients for the left side
            g_sum_left = g[s_idx, :].cumsum(axis=0)
            g_sum_left = g_sum_left[splits - 1, :]

            # Compute the sum of gradients for the right side
            g_sum_right = g_sum - g_sum_left

            # Additional code needed for renormalization
            if self.criterion == _CRITERION_GRADIENT_RENORM:
                # Renormalization with z-transform:
                #   compute mean and variance for each potential left and right child (i.e. for each split)

                # For computational stability.
                # See:
                #   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data
                #   with K = mu

                # Total mean as row vector (for shifting the input)
                mu = np.reshape(np.mean(X, axis=0), (1, -1))

                # reshape for broadcasting purposes
                n_l = np.reshape(n_left, (-1,1))
                n_r = np.reshape(n_right, (-1, 1))

                # Reorder and shift X to efficiently compute variance and mean
                Xrs = X[s_idx] - mu

                # Compute cumulative sum
                #   (i.e. for each split the sum of features of samples that go to the left side)
                Xcs_l = Xrs.cumsum(axis=0)

                # Compute the corresponding sums for the right side:
                Xcs_r = Xcs_l[-1:,:] - Xcs_l   # here Xcs_l[-1:,:] is the sum of all samples

                # Compute mean of left and right side for all splits
                mu_l = Xcs_l[splits - 1, :] / n_l
                mu_r = Xcs_r[splits - 1, :] / n_r

                # Equivalently, compute the sum of the squares (as preparation for the variance computation)
                X2cs_l = np.power(Xrs, 2).cumsum(axis=0)
                X2cs_r = X2cs_l[-1:, :] - X2cs_l

                # Compute standard deviation of left and right side for all splits
                sigma_l = np.sqrt(np.maximum(X2cs_l[splits - 1, :] / (n_l - 1) - np.power(mu_l, 2), _EPS**2))
                sigma_r = np.sqrt(np.maximum(X2cs_r[splits - 1, :] / (n_r - 1) - np.power(mu_r, 2), _EPS**2))

                # Correct for previous shift (it was only done on X, not on the gradients)
                # NOTE: This needs to be done AFTER computing sigma with shifted values.
                mu_l = mu_l + mu
                mu_r = mu_r + mu

                # Renormalize gradients
                g_sum_left = self.rnf_(model, g_sum_left, 1/sigma_l, -mu_l/sigma_l)
                g_sum_right = self.rnf_(model, g_sum_right, 1/sigma_r, -mu_r/sigma_r)

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

    def _apply_sample_wise_function_on_leafs(self, fct, X):
        """
        Helper function that applies a function on the leaf estimators and returns the recombined results.

        This method is intended to avoid duplicate code for estimator methods such as
        `predict`, `predict_proba`, `predict_log_proba` and `decision_function`.

        Parameters
        ----------
        fct: callable
            A function that takes two parameters: an estimator and a matrix of samples (see also parameter `X`)
            The result must also be an array, where the first axis corresponds to samples.
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data

        Returns
        -------
        output: array
            The recombines results of the function calls on the leafs. The order of the input `X`is maintained.
            The shape is (except for the sample-axis 0) the same as for the call on the leafs.
        """
        # Convert to numpy array
        X = np.asarray(X)

        # Compute results on leafs
        idx, leafs = self.root_.map_to_leaf(X)

        results = [fct(leafs[i].estimator, X[idx == i])
                   for i in range(len(leafs))]

        # Get output shape of one leaf and extend it along axis 0
        shape = list(np.shape(results[0]))
        shape[0] = np.shape(X)[0]

        # Create output array
        dt = results[0].dtype
        output = np.zeros(shape, dtype=dt)

        # Copy leaf outputs into the finale output array
        for i in range(len(leafs)):
            output[idx == i] = results[i]

        return output

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the samples

        Returns
        -------
        O : array
            Prediction per sample
        """
        return self._apply_sample_wise_function_on_leafs(lambda estimator, X_: estimator.predict(X_), X)


class ModelTreeRegressor(BaseModelTree, RegressorMixin):
    """
    Model Tree implementation for regression problems.

    This algorithm uses the gradient-based split criterion [1]_ to create the model tree structure.
    Note that this criterion requires to compute gradients of the base estimators.

    Parameters
    ----------
    base_estimator
        Base estimator to be used in the nodes. This should be an scikit-learn compatible regressor.
    criterion : str
        Split Criterion. Supported values are: `"gradient"` and `"gradient-renorm-z"`
    max_depth : int (default = 3)
        Maximal depth of the tree
    min_samples_split : int (default = 10)
        Minimal number of samples that go to each split
    gradient_function : callable
        A function that computes the gradient of a model with respect to the model parameters at given points.
        The gradient_function gets 3 parameters: a fitted model (see base_estimator),
        the input matrix X and the target vector y.
    renorm_function : callable
        A function that allows to renormalize gradients of a model.
        The renorm_function gets 4 parameters: a fitted model (see base_estimator),
        the gradient matrix g, a scale factor and a shift vector.
        *The renormalization function must be set based on the `gradient_function`. Unsuitable functions might
        result in bad results, exceptions and unexpected outcomes.*
        **Note: Only set this parameter if you use a custom gradient function and know what you are doing.**

    Attributes
    ----------
    root_ : TreeNode
        Root Node of the tree structure

    References
    ----------
    .. [1] Broelemann, K. and Kasneci, G.,
       "A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees",
       Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2019
    """

    def __init__(self,
                 base_estimator=LinearRegression(),
                 criterion="gradient",
                 max_depth=3,
                 min_samples_split=10,
                 gradient_function=None,
                 renorm_function=None):
        super().__init__(
            base_estimator=base_estimator,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            gradient_function=gradient_function,
            renorm_function=renorm_function
        )


class ModelTreeClassifier(BaseModelTree, ClassifierMixin):
    """
    Model Tree implementation for classification problems.

    This algorithm uses the gradient-based split criterion [1]_ to create the model tree structure.
    Note that this criterion requires to compute gradients of the base estimators.

    Parameters
    ----------
    base_estimator
        Base estimator to be used in the nodes. This should be an scikit-learn compatible regressor.
    criterion : str
        Split Criterion. Supported values are: `"gradient"` and `"gradient-renorm-z"`
    max_depth : int (default = 3)
        Maximal depth of the tree
    min_samples_split : int (default = 10)
        Minimal number of samples that go to each split
    gradient_function : callable
        A function that computes the gradient of a model with respect to the model parameters at given points.
        The gradient_function gets 3 parameters: a fitted model (see base_estimator),
        the input matrix X and the target vector y.
    renorm_function : callable
        A function that allows to renormalize gradients of a model.
        The renorm_function gets 4 parameters: a fitted model (see base_estimator),
        the gradient matrix g, a scale factor and a shift vector.
        *The renormalization function must be set based on the `gradient_function`. Unsuitable functions might
        result in bad results, exceptions and unexpected outcomes.*
        **Note: Only set this parameter if you use a custom gradient function and know what you are doing.**
    dummy_classifier
        Base estimator that is used in nodes where all training samples belong to one class.

    Attributes
    ----------
    root_ : TreeNode
        Root Node of the tree structure
    classes_ : array, shape (n_classes, )
        A list of class labels known to the classifier.

    References
    ----------
    .. [1] Broelemann, K. and Kasneci, G.,
       "A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees",
       Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2019
    """

    def __init__(self,
                 base_estimator=LogisticRegression(solver="liblinear"),
                 criterion="gradient",
                 max_depth=3,
                 min_samples_split=10,
                 gradient_function=None,
                 renorm_function=None,
                 dummy_classifier=DummyClassifier(strategy="prior")):
        super().__init__(
            base_estimator=base_estimator,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            gradient_function=gradient_function,
            renorm_function=renorm_function)
        self.dummy_classifier = dummy_classifier

    def predict_proba(self, X):
        """
        Predict the probabilities for each class

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the samples

        Returns
        -------
        P: array, shape= [n_samples, n_classes]
            Probabilities of samples to belong to the classes.
        """

        def leaf_predict_proba(estimator, X_):
            # Predict probabilities
            p = estimator.predict_proba(X_)
            # Map classes
            p = _map_columns(
                output_columns=self.classes_,
                input_columns=estimator.classes_,
                X_input=p,
                default_value=0
            )

            return p

        return self._apply_sample_wise_function_on_leafs(leaf_predict_proba, X)

    def predict_log_proba(self, X):
        """
        Predict the probabilities for each class

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the samples

        Returns
        -------
        P: array, shape= [n_samples, n_classes]
            Probabilities of samples to belong to the classes.
        """

        def leaf_predict_log_proba(estimator, X_):
            # Predict probabilities
            p = estimator.predict_log_proba(X_)
            # Map classes
            p = _map_columns(
                output_columns=self.classes_,
                input_columns=estimator.classes_,
                X_input=p,
                default_value=-np.inf
            )

            return p

        return self._apply_sample_wise_function_on_leafs(leaf_predict_log_proba, X)

    def _create_and_fit_estimator(self, X, y):
        """
        Creates a new estimator for a node and trains it with the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input Features of the training data
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target variable.

        Returns
        -------
        estimator
            The trained estimator

        Notes
        -----
        This methods overloads the method of the parent class to allow for DummyClassifier instances in case of nodes
        that only see one class. This is necessary, because other classifiers, such as LogisticRegression require
        at least two classes.
        """
        if len(np.unique(y)) == 1 and self.dummy_classifier is not None:
            cls = clone(self.dummy_classifier)
            cls.fit(X, y)
            return cls
        else:
            return super()._create_and_fit_estimator(X, y)

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        super().fit(X, y)


def _map_columns(output_columns, input_columns, X_input, default_value=0):
    """
    Helper function that maps one 2-dimensional array to another by reordering columns and filling missing columns
    with default values.


    Parameters
    ----------
    output_columns: array-like, shape=(m1)
        Columns of the output array. These can me class names or just numbers.
    input_columns: array-like, shape=(m2)
        Columns of the input array. These can me class names or just numbers. Datatype should be the same as of
        `output_columns`. Typically (but not necessarily), `input_columns` is a subset of `output_columns`
    X_input: array-like, shape=(n, m2)
        The input data that should be mapped to the defined output columns
    default_value
        Value to fill into missing columns (i.e. columns that are not given in the input but expected in the output)

    Returns
    -------
    X: array, shape=(n,m1)
        The output matrix. This is done by padding the input data with default values in the missing columns.
    Notes
    -----
    The intention of this function is to handle model tree classifiers where subtrees to not contain samples of
    all classes (up to the case that a node only gets samples of one class). In this case, the output shape of
    `predict_proba` contains too few columns. In order to merge the results from multiple leafs, this function
    fills missing columns with default values.

    """
    # if output and input columns are the same, nothing has to be done:
    if np.array_equiv(output_columns, input_columns):
        return X_input

    # Identify columns:
    #       output columns at index o_idx corresponds to input columns at index i_idx
    idx = np.searchsorted(output_columns, input_columns)
    i_idx, = np.where(input_columns == output_columns[idx])
    o_idx = idx[i_idx]

    # Create result array
    n_samples = np.shape(X_input)[0]
    n_columns = len(output_columns)
    X = np.full((n_samples, n_columns), default_value, dtype=X_input.dtype)

    # Copy input data into the correct columns of X
    X[:, o_idx] = X_input[:, i_idx]

    return X
