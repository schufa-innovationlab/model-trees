"""
This module contains pre-defined methods to compute gradients for common estimators in
combination with common loss-functions. These gradients can be used to train model trees [1]_

All these gradients are computed with respect to the models parameters.

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

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression


def get_default_gradient_function(model):
    """
    Returns the default
    Parameters
    ----------
    model

    Returns
    -------

    """
    if type(model) not in _DEFAULT_GRADIENTS:
        raise ValueError(f"No default gradient defined for {type(model)}.")
    return _DEFAULT_GRADIENTS[type(model)]


def gradient_logistic_regression_cross_entropy(model, X, y):
    """
    Computes the gradients of a logistic regression model with cross validation loss

    Parameters
    ----------
    model : LogisticRegression
        The model of which the gradient shall be computed.
        The model should already be fitted to some data (typically to the data of the parent node)
    X : array-like, shape = [n_samples, n_features]
        Input Features of the points at which the gradient should be computed
    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Target variable. Corresponds to the samples in `X`

    Returns
    -------
    g: array-like, shape = [n_samples, n_parameters]
        Gradient of the cross entropy loss with respect to the model parameters at the samples given by `X` and `y`

    Notes
    -----
    * The number of model parameters is equal to the number of features (if the intercept is not trainable) or
      has one additional parameter (if the intercept is trainable)
    * See [1]_ for the math behind it

    References
    ----------
    .. [1] https://peterroelants.github.io/posts/cross-entropy-logistic/
    """
    if len(model.classes_) > 2:
        # TODO: multi-class case is not supported, yet
        raise ValueError(f"This method currently only supports binary classification problems, but we got {len(model.classes_)} classes.")

    # Compute Gradient (see also [1])
    factor = model.predict_proba(X)[:, 1:2] - np.reshape(y, (-1, 1))
    g = factor * X

    if model.fit_intercept:
        # Append artificial intercept gradient
        n_intercept = np.prod(np.shape(model.intercept_))
        n_samples = np.shape(X)[0]
        g = np.concatenate([g, factor * np.ones((n_samples, n_intercept))], axis=1)

    return g


def gradient_linear_regression_square_loss(model, X, y):
    """
    Computes the gradients of a logistic regression model with cross validation loss

    Parameters
    ----------
    model : LinearRegression
        The model of which the gradient shall be computed.
        The model should already be fitted to some data (typically to the data of the parent node)
    X : array-like, shape = [n_samples, n_features]
        Input Features of the points at which the gradient should be computed
    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Target variable. Corresponds to the samples in `X`

    Returns
    -------
    g: array-like, shape = [n_samples, n_parameters]
        Gradient of the square loss with respect to the model parameters at the samples given by `X` and `y`

    Notes
    -----
    * The number of model parameters is equal to the number of features (if the intercept is not trainable) or
      has one additional parameter (if the intercept is trainable)
    * See [1]_ for the math behind it

    """
    # Prediction
    y_ = model.predict(X)
    # Residuals
    r = y_ - y    # TODO: handle shapes (n) and (n,1) in y_and y
    if len(r.shape) == 1:
        r = np.reshape(r, (-1,1))

    n_out = r.shape[1]
    # Gradient by output
    g = [r[:,o:o+1] * X for o in range(n_out)]
    # Concatenate along parameter axis (axis = 1)
    g = np.concatenate(g, axis=1)

    if model.fit_intercept:
        # Append intercept gradient: The intercept gradient equals to the residuals
        g = np.concatenate([g, r], axis=1)

    return g


# Default gradients allow to use Model Trees without explicitly defining the gradient computation
_DEFAULT_GRADIENTS = {
    LinearRegression: gradient_linear_regression_square_loss,
    LogisticRegression: gradient_logistic_regression_cross_entropy
}
