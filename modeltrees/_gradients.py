"""
This module contains pre-defined methods to compute gradients for common estimators in
combination with common loss-functions. These gradients can be used to train model trees [1]_
All these gradients are computed with respect to the models parameters.

In addition, the module also provides pre-defined methods to renormalize gradients of common estimators. Details
can be also found in [1]_.

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
    Returns the default gradient computation method for well-know models and default loss functions

    Parameters
    ----------
    model
        A predictive model

    Returns
    -------
    gradient_function: callable
        A function that computes gradients of the loss for the given type of model.

    See Also
    --------
    gradient_logistic_regression_cross_entropy, gradient_linear_regression_square_loss

    """
    if type(model) not in _DEFAULT_GRADIENTS:
        raise ValueError(f"No default gradient defined for {type(model)}.")
    return _DEFAULT_GRADIENTS[type(model)]


def get_default_renormalization_function(model):
    """
    Returns the default renormalization function for gradients of well-know models and default loss functions

    Parameters
    ----------
    model
        A predictive model

    Returns
    -------
    gradient_function: callable
        A function that computes gradients of the loss for the given type of model.

    See Also
    --------
    renormalize_linear_model_gradients
        Example function
    get_get_default_gradient_function
        Default gradient computation
    """
    if type(model) not in _DEFAULT_RENORMALIZATION:
        raise ValueError(f"No default renormalization defined for {type(model)}.")
    return _DEFAULT_RENORMALIZATION[type(model)]


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
    * See [2]_ for the math behind it

    References
    ----------
    .. [2] https://peterroelants.github.io/posts/cross-entropy-logistic/
    """
    if len(model.classes_) > 2:
        # TODO: multi-class case is not supported, yet
        raise ValueError(
            f"This method currently only supports binary classification problems, but we got {len(model.classes_)} classes.")

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

    """
    # Prediction
    y_ = model.predict(X)
    # Residuals
    r = y_ - y  # TODO: handle shapes (n) and (n,1) in y_and y
    if len(r.shape) == 1:
        r = np.reshape(r, (-1, 1))

    n_out = r.shape[1]
    # Gradient by output
    g = [r[:, o:o + 1] * X for o in range(n_out)]
    # Concatenate along parameter axis (axis = 1)
    g = np.concatenate(g, axis=1)

    if model.fit_intercept:
        # Append intercept gradient: The intercept gradient equals to the residuals
        g = np.concatenate([g, r], axis=1)

    return g


def renormalize_linear_model_gradients(model, gradients, a, c):
    """
    Renormalizes gradients of a linear model.

    This function applies to the linear case where a vector x is linearly normalized by `a * x + c`.

    Parameters
    ----------
    model: LinearRegression or LogisticRegression
        The model that generated the gradients
    gradients: array, shape=[n_samples, n_params]
        A matrix of gradients where each row corresponds to one gradient
    a: array, shape=[n_samples, n_features]
        The normalization factor
    c: array, shape=[n_samples, n_features]
        The normalization offset

    Returns
    -------
    gradients: array, shape=[n_samples, n_params]
        Renormalized gradients

    Warnings
    --------
    Note that this method modifies gradients inplace.

    """
    # Shape of the coefficients
    c_shape = np.shape(model.coef_)

    # Number of input features
    m = len(a)

    if len(c_shape) == 2:
        # 2-dim coefficients
        #   --> multiple outputs
        d = c_shape[0]  # Dimension of the output

        # Multi elements of the gradient need to be normalized by the same factor
        #   --> Repeat a
        a = np.repeat(a, d, axis=1)

    # Compute number of parameter to modify
    n = np.shape(a)[1]

    # Modify the gradients according to eq. (14) of [1]_
    #       here, A is a diagonal matrix with diagonal elements `a`
    # Note: in case of multi-dimensional outputs, c must be
    c = [c * gradients[:, i:i + 1] for i in range(n, np.shape(gradients)[1])]
    c = np.concatenate(c, axis=1)

    gradients[:, :n] = gradients[:, :n] * a + c

    return gradients


# Default gradients allow to use Model Trees without explicitly defining the gradient computation
_DEFAULT_GRADIENTS = {
    LinearRegression: gradient_linear_regression_square_loss,
    LogisticRegression: gradient_logistic_regression_cross_entropy
}

# Default renormalization functions allow to use Model Trees without explicitly defining the renormalization methods
_DEFAULT_RENORMALIZATION = {
    LinearRegression: renormalize_linear_model_gradients,
    LogisticRegression: renormalize_linear_model_gradients
}
