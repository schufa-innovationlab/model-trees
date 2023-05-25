"""
Predefined default gradients for some models.
In most cases, these are sufficient to use.
If nothing else is specified, :func:`modeltrees.criteria.GradientSplitCriterion` uses these default gradients.
"""

#  Copyright 2023 SCHUFA Holding AG
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

from sklearn.linear_model import LinearRegression, LogisticRegression

from .linear import \
    gradient_linear_regression_square_loss, \
    gradient_logistic_regression_cross_entropy, \
    renormalize_linear_model_gradients


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
