"""
This module provides functions to use Pipelines as weak models in a modeltree
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


def pipeline_gradient(estimator_gradient):
    """
    Creates a gradient computation function for a pipeline estimator.
    For this method, a gradient function for the final step of the pipeline is used.
    Gradients for the previous steps are not computed.

    Parameters
    ----------
    estimator_gradient: callable
        A gradient function for the final step of the pipeline

    Returns
    -------
    gradient_fct: callable
        A gradient function for the whole pipeline.

    """
    # Gradient function wrapper
    def gradient_fct(pipe, X, y):
        # Transform x
        Xt = X
        for name, transformer in pipe.steps[:-1]:
            Xt = transformer.transform(Xt)

        # Compute gradient of final estimator
        estimator = pipe.steps[-1][1]
        return estimator_gradient(estimator, Xt, y)

    # Return pipeline gradients function
    return gradient_fct
