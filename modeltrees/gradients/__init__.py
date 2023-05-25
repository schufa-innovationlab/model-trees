"""
This package contains pre-defined methods to compute gradients for common estimators in
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

from .default import get_default_gradient_function, get_default_renormalization_function
