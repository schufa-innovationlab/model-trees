"""
This package defines model trees scikit-learn compatible estimators.
These can be used for both classification and regression.

The model trees use the gradient-based split criterion from [1]_

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

from ._trees import ModelTreeRegressor, ModelTreeClassifier

__all__ = ["ModelTreeRegressor", "ModelTreeClassifier"]