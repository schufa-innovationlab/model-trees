# Model Trees
This package defines model trees as scikit-learn compatible estimators.
These can be used for both classification and regression.

The model trees use the gradient-based split criterion from [[1]](#References)

## Usage
The package provides two estimator classes:
`ModelTreeRegressor` for regression and `ModelTreeClassifier` for classification.
Both provide default settings for the base estimators used in the leafs and can directly be used.

### Introductory Example
```python
from modeltrees import ModelTreeRegressor
import numpy as np

X = np.random.randn(10000, 3)
y = np.matmul(X, [[-1], [2], [1.5]]) + np.random.randn(10000, 1) * 0.2

mt = ModelTreeRegressor()
mt.fit(X,y)

mt.predict([[1, 2, 3]])
```

## References
[1] Broelemann, K. and Kasneci, G.;
A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees;
International Joint Conference on Artificial Intelligence (IJCAI) 2019; [pdf](https://arxiv.org/abs/1809.09703)
<details><summary>Bibtex</summary>
<p>

```
@inproceedings{Broelemann2019modeltrees,
    author = {Klaus Broelemann and Gjergji Kasneci},
    title  = {A Gradient-Based Split Criterion for Highly
              Accurate and Transparent Model Trees},
    booktitle = {Proceedings of the 28th International Joint
              Conference on Artificial Intelligence, {IJCAI} 2019},
    year = 2019
}
```

</p>
</details>