# Model Trees
This package defines model trees as scikit-learn compatible estimators.
These can be used for both classification and regression.

The model trees use the gradient-based split criterion from [[1]](#References)

## Installation
The modeltrees package can directly be installed from GitHub using pip:
```shell scrip
pip install https://github.com/schufa-innovationlab/model-trees/archive/master.zip
```

Now you are ready to start, e.g. with our [introductory example](#introductory-example)

### Development Version
The modeltrees package is still under development. You can install the current development version via
```shell scrip
pip install https://github.com/schufa-innovationlab/model-trees/archive/dev.zip
```

### Alternative ways
There might be reasons not to use pip with GitHub, e.g. because some
proxy does not allows to connect to GitHub, or pip is just not
your tool of choice (keep in mind: you can also use pip in a conda setup). 

In that case you could:
1. Clone the repository locally and use the local path instead of aboves 
GitHub path.
2. You can just copy the `modeltrees` folder into your projects source root.
In that case, you have to install the [dependencies](requirements.txt) manually.


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

p = mt.predict([[1, 2, 3]])
print(f"Prediction: {p}")
```

### Further examples
There is an `examples` folder that contains further examples and evaluations.
See the [corresponding readme](examples/README.md) for further details and a list of examples.

## References
[1] Broelemann, K. and Kasneci, G.;
A Gradient-Based Split Criterion for Highly Accurate and Transparent Model Trees;
International Joint Conference on Artificial Intelligence (IJCAI) 2019; [pdf](https://www.ijcai.org/proceedings/2019/0281.pdf)
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