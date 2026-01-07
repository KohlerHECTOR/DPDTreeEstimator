Dynamic Programming Decision Trees
============================================================
### A new tree-based estimator.
```bash
pip install git+https://github.com/KohlerHECTOR/DPDTreeEstimator
```
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dpdt import DPDTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

CART = DecisionTreeClassifier(max_depth=3, random_state=42)
DPDT = DPDTreeClassifier(max_depth=3, random_state=42)
CART.fit(X_train, y_train)
DPDT.fit(X_train, y_train)

assert DPDT.score(X_train, y_train) >= CART.score(X_train, y_train), 'DPDT does not have better train accuract than CART'
print(f'CART test accuracy={CART.score(X_test, y_test)}')
print(f'DPDT test accuracy={DPDT.score(X_test, y_test)}')
```

### AdaBoostDPDT
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from dpdt import AdaBoostDPDT

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

adaboost = AdaBoostClassifier(random_state=42, n_estimators=50)
adaboost_dpdt = AdaBoostDPDT(random_state=42, n_estimators=50)
adaboost.fit(X_train, y_train)
adaboost_dpdt.fit(X_train, y_train)

print(f'AdaBoost test accuracy={adaboost.score(X_test, y_test)}')
print(f'AdaBoostDPDT test accuracy={adaboost_dpdt.score(X_test, y_test)}')
```


### More about the hyperparameters here
https://github.com/KohlerHECTOR/DPDTreeEstimator/issues/15
### Comparison of different classifiers.
![Classifier Comparison](examples/compare_classif.png)

### CITE
```
@inproceedings{DPDT-Kohler,
author = {Kohler, Hector and Akrour, Riad and Preux, Philippe},
title = {Breiman meets Bellman: Non-Greedy Decision Trees with MDPs},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3736868},
doi = {10.1145/3711896.3736868},
pages = {1207–1218},
numpages = {12},
keywords = {decision trees, markov decision processes},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```
