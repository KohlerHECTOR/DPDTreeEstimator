import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dpdt import DPDTreeClassifier, DPDTreeRegressor


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_dpdt_classifier(data):
    """Check the internals and behaviour of `DPDTreeClassifier`."""
    X, y = data
    clf = DPDTreeClassifier()
    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


@pytest.mark.parametrize(
    "n_samples",
    [10, 1000],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 500],
)
@pytest.mark.parametrize(
    "centers",
    [2, 4, 6],
)
@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (128,), (3, 5, 4, 1), (6, 6)])
def test_better_cart(n_samples, n_features, centers, max_depth, cart_nodes_list):
    X, y = make_blobs(n_samples, centers=centers, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifier(max_depth, cart_nodes_list=cart_nodes_list)
    clf.fit(X, y)
    cart = DecisionTreeClassifier(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y)
    assert clf.score(X, y) >= cart.score(X, y)


@pytest.mark.parametrize(
    "n_samples",
    [10, 1000],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 500],
)
@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize("max_nb_trees", [1, 100])
@pytest.mark.parametrize("n_jobs", [None, 4, "best"])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (128,), (3, 5, 4, 1), (6, 6)])
def test_dpdt_learning(
    n_samples, n_features, max_depth, max_nb_trees, cart_nodes_list, n_jobs
):
    X, y = make_blobs(n_samples, centers=2, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifier(max_depth, max_nb_trees, cart_nodes_list, n_jobs=n_jobs)
    clf.fit(X, y)
    clf.get_pareto_front(X, y)
    clf.predict(X)
    assert clf.score(X, y) >= 0.48


@pytest.mark.parametrize(
    "n_samples",
    [10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 200],
)
@pytest.mark.parametrize(
    "centers",
    [2, 4, 6],
)
@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize("cart_nodes_list", [(3,)])
@pytest.mark.parametrize("n_jobs", [None, 3, "best"])
def test_better_cart_multiout(
    n_samples, n_features, centers, max_depth, cart_nodes_list, n_jobs
):
    X = np.random.random(size=(n_samples, n_features))
    y = [[x[0] ** i for i in range(centers)] for x in X]
    clf = DPDTreeRegressor(max_depth, cart_nodes_list=cart_nodes_list, n_jobs=n_jobs)
    clf.fit(X, y)
    cart = DecisionTreeRegressor(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y)
    dpdt_score = clf.score(X, y)
    cart_score = cart.score(X, y)
    assert np.allclose(dpdt_score, cart_score, rtol=1e-1) or dpdt_score >= cart_score


@pytest.mark.parametrize(
    "n_samples",
    [10, 1000],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 500],
)
@pytest.mark.parametrize(
    "centers",
    [2, 4, 6],
)
@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
@pytest.mark.parametrize("n_jobs", [None, 3, "best"])
def test_better_cart_reg(
    n_samples, n_features, centers, max_depth, cart_nodes_list, n_jobs
):
    X = np.random.random(size=(n_samples, n_features))
    y = np.array([sum([x[0] ** i for i in range(centers)]) for x in X])
    y = y.reshape(-1, 1)
    clf = DPDTreeRegressor(max_depth, cart_nodes_list=cart_nodes_list, n_jobs=n_jobs)
    clf.fit(X, y)
    cart = DecisionTreeRegressor(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y)
    dpdt_score = clf.score(X, y)
    cart_score = cart.score(X, y)
    assert np.allclose(dpdt_score, cart_score, rtol=1e-3) or dpdt_score >= cart_score
