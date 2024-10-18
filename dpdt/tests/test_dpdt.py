import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dpdt import (
    DPDTreeClassifier,
    DPDTreeRegressor,
    GradientBoostingDPDTClassifier,
)


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
    [10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 50],
)
@pytest.mark.parametrize(
    "centers",
    [2, 6],
)
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize(
    "cart_nodes_list",
    [
        (3,),
        (
            3,
            3,
        ),
    ],
)
@pytest.mark.parametrize("n_estimators", [3, 4])
@pytest.mark.parametrize("n_jobs_dpdt", [None, 4])
@pytest.mark.parametrize("xgboost", [False, True])
@pytest.mark.parametrize("reg_lambda", [0, 0.1])
def test_better_cart_gb(
    n_samples,
    n_features,
    centers,
    max_depth,
    cart_nodes_list,
    n_estimators,
    n_jobs_dpdt,
    xgboost,
    reg_lambda,
):
    X, y = make_blobs(n_samples, centers=centers, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf = GradientBoostingDPDTClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        cart_nodes_list=cart_nodes_list,
        n_jobs_dpdt=n_jobs_dpdt,
        xgboost=xgboost,
        reg_lambda=reg_lambda,
    )
    clf.fit(X, y)
    cart = GradientBoostingDPDTClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        use_default_dt=True,
        xgboost=xgboost,
        reg_lambda=reg_lambda,
    )
    cart.fit(X, y)
    assert clf.score(X, y) >= cart.score(X, y)


@pytest.mark.parametrize(
    "n_samples",
    [10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 50],
)
@pytest.mark.parametrize(
    "centers",
    [2, 6],
)
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize(
    "cart_nodes_list",
    [
        (3,),
        (
            3,
            3,
        ),
    ],
)
@pytest.mark.parametrize("n_jobs", [None, 4])
@pytest.mark.parametrize("sw", [True, False])
def test_better_cart(
    n_samples, n_features, centers, max_depth, cart_nodes_list, n_jobs, sw
):
    X, y = make_blobs(n_samples, centers=centers, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifier(max_depth, cart_nodes_list=cart_nodes_list, n_jobs=n_jobs)
    if sw:
        sample_weight = np.random.default_rng(42).random(len(X))
    else:
        sample_weight = None
    clf.fit(X, y, sample_weight)
    cart = DecisionTreeClassifier(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y, sample_weight)
    assert clf.score(X, y) >= cart.score(X, y)


@pytest.mark.parametrize(
    "n_samples",
    [10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 50],
)
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("max_nb_trees", [1, 10])
@pytest.mark.parametrize("n_jobs", [None, 4])
@pytest.mark.parametrize(
    "cart_nodes_list",
    [
        (3,),
        (12,),
        (
            3,
            3,
        ),
    ],
)
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
    [5, 20],
)
@pytest.mark.parametrize(
    "centers",
    [2, 6],
)
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("cart_nodes_list", [(3,)])
@pytest.mark.parametrize("n_jobs", [None, 4])
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
    assert np.allclose(dpdt_score, cart_score, rtol=1e-2) or dpdt_score >= cart_score


@pytest.mark.parametrize(
    "n_samples",
    [10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 50],
)
@pytest.mark.parametrize(
    "centers",
    [2, 6],
)
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 3)])
@pytest.mark.parametrize("n_jobs", [None, 4])
@pytest.mark.parametrize("sw", [True, False])
def test_better_cart_regress(
    n_samples, n_features, centers, max_depth, cart_nodes_list, n_jobs, sw
):
    X = np.random.random(size=(n_samples, n_features))
    y = np.array([sum([x[0] ** i for i in range(centers)]) for x in X])
    y = y.reshape(-1, 1)
    if sw:
        sample_weight = np.random.default_rng(42).random(len(X))
    else:
        sample_weight = None
    clf = DPDTreeRegressor(max_depth, cart_nodes_list=cart_nodes_list, n_jobs=n_jobs)
    clf.fit(X, y, sample_weight)
    cart = DecisionTreeRegressor(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y, sample_weight)
    dpdt_score = clf.score(X, y)
    cart_score = cart.score(X, y)
    assert np.allclose(dpdt_score, cart_score, rtol=1e-3) or dpdt_score >= cart_score
