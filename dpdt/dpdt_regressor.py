"""Dynamic Programming Decision Tree (DPDTree) regressor implementation.

The heavy lifting (MDP construction and the dynamic program) is implemented in
C++ (see ``dpdt/_core.cpp``); this module is a thin scikit-learn compatible
wrapper around it. The public API (``DPDTreeRegressor().fit(X, y)`` /
``predict`` / ``get_pareto_front``) is unchanged.
"""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context
from sklearn.metrics import r2_score
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
    validate_data,
)

try:
    from . import _core
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The compiled extension dpdt._core is missing. Build it with "
        "`pip install -e .` (requires a C++17 compiler and pybind11)."
    ) from e


class DPDTreeRegressor(RegressorMixin, MultiOutputMixin, BaseEstimator):
    """Dynamic Programming Decision Tree (DPDTree) regressor.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_nb_trees : int, default=100
        The maximum number of trees.
    random_state : int, default=42
        Fixes randomness of the regressor. Randomness happens in the calls to cart.
    cart_nodes_list : list of int, default=(32,)
        List containing the number of leaf nodes for the CART trees at each depth.
    n_jobs : int, default=None
        The number of jobs to run in parallel.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at
          each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. note::

            The search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    trees_ : list of dict of ndarray
        One flat tree per zeta value. Each tree is a dict with keys
        ``feature`` (-1 for leaves), ``threshold``, ``children_left``,
        ``children_right`` and ``value`` (per-output leaf means).
    zetas_ : ndarray
        Array of zeta values used in the computation.

    Examples
    --------
    >>> from dpdt import DPDTreeRegressor
    >>> clf = DPDTreeRegressor()
    >>> clf.fit([[0, 0], [1, 1]], [[0], [1]])
    DPDTreeRegressor()
    >>> clf.predict([[2., 2.]])
    array([[1.]])
    >>> clf.fit([[0, 0], [1, 1]], [[0, 0], [1, 1]])
    DPDTreeRegressor()
    >>> clf.predict([[0., 0], [2., 2.]])
    array([[0., 0.],
           [1., 1.]])
    """

    _parameter_constraints = {
        "max_nb_trees": [Interval(Integral, 1, None, closed="left")],
        "cart_nodes_list": ["array-like"],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0.0, 1.0, closed="neither"),
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "random_state": ["random_state"],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
        "n_jobs": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"best"}),
        ],
    }

    def __init__(
        self,
        max_depth=10,
        max_nb_trees=1,
        cart_nodes_list=(
            8,
            3,
        ),
        random_state=None,
        n_jobs=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        min_impurity_decrease=0.0,
    ):
        """Initialize the DPDTreeRegressor."""
        self.max_depth = max_depth
        self.max_nb_trees = max_nb_trees
        self.cart_nodes_list = cart_nodes_list
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the DPDTreeRegressor to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or
            negative weight are ignored while searching for a split in each node.
            Splits are also ignored if they would result in any single
            class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            if y.squeeze().ndim > 1:
                raise AssertionError(
                    "sample weights are not yet supported for multioutput predictions"
                )
        else:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)

        self.X_ = X
        self.y_ = y.astype(float)

        # The core always works with 2D targets of shape (n_samples, n_outputs).
        y2d = np.ascontiguousarray(self.y_.reshape(len(self.y_), -1))

        n_samples, n_features = X.shape

        if self.max_nb_trees == 1:
            self.zetas_ = np.zeros(1, dtype=np.float64)
        else:
            self.zetas_ = np.linspace(-1, 0, self.max_nb_trees, dtype=np.float64)

        # --- resolve sklearn-style hyper-parameters for the C++ core --------
        # max_depth=None means "grow until pure"; a chain of binary splits can
        # never be deeper than n_samples - 1, which the C++ core uses as the
        # upper bound to pre-size its per-depth scratch buffers. [BOUND-1]
        max_depth = (
            self.max_depth if self.max_depth is not None else max(1, n_samples - 1)
        )

        if self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(n_features)))
            else:  # log2
                max_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, Integral):
            max_features = int(self.max_features)
        else:  # fraction
            max_features = max(1, int(self.max_features * n_features))

        # Fractional min_samples_* are resolved *per state* inside the core
        # (each inner CART is fit on the state's samples, mirroring the
        # previous implementation which called DecisionTreeRegressor.fit on
        # X[state]). We therefore forward the raw value plus a fraction flag.
        mss_is_fraction = not isinstance(self.min_samples_split, Integral)
        msl_is_fraction = not isinstance(self.min_samples_leaf, Integral)

        seed = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)

        if self.n_jobs is None:
            n_jobs = 1
        elif self.n_jobs == "best":
            n_jobs = 0  # let the core use min(#root candidates, hw threads)
        else:
            n_jobs = int(self.n_jobs)

        # The C++ core builds the MDP, runs the branch-and-bound dynamic
        # program and returns one already-extracted flat tree per zeta.
        self.trees_ = _core.fit_regressor(
            np.ascontiguousarray(X, dtype=np.float64),
            y2d,
            sample_weight,
            self.zetas_,
            int(max_depth),
            [int(c) for c in self.cart_nodes_list],
            float(self.min_samples_split),
            bool(mss_is_fraction),
            float(self.min_samples_leaf),
            bool(msl_is_fraction),
            float(self.min_weight_fraction_leaf),
            int(max_features),
            float(self.min_impurity_decrease),
            int(seed),
            n_jobs,
        )
        return self

    def _predict_zeta(self, X, zeta_index):
        """Route X through the tree of one zeta; return predictions and stats."""
        t = self.trees_[zeta_index]
        y_pred, lengths, n_nodes = _core.predict_tree_reg(
            np.ascontiguousarray(X, dtype=np.float64),
            t["feature"],
            t["threshold"],
            t["children_left"],
            t["children_right"],
            t["value"],
        )
        # Follow the shape of the fit-time targets (1D or 2D).
        if self.y_.ndim == 1:
            y_pred = y_pred.ravel()
        return (
            y_pred,
            float(lengths.mean()),
            int(lengths.max()),
            int(n_nodes),
        )

    def predict(self, X):
        """Predict regression targets for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, y="no_validation", reset=False)
        return self._predict_zeta(X, -1)[0]

    def get_pareto_front(self, X, y):
        """Compute the decision path lengths / test accuracy Pareto front of DPDTrees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        scores : array-like of shape (n_samples)
            The test accuracies of the trees.
        decision_path_length : array-like of shape (n_samples)
            The average number of decision nodes traversal in each tree.
        """
        check_is_fitted(self)
        n_zetas = len(self.zetas_)
        scores = np.zeros(n_zetas, dtype=np.float32)
        decision_path_length = np.zeros(n_zetas, dtype=np.float32)
        decision_path_max_depth = np.zeros(n_zetas, dtype=np.int32)
        decision_path_nodes = np.zeros(n_zetas, dtype=np.int32)

        # Tree traversal is in C++ now; joblib fan-out would only add overhead.
        for z in range(n_zetas):
            y_pred, mean_len, max_len, n_nodes = self._predict_zeta(X, z)
            scores[z] = r2_score(y, y_pred)
            decision_path_length[z] = mean_len
            decision_path_max_depth[z] = max_len
            decision_path_nodes[z] = n_nodes

        return (
            scores,
            decision_path_length,
            decision_path_max_depth,
            decision_path_nodes,
        )
