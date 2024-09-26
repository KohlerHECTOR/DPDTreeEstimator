"""Gradient Boosting Dynamic Programming Decision Tree (DPDTree)."""
from numbers import Integral, Real

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_array, check_is_fitted

from .dpdt_regressor import DPDTreeRegressor


class GradientBoostingDPDTClassifier(ClassifierMixin, BaseEstimator):
    """
    Gradient Boosting classifier using DPDTreeRegressor as base estimator.

    This classifier implements gradient boosting for multiclass classification
    problems using DPDTreeRegressor as the base learner. It fits a sequence of
    weak learners on the negative gradient of the loss function.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    max_depth : int, default=3
        The maximum depth of the individual regression estimators.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    n_jobs : int, default=None
        The number of jobs to run in parallel for fitting the trees.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    estimators_ : list of lists of DPDTreeRegressor
        The collection of fitted sub-estimators.
    initial_predictions_ : ndarray of shape (n_classes,)
        The initial predictions for each class.

    """

    _parameter_constraints = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="left")],
        "max_depth": [Interval(Integral, 2, None, closed="left")],
        "cart_nodes_list": ["array-like"],
        "random_state": [Interval(Integral, 0, None, closed="left")],
        "use_default_dt": ["boolean"],
        "n_jobs": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"best"}),
        ],
        "n_jobs_dpdt": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"best"}),
        ],
    }

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        cart_nodes_list=(16,),
        random_state=42,
        use_default_dt=False,
        n_jobs=None,
        n_jobs_dpdt=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.cart_nodes_list = cart_nodes_list
        self.random_state = random_state
        self.use_default_dt = use_default_dt
        self.n_jobs = n_jobs
        self.n_jobs_dpdt = n_jobs_dpdt

    def _fit_tree(self, X, gradients, class_idx):
        if self.use_default_dt:
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            tree = DPDTreeRegressor(
                max_depth=self.max_depth,
                cart_nodes_list=self.cart_nodes_list,
                random_state=self.random_state,
                max_nb_trees=1,
                n_jobs=self.n_jobs_dpdt,
            )
        tree.fit(X, -gradients[:, class_idx])  # Note the negative sign
        return tree

    def fit(self, X, y):
        """
        Build a gradient boosting classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_classes_ = len(self.classes_)

        # Convert y to one-hot encoding
        y_one_hot = np.eye(self.n_classes_)[np.searchsorted(self.classes_, y)]

        self.estimators_ = []
        self.initial_predictions_ = np.log(np.mean(y_one_hot, axis=0))
        if self.n_jobs == "best":
            n_jobs = self.n_classes_
        else:
            n_jobs = self.n_jobs

        for _ in range(self.n_estimators):
            probas = self.predict_proba(X)
            gradients = probas - y_one_hot  # This is the correct gradient for log loss

            trees = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._fit_tree)(X, gradients, class_idx)
                for class_idx in range(self.n_classes_)
            )

            self.estimators_.append(trees)

        return self

    def decision_function(self, X):
        """
        Compute the decision function of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        score : ndarray of shape (n_samples, n_classes)
            The decision function of the input samples for each class.
        """
        check_is_fitted(self)
        X = check_array(X)

        decision = np.tile(self.initial_predictions_, (X.shape[0], 1))
        for trees in self.estimators_:
            for class_idx, tree in enumerate(trees):
                decision[:, class_idx] += self.learning_rate * tree.predict(X)

        return decision

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        decision = self.decision_function(X)
        return softmax(decision, axis=1)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def staged_predict(self, X):
        """
        Predict class labels at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        y : ndarray of shape (n_samples,)
            The predicted class labels at each stage.
        """
        check_is_fitted(self)
        X = check_array(X)

        decision = np.tile(self.initial_predictions_, (X.shape[0], 1))
        # yield self.classes_[np.argmax(softmax(decision, axis=1), axis=1)]

        for trees in self.estimators_:
            for class_idx, tree in enumerate(trees):
                decision[:, class_idx] += self.learning_rate * tree.predict(X)
            yield self.classes_[np.argmax(softmax(decision, axis=1), axis=1)]
