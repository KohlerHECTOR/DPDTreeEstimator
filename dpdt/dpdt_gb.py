"""Gradient Boosting Dynamic Programming Decision Tree (DPDTree)."""
from numbers import Integral, Real

import numpy as np
from sklearn._loss.loss import (
    HalfBinomialLoss,
    HalfMultinomialLoss,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from .dpdt_regressor import DPDTreeRegressor


def predict_stage(estimator_at_i, X, learning_rate, raw_predictions):
    for k, tree in enumerate(estimator_at_i):
        raw_predictions[:, k] += learning_rate * tree.predict(X)
    return raw_predictions


def predict_stages(estimators, X, learning_rate, raw_predictions):
    for i in range(estimators.shape[0]):
        raw_predictions = predict_stage(
            estimators[i], X, learning_rate, raw_predictions
        )
    return raw_predictions


def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    """Return the initial raw predictions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data array.
    estimator : object
        The estimator to use to compute the predictions.
    loss : BaseLoss
        An instance of a loss function class.
    use_predict_proba : bool
        Whether estimator.predict_proba is used instead of estimator.predict.

    Returns
    -------
    raw_predictions : ndarray of shape (n_samples, K)
        The initial raw predictions. K is equal to 1 for binary
        classification and regression, and equal to the number of classes
        for multiclass classification. ``raw_predictions`` is casted
        into float64.
    """
    # TODO: Use loss.fit_intercept_only where appropriate instead of
    # DummyRegressor which is the default given by the `init` parameter,
    # see also _init_state.
    if use_predict_proba:
        # Our parameter validation, set via _fit_context and _parameter_constraints
        # already guarantees that estimator has a predict_proba method.
        predictions = estimator.predict_proba(X)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]  # probability of positive class
        eps = np.finfo(np.float32).eps  # FIXME: This is quite large!
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)


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
        cart_nodes_list=(32,),
        random_state=42,
        use_default_dt=False,
        n_jobs_dpdt=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.cart_nodes_list = cart_nodes_list
        self.random_state = random_state
        self.use_default_dt = use_default_dt
        self.n_jobs_dpdt = n_jobs_dpdt

    def _is_fitted(self):
        return len(getattr(self, "estimators_", [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

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
        X, y = self._validate_data(X, y, dtype=DTYPE, multi_output=True)
        y = self._encode_y(y)
        self.estimators_ = np.empty(
            (self.n_estimators, self.n_trees_per_iteration_), dtype=object
        )
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        y = column_or_1d(y, warn=True)  # TODO: Is this still required?
        X_train, y_train = X, y
        # self.loss is guaranteed to be a string
        self._loss = self._get_loss()

        self.init_ = DummyClassifier(strategy="prior")  # Model 0
        self.init_.fit(X_train, y_train)
        begin_at_stage = 0
        raw_predictions = _init_raw_predictions(X_train, self.init_, self._loss, True)

        self.n_estimators_ = self._fit_stages(
            X_train,
            y_train,
            raw_predictions,
            begin_at_stage,
        )
        return self

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
    ):
        """Fit another stage of ``n_trees_per_iteration_`` trees."""
        original_y = y
        # Note: We need the negative gradient!
        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
        )
        # 2-d views of shape (n_samples, n_trees_per_iteration_) or (n_samples, 1)
        # on neg_gradient to simplify the loop over n_trees_per_iteration_.
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                y = np.array(original_y == k, dtype=np.float64)
            # induce regression tree on the negative gradient
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

            tree.fit(X, neg_g_view[:, k])

            # add tree to ensemble
            self.estimators_[i, k] = tree

        # Update raw_predictions with the new tree's predictions
        raw_predictions = predict_stage(
            self.estimators_[i], X, self.learning_rate, raw_predictions
        )
        return raw_predictions

    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        begin_at_stage=0,
    ):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        if isinstance(
            self._loss,
            (HalfBinomialLoss,),
        ):
            factor = 2
        else:
            factor = 1

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):
            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i,
                X,
                y,
                raw_predictions,
            )

            self.train_score_[i] = factor * self._loss(
                y_true=y,
                raw_prediction=raw_predictions,
            )

        return i + 1

    def _encode_y(self, y):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y_int.astype(float, copy=False)

        # From here on, it is additional to the HGBT case.
        # expose n_classes_ attribute
        self.n_classes_ = n_classes
        n_trim_classes = n_classes

        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        return encoded_y

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = check_array(X)
        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64
            )
        else:
            raw_predictions = _init_raw_predictions(X, self.init_, self._loss, True)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        return predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)

    def _staged_raw_predict(self, X, check_input=True):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        check_input : bool, default=True
            If False, the input arrays X will not be checked.

        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, order="C", reset=False)
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            raw_predictions = predict_stage(
                self.estimators_[i], X, self.learning_rate, raw_predictions
            )
            yield raw_predictions.copy()

    def _get_loss(self):
        if self.n_classes_ == 2:
            return HalfBinomialLoss()
        else:
            return HalfMultinomialLoss(n_classes=self.n_classes_)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        score : ndarray of shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            :term:`classes_`. Regression and binary classification produce an
            array of shape (n_samples,).
        """
        X = self._validate_data(X, dtype=DTYPE, order="C", reset=False)
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    # def staged_decision_function(self, X):
    #     """Compute decision function of ``X`` for each iteration.

    #     This method allows monitoring (i.e. determine error on testing set)
    #     after each stage.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         The input samples. Internally, it will be converted to
    #         ``dtype=np.float32``.

    #     Yields
    #     ------
    #     score : generator of ndarray of shape (n_samples, k)
    #         The decision function of the input samples, which corresponds to
    #         the raw values predicted from the trees of the ensemble . The
    #         classes corresponds to that in the attribute :term:`classes_`.
    #         Regression and binary classification are special cases with
    #         ``k == 1``, otherwise ``k==n_classes``.
    #     """
    #     yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        raw_predictions = self.decision_function(X)
        if raw_predictions.ndim == 1:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
        return self.classes_[encoded_classes]

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        if self.n_classes_ == 2:  # n_trees_per_iteration_ = 1
            for raw_predictions in self._staged_raw_predict(X):
                encoded_classes = (raw_predictions.squeeze() >= 0).astype(int)
                yield self.classes_.take(encoded_classes, axis=0)
        else:
            for raw_predictions in self._staged_raw_predict(X):
                encoded_classes = np.argmax(raw_predictions, axis=1)
                yield self.classes_.take(encoded_classes, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        raw_predictions = self.decision_function(X)
        return self._loss.predict_proba(raw_predictions)

    # def predict_log_proba(self, X):
    #     """Predict class log-probabilities for X.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         The input samples. Internally, it will be converted to
    #         ``dtype=np.float32``.

    #     Returns
    #     -------
    #     p : ndarray of shape (n_samples, n_classes)
    #         The class log-probabilities of the input samples. The order of the
    #         classes corresponds to that in the attribute :term:`classes_`.

    #     Raises
    #     ------
    #     AttributeError
    #         If the ``loss`` does not support probabilities.
    #     """
    #     proba = self.predict_proba(X)
    #     return np.log(proba)

    # def staged_predict_proba(self, X):
    #     """Predict class probabilities at each stage for X.

    #     This method allows monitoring (i.e. determine error on testing set)
    #     after each stage.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         The input samples. Internally, it will be converted to
    #         ``dtype=np.float32``.

    #     Yields
    #     ------
    #     y : generator of ndarray of shape (n_samples,)
    #         The predicted value of the input samples.
    #     """
    #     try:
    #         for raw_predictions in self._staged_raw_predict(X):
    #             yield self._loss.predict_proba(raw_predictions)
    #     except NotFittedError:
    #         raise
    #     except AttributeError as e:
    #         raise AttributeError(
    #             "loss=%r does not support predict_proba" % self.loss
    #         ) from e
