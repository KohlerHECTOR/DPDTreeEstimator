"""Dynamic Programming Decision Tree (DPDTree) classifier implementation."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class State:
    """Represent a state in the Markov Decision Process (MDP).

    Parameters
    ----------
    label : array-like
        The observation label for the state.
    nz : array-like of bool
        Boolean array indicating which samples are present in the state.
    is_terminal : bool, default=False
        Indicates if the state is a terminal state.
    """

    __slots__ = ["obs", "actions", "qs", "is_terminal", "nz"]

    def __init__(self, label, nz, is_terminal=False):
        """Initialize the State object."""
        self.obs = label
        self.actions = []
        self.qs = []
        self.is_terminal = is_terminal
        self.nz = nz

    def add_action(self, action):
        """Add an action to the state.

        Parameters
        ----------
        action : Action
            The action to be added to the state.
        """
        self.actions.append(action)


class Action:
    """Represent an action in the Markov Decision Process (MDP).

    Parameters
    ----------
    action : object
        The action representation (e.g., a split decision).
    """

    def __init__(self, action):
        """Initialize the Action object."""
        self.action = action
        self.rewards = []
        self.probas = []
        self.next_states = []

    def transition(self, reward, proba, next_s):
        """Add a transition for the action.

        Parameters
        ----------
        reward : float
            The reward associated with the transition.
        proba : float
            The probability of the transition.
        next_s : State
            The next state resulting from the transition.
        """
        self.rewards.append(reward)
        self.probas.append(proba)
        self.next_states.append(next_s)


class DPDTreeClassifier(ClassifierMixin, BaseEstimator):
    """Dynamic Programming Decision Tree (DPDTree) classifier.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_nb_trees : int, default=1000
        The maximum number of trees.
    random_state : int, default=42
        Fixes randomness of the classifier. Randomness happens in the calls to cart.
    cart_nodes_list : list of int, default=(3,)
        List containing the number of leaf nodes for the CART trees at each depth.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    mdp : list of list of State
        The Markov Decision Process represented as a list of lists of states,
        where each inner list contains the states at a specific depth.
    zetas : array-like
        Array of zeta values to be used in the computation.
    trees : dict
        A dictionary representing the tree policies. The keys are tuples representing
        the state observation and depth, and the values are the optimal tree
        for each zeta value.
    init_o : array-like
        The initial observation of the MDP.

    Examples
    --------
    >>> from dpdt import DPDTreeClassifier
    >>> from sklearn import datasets
    >>> X, y = datasets.load_breast_cancer(return_X_y=True)
    >>> clf = DPDTreeClassifier(max_depth=3, random_state=42)
    >>> clf.fit(X, y)
    >>> print(clf.score(X, y))
    """

    _parameter_constraints = {
        "max_depth": [Interval(Integral, 2, None, closed="left")],
        "max_nb_trees": [Interval(Integral, 1, None, closed="left")],
        "cart_nodes_list": ["array-like"],
        "random_state": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self, max_depth=3, max_nb_trees=1000, cart_nodes_list=(3,), random_state=42
    ):
        """Initialize the DPDTreeClassifier."""
        self.max_depth = max_depth
        self.max_nb_trees = max_nb_trees
        self.cart_nodes_list = cart_nodes_list
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the DPDTreeClassifier to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

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

        if self.max_nb_trees == 1:
            self._zetas = np.zeros(1, dtype=np.float32)
        else:
            self._zetas = np.linspace(-1, 0, self.max_nb_trees, dtype=np.float32)
            assert len(self._zetas) == self.max_nb_trees

        self._root = State(
            np.concatenate(
                (self.X_.min(axis=0) - 1e-3, self.X_.max(axis=0) + 1e-3),
                dtype=np.float64,
            ),
            nz=np.ones(self.X_.shape[0], dtype=bool),
        )

        self._terminal_state = np.zeros(2 * self.X_.shape[1], dtype=np.float32)

        self._trees = self._build_mdp_opt_pol()
        return self

    def _build_mdp_opt_pol(self):
        """Build the Markov Decision Process (MDP) for the trees.

        This method constructs an MDP using a depth-first search approach. Each node
        in the tree represents a state in the MDP, and actions are determined based
        on potential splits from a decision tree classifier.

        Returns
        -------
        dict
            A dictionary representing the tree policies.

        References
        ----------
        .. [1] H. Kohler et. al., "Interpretable Decision Tree Search as a Markov
               Decision Process" arXiv https://arxiv.org/abs/2309.12701.
        """
        stack = [(self._root, 0)]
        expanded = [None]
        trees = {}
        while stack:
            tmp, d = stack[-1]

            if tmp is expanded[-1]:
                tmp.qs = np.zeros(
                    (len(tmp.actions), self.max_nb_trees), dtype=np.float32
                )
                for a_idx, a in enumerate(tmp.actions):
                    q = np.zeros(self.max_nb_trees, dtype=np.float32)
                    for s, p in zip(a.next_states, a.probas):
                        q += p * s.qs.max(axis=0)
                    tmp.qs[a_idx, :] = np.mean(a.rewards, axis=0) + q

                idx = np.argmax(tmp.qs, axis=0)

                trees[tuple(tmp.obs.tolist() + [d])] = [
                    tmp.actions[i].action for i in idx
                ]

                tmp.actions = None  # for memory saving

                expanded.pop()
                stack.pop()

            elif not tmp.is_terminal:
                tmp = self._expand_node(tmp, d)
                expanded.append(tmp)
                all_next_states = [
                    j for sub in [a.next_states for a in tmp.actions] for j in sub
                ]
                stack.extend((j, d + 1) for j in all_next_states)

            else:  # tmp is a terminal state
                # do backprop
                expanded[-1].actions[0].next_states[0].qs = np.zeros(
                    (1, self.max_nb_trees), dtype=np.float32
                )
                trees[tuple(tmp.obs.tolist() + [d])] = None
                stack.pop()

        return trees

    def _expand_node(self, node, depth=0):
        """Expand a node in the MDP.

        This method performs node expansion, action creation, and transition
        for the given node in the MDP.

        Parameters
        ----------
        node : State
            The node to expand.
        depth : int, default=0
            The current depth of the node.

        Returns
        -------
        State
            The expanded node.
        """
        classes, counts = np.unique(self.y_[node.nz], return_counts=True)
        rstar = max(counts) / node.nz.sum() - 1.0
        astar = classes[np.argmax(counts)]
        next_state = State(self._terminal_state, [0], is_terminal=True)
        a = Action(astar)
        a.transition([rstar] * self.max_nb_trees, 1, next_state)
        node.add_action(a)
        # If there is still depth budget and the current split has more than 1 class:
        if rstar < 0 and depth < self.max_depth:
            # Get the splits from CART
            # Note that that 2 leaf nodes means that the split is greedy.
            if depth <= len(self.cart_nodes_list) - 1:
                clf = DecisionTreeClassifier(
                    max_leaf_nodes=max(2, self.cart_nodes_list[depth]),
                    random_state=self.random_state,
                )
            # If depth budget reaches limit, get the max entropy split.
            else:
                clf = DecisionTreeClassifier(
                    max_leaf_nodes=2, random_state=self.random_state
                )

            clf.fit(self.X_[node.nz], self.y_[node.nz])

            # Extract the splits from the CART tree.

            masks = clf.tree_.feature >= 0  # get tested features.

            # Apply mask to features and thresholds to get valid indices
            valid_features = clf.tree_.feature[masks]
            valid_thresholds = clf.tree_.threshold[masks]
            lefts = (
                self.X_[:, valid_features] <= valid_thresholds
            )  # is a 2D array with nb CART tree tests columns.
            rights = np.logical_not(
                lefts
            )  # as many rows as data in the whole training set.

            # Masking data passing threshold and precedent thresholds.
            lefts *= node.nz[:, np.newaxis]
            rights *= node.nz[:, np.newaxis]

            # Compute probabilities
            p_left = lefts.sum(axis=0) / node.nz.sum()  # summing column values.
            # Non-zero values are data indices passing all tests in the MDP trajectory.
            p_right = 1 - p_left

            feat_thresh = list(
                zip(valid_features, valid_thresholds)
            )  # len of the list is nb tests in CART tree.

            # Precompute next observations for left and right splits
            next_obs_left = np.tile(node.obs, (len(feat_thresh), 1))
            next_obs_right = np.tile(node.obs, (len(feat_thresh), 1))
            indices = np.arange(len(feat_thresh))

            # The next obs bounds updated as the threshold values.
            next_obs_left[indices, self.X_.shape[1] + valid_features] = valid_thresholds
            next_obs_right[indices, valid_features] = valid_thresholds

            # Create Action objects for each split
            actions = [Action(split) for split in feat_thresh]

            # Precompute next states for left and right
            # There should be a pair of next_states per tested features.
            next_states_left = [
                State(next_obs_left[i], lefts[:, i]) for i in range(len(valid_features))
            ]
            next_states_right = [
                State(next_obs_right[i], rights[:, i])
                for i in range(len(valid_features))
            ]

            # Perform transitions, the reward is the regulizer term.
            for i in range(len(valid_features)):
                actions[i].transition(
                    self._zetas,
                    p_left[i],
                    next_states_left[i],
                )

            for i in range(len(valid_features)):
                actions[i].transition(
                    self._zetas,
                    p_right[i],
                    next_states_right[i],
                )
            for action in actions:
                node.add_action(action)
        return node

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return self._predict_zeta(X, -1)[0]  # just scores, not lengths

    def _predict_zeta(self, X, zeta_index):
        """Predict class for X using a specific zeta index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        zeta_index : int
            The index of the zeta value to use for prediction.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        lengths : float
            The average number of decision nodes traversed.
        """
        init_a = self._trees[tuple(self._root.obs.tolist() + [0])][zeta_index]
        y_pred = np.zeros(len(X), dtype=self.y_.dtype)
        lengths = np.zeros(X.shape[0], dtype=np.float32)
        for i, x in enumerate(X):
            a = init_a
            o = self._root.obs.copy()
            H = 0
            while isinstance(a, tuple):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                if x[feature] <= threshold:
                    o[x.shape[0] + feature] = threshold
                else:
                    o[feature] = threshold
                a = self._trees[tuple(o.tolist() + [H])][zeta_index]
            lengths[i] = H
            y_pred[i] = a
        return y_pred, lengths.mean()

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
        scores = np.zeros(len(self._zetas), dtype=np.float32)
        decision_path_length = np.zeros(len(self._zetas), dtype=np.float32)
        for z in range(len(self._zetas)):
            pred, decision_path_length[z] = self._predict_zeta(X, z)
            scores[z] = accuracy_score(y, pred)
        return scores, decision_path_length
