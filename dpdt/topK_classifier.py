"""Dynamic Programming Decision Tree (TopKTree) classifier implementation."""
from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.metrics import accuracy_score
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted


def gini_impurity(y):
    """
    Compute the Gini impurity of a dataset.

    Parameters
    ----------
    y : array-like
        The target values (class labels) of the dataset.

    Returns
    -------
    float
        The Gini impurity of the dataset.
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities**2)
    return gini


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

    def __init__(self, label, nz, is_terminal=False, max_action_nb=32):
        """Initialize the State object."""
        self.obs = label
        self.is_terminal = is_terminal
        self.nz = nz
        self._actions = [0] * max_action_nb
        self._counter_action = 0

    def add_action(self, action):
        """Add an action to the state.

        Parameters
        ----------
        action : Action
            The action to be added to the state.
        """
        self._actions[self._counter_action] = action
        self._counter_action += 1

    def valid_actions(self):
        """return the list of possible actions in that state.

        Returns
        -------
        valid_acts : list
            Returns non-zero actions.
        """
        return self._actions[: self._counter_action]


class Action:
    """Represent an action in the Markov Decision Process (MDP).

    Parameters
    ----------
    action : object
        The action representation (e.g., a split decision).
    """

    def __init__(self, action, reward, proba, next_s):
        """Initialize the Action object."""
        self.action_label = action
        self.rewards = reward
        self.probas = proba
        self.next_states = next_s


class TopKTreeClassifier(ClassifierMixin, BaseEstimator):
    """TopK classifier.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_nb_trees : int, default=100
        The maximum number of trees.
    random_state : int, default=42
        Fixes randomness of the classifier.
    k: int, default=8
        Number of splits to generate for each child dataset.
    n_jobs : int, default=None
        The number of jobs to run in parallel.

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
    >>> from dpdt import TopKTreeClassifier
    >>> clf = TopKTreeClassifier()
    >>> clf.fit([[0, 0], [1, 1]], [0, 1])
    TopKTreeClassifier()
    >>> clf.predict([[2., 2.]])
    array([1])
    """

    _parameter_constraints = {
        "max_depth": [Interval(Integral, 2, None, closed="left")],
        "max_nb_trees": [Interval(Integral, 1, None, closed="left")],
        "k": [Interval(Integral, 1, None, closed="left")],
        "different_feat": ["boolean"],
        "random_state": [Interval(Integral, 0, None, closed="left")],
        "n_jobs": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"best"}),
        ],
    }

    def __init__(
        self,
        max_depth=3,
        max_nb_trees=100,
        k=8,
        different_feat=True,
        random_state=42,
        n_jobs=None,
    ):
        """Initialize the TopKTreeClassifier."""
        self.max_depth = max_depth
        self.max_nb_trees = max_nb_trees
        self.k = k
        self.different_feat = different_feat
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the TopKTreeClassifier to the training data.

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
        X, y = self._validate_data(X, y)
        check_classification_targets(y)

        self._sample_weight = np.ones(len(X), dtype=np.float32)

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
            ),
            nz=np.ones(self.X_.shape[0], dtype=bool),
            max_action_nb=self.k + 1,
        )
        self._count_ops = 0
        self._terminal_state = np.zeros(2 * self.X_.shape[1], dtype=np.float64)

        self._trees = self._build_mdp_opt_pol_parallel()
        return self

    def _build_mdp_opt_pol_parallel(self):
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
        # BFS
        root = self._expand_node(self._root, 0)
        depth_0 = [
            state for action in root.valid_actions() for state in action.next_states
        ]

        trees = {}
        list_s_with_qs = [None] * len(depth_0)

        # Initialize terminal states
        for i in range(2):
            list_s_with_qs[i] = deepcopy(depth_0[i])
            list_s_with_qs[i].qs = np.zeros((1, self.max_nb_trees), dtype=np.float32)

        # Process non-terminal states
        # Copies n_jobs times the Datasets (heavy RAM)
        if self.n_jobs == "best":
            n_jobs = max(1, len(depth_0[2:]))
        else:
            n_jobs = self.n_jobs

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._dfs)(deepcopy(root), deepcopy(s), depth=1)
            for s in depth_0[2:]
        )

        for i, (s_with_qs, local_trees) in enumerate(results, start=2):
            list_s_with_qs[i] = s_with_qs
            trees.update(local_trees)
        del depth_0

        qs = np.zeros((root._counter_action, self.max_nb_trees), dtype=np.float32)
        for a_idx, a in enumerate(root.valid_actions()):
            q = sum(
                p * s.qs.max(axis=0)
                for s, p in zip(list_s_with_qs[a_idx * 2 : (a_idx + 1) * 2], a.probas)
            )
            qs[a_idx, :] = np.mean(a.rewards, axis=0) + q
        idx = np.argmax(qs, axis=0)
        root.qs = qs
        trees[tuple(root.obs.tolist() + [0])] = [
            root.valid_actions()[i].action_label for i in idx
        ]
        root._actions = None  # for memory saving
        return trees

    def _dfs(self, root_copy, state_copy, depth):
        stack = [(state_copy, depth)]
        expanded = [None, root_copy]
        local_trees = {}
        while stack:
            tmp, d = stack[-1]

            if tmp is expanded[-1]:
                # Do backprop
                qs = np.array(
                    [
                        a.rewards.mean(axis=0)
                        + sum(
                            p * s.qs.max(axis=0)
                            for s, p in zip(a.next_states, a.probas)
                        )
                        for a in tmp.valid_actions()
                    ]
                )

                idx = np.argmax(qs, axis=0)
                tmp.qs = qs
                local_trees[tuple(tmp.obs.tolist() + [d])] = [
                    tmp.valid_actions()[i].action_label for i in idx
                ]

                tmp._actions = None  # for memory saving
                expanded.pop()
                stack.pop()

            elif not tmp.is_terminal:
                tmp = self._expand_node(tmp, d)
                expanded.append(tmp)
                stack.extend(
                    (s, d + 1) for a in tmp.valid_actions() for s in a.next_states
                )

            else:  # tmp is a terminal state
                # Set qs for terminal states
                for next_state in expanded[-1].valid_actions()[0].next_states:
                    next_state.qs = np.zeros((1, self.max_nb_trees), dtype=np.float32)
                stack.pop()
        return state_copy, local_trees

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
        weighted_counts = np.zeros_like(counts, dtype=np.float32)
        for c_idx, c in enumerate(classes):
            weighted_counts[c_idx] = (
                self._sample_weight[node.nz][self.y_[node.nz] == c]
            ).sum()
        max_idx = np.argmax(weighted_counts)
        rstar = weighted_counts[max_idx] / self._sample_weight[node.nz].sum() - 1.0
        astar = classes[max_idx]

        rew = np.full((2, self.max_nb_trees), rstar, dtype=np.float32)

        terminal_state = State(label=self._terminal_state, nz=[0], is_terminal=True)
        node.add_action(Action(astar, rew, (1, 0), (terminal_state, terminal_state)))

        if rstar < 0 and depth < self.max_depth:
            n_features = self.X_.shape[1]
            splits = []

            for feature in range(n_features):
                feature_values = self.X_[node.nz][:, feature]
                unique_values = np.unique(feature_values)
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

                for threshold in thresholds:
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask

                    left_gini = gini_impurity(self.y_[node.nz][left_mask])
                    right_gini = gini_impurity(self.y_[node.nz][right_mask])

                    n_left = np.sum(left_mask)
                    n_right = np.sum(right_mask)
                    n_total = len(self.y_[node.nz])

                    weighted_gini = (n_left / n_total) * left_gini + (
                        n_right / n_total
                    ) * right_gini

                    splits.append((feature, threshold, weighted_gini))

            # Sort splits by gini impurity (ascending) and select top k

            sorted_splits = sorted(splits, key=lambda x: x[2])

            if not self.different_feat:
                top_k_splits = sorted_splits[: min(self.k, len(splits))]
            else:
                top_k_splits = []
                seen_feat = []
                while len(top_k_splits) < min(self.k, self.X_.shape[1]) and len(sorted_splits) > 0:
                    split = sorted_splits.pop(0)
                    if split[0] not in seen_feat:
                        top_k_splits.append(split)

            # Return only the feature indices and thresholds
            valid_features = np.array([feature for feature, _, _ in top_k_splits])
            valid_thresholds = np.array([threshold for _, threshold, _ in top_k_splits])

            lefts = (self.X_[:, valid_features] <= valid_thresholds) & node.nz[
                :, np.newaxis
            ]
            rights = ~lefts & node.nz[:, np.newaxis]

            p_left = lefts.sum(axis=0) / node.nz.sum()
            p_right = 1 - p_left

            feat_thresh = list(zip(valid_features, valid_thresholds))

            next_obs_left = np.tile(node.obs, (len(feat_thresh), 1))
            next_obs_right = np.tile(node.obs, (len(feat_thresh), 1))

            next_obs_left[
                np.arange(len(feat_thresh)), self.X_.shape[1] + valid_features
            ] = valid_thresholds
            next_obs_right[np.arange(len(feat_thresh)), valid_features] = (
                valid_thresholds
            )
            act_max = len(feat_thresh)

            next_states_left = [
                State(obs, nz, max_action_nb=act_max + 1)
                for obs, nz in zip(next_obs_left, lefts.T)
            ]
            next_states_right = [
                State(obs, nz, max_action_nb=act_max + 1)
                for obs, nz in zip(next_obs_right, rights.T)
            ]

            actions = [
                Action(split, np.tile(self._zetas, (2, 1)), (pl, pr), (sl, sr))
                for split, pl, pr, sl, sr in zip(
                    feat_thresh, p_left, p_right, next_states_left, next_states_right
                )
            ]

            for action in actions:
                node.add_action(action)
            self._count_ops += len(actions)
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
        return (y_pred, lengths.mean())

    def get_pareto_front(self, X, y):
        """Compute the decision path lengths / test accuracy Pareto front of TopKTrees.

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

        if self.n_jobs == "best":
            n_jobs = max(1, len(self._zetas))
        else:
            n_jobs = self.n_jobs

        results = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
        )(delayed(self._predict_zeta)(X, z) for z in range(len(self._zetas)))

        for z, pred_length in enumerate(results):
            scores[z] = accuracy_score(y, pred_length[0])
            decision_path_length[z] = pred_length[1]
        return scores, decision_path_length
