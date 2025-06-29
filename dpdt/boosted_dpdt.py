from sklearn.ensemble import AdaBoostClassifier

from .dpdt_classifier import DPDTreeClassifier


class AdaBoostDPDT(AdaBoostClassifier):
    def __init__(
        self,
        max_depth=1,
        min_samples_split=2,
        min_impurity_decrease=0,
        cart_nodes_list=(
            8,
            3,
        ),
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features=None,
        random_state=None,
        n_jobs=None,
        n_estimators=50,
        learning_rate=1,
    ):
        super().__init__(
            estimator=DPDTreeClassifier(
                max_nb_trees=1,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_impurity_decrease=min_impurity_decrease,
                cart_nodes_list=cart_nodes_list,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.cart_nodes_list = cart_nodes_list
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
