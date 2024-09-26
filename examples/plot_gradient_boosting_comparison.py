"""
=====================================
Multi-class Gradient Boosting Trees
=====================================
We compare gradient boosting with DPDT against
gradient boosting with CART.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from dpdt import DPDTreeClassifier, GradientBoostingDPDTClassifier

X_train = np.load(
    "eye_movements_data/eye_movements_seed_0_x_train.npy", allow_pickle=True
)
y_train = np.load(
    "eye_movements_data/eye_movements_seed_0_y_train.npy", allow_pickle=True
)
X_test = np.load(
    "eye_movements_data/eye_movements_seed_0_x_test.npy", allow_pickle=True
)
y_test = np.load(
    "eye_movements_data/eye_movements_seed_0_y_test.npy", allow_pickle=True
)

n_estimators = 50
weak_cart = DecisionTreeClassifier(max_depth=3, random_state=42)
weak_dpdt = DPDTreeClassifier(max_depth=3)
gb_dpdt = GradientBoostingDPDTClassifier(n_estimators=n_estimators, n_jobs="best")
gb_cart = GradientBoostingDPDTClassifier(
    n_estimators=n_estimators, n_jobs="best", use_default_dt=True
)

gb_dpdt.fit(X_train, y_train)
gb_cart.fit(X_train, y_train)


dummy_clf = DummyClassifier()


def misclassification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


weak_carts_misclassification_error = misclassification_error(
    y_test, weak_cart.fit(X_train, y_train).predict(X_test)
)

weak_dpdts_misclassification_error = misclassification_error(
    y_test, weak_dpdt.fit(X_train, y_train).predict(X_test)
)

print(
    "DecisionTreeClassifier's misclassification_error: "
    f"{weak_carts_misclassification_error:.3f}"
)

print(
    "DPDTreeClassifier's misclassification_error: "
    f"{weak_dpdts_misclassification_error:.3f}"
)

boosting_errors_dpdt = pd.DataFrame(
    {
        "Number of trees": range(1, gb_dpdt.n_estimators + 1),
        "GB-DPDT": [
            misclassification_error(y_test, y_pred)
            for y_pred in gb_dpdt.staged_predict(X_test)
        ],
    }
).set_index("Number of trees")
ax = boosting_errors_dpdt.plot()
ax.set_ylabel("Misclassification error on test set")
ax.set_title("Convergence of GB-DPDT algorithm")


plt.plot(
    range(1, gb_cart.n_estimators + 1),
    [
        misclassification_error(y_test, y_pred)
        for y_pred in gb_cart.staged_predict(X_test)
    ],
)

plt.plot(
    [boosting_errors_dpdt.index.min(), boosting_errors_dpdt.index.max()],
    [weak_carts_misclassification_error, weak_carts_misclassification_error],
    color="tab:orange",
    linestyle="dotted",
)

plt.plot(
    [boosting_errors_dpdt.index.min(), boosting_errors_dpdt.index.max()],
    [weak_dpdts_misclassification_error, weak_dpdts_misclassification_error],
    color="tab:blue",
    linestyle="dashed",
)

plt.legend(["GB-DPDT", "GB-CART", "DecisionTreeClassifier", "DPDTreeClassifier"], loc=1)
plt.savefig("gb_boosting_eye_movements")
