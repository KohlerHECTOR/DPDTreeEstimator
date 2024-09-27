"""
=====================================
Multi-class Gradient Boosting Trees
=====================================
We compare gradient boosting with DPDT against
gradient boosting with CART.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_gaussian_quantiles
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dpdt import DPDTreeClassifier, GradientBoostingDPDTClassifier

X, y = make_gaussian_quantiles(
    n_samples=2_000, n_features=10, n_classes=3, random_state=1
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)
n_estimators = 50
weak_cart = DecisionTreeClassifier(max_depth=3, random_state=42)
weak_dpdt = DPDTreeClassifier(max_depth=3) #Not that weak
gb_dpdt = GradientBoostingDPDTClassifier(n_estimators=n_estimators)
gb_cart = GradientBoostingDPDTClassifier(n_estimators=n_estimators, use_default_dt=True)

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
plt.savefig("gb_boosting_compare")
