"""
============================
A Pathological Dataset
============================
A Dataset where the base :class:`sklearn.tree.DecisionTreeClassifier` fails.
"""
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

from dpdt import DPDTreeClassifier


def create_pathological_dataset(n_samples=1000):
    # Generate random points
    X = np.random.default_rng(42).random((n_samples, 2))
    y = np.zeros(n_samples)

    # Define regions for class 1 (checkerboard pattern)
    class_1_mask = (
        ((X[:, 0] < 0.5) & (X[:, 1] < 0.5))
        | ((X[:, 0] >= 0.5) & (X[:, 1] >= 0.5))
        | ((X[:, 0] < 0.25) & (X[:, 1] >= 0.5))
        | ((X[:, 0] >= 0.75) & (X[:, 1] < 0.5))
    )

    # Assign labels
    y[class_1_mask] = 1

    return X, y


# Create dataset
x, y = create_pathological_dataset(10_000)

# Create meshgrid
feature_1, feature_2 = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max()), np.linspace(x[:, 1].min(), x[:, 1].max())
)
grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

# Create figure with 3 subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3))

# Plot 1: Pathological dataset
ax1.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolor="black")
ax1.set_title("Pathological Dataset")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks(())
ax1.set_yticks(())

# Plot 2: DecisionTreeClassifier
start = time()
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x, y)
end = time() - start
score_ = tree.score(x, y)
DecisionBoundaryDisplay.from_estimator(tree, x, cmap=cm, alpha=0.8, ax=ax2, eps=0.5)
# ax2.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolors="k", alpha=0.2)
ax2.set_title(
    "Decision Tree\n Accuracy:{}%, Time: {}s".format(round(score_ * 100), round(end, 4))
)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xticks(())
ax2.set_yticks(())

# Plot 3: DPDTreeClassifier
start = time()
dpd_tree = DPDTreeClassifier(max_depth=3, random_state=0, cart_nodes_list=(8,)).fit(
    x, y
)
end = time() - start
score_ = dpd_tree.score(x, y)
DecisionBoundaryDisplay.from_estimator(dpd_tree, x, cmap=cm, alpha=0.8, ax=ax3, eps=0.5)
# ax3.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolors="k", alpha=0.2)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xticks(())
ax3.set_yticks(())
ax3.set_title(
    "DP Decision Tree\n Accuracy:{}%, Time: {}s".format(
        round(score_ * 100), round(end, 4)
    )
)


DecisionBoundaryDisplay.from_estimator(dpd_tree, x, cmap=cm, alpha=0.8, ax=ax4, eps=0.5)
# ax3.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolors="k", alpha=0.2)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_xticks(())
ax4.set_yticks(())
ax4.set_title("Opt Decision Tree\n Accuracy:100%, Time: 92s")


# Adjust layout and save figure
plt.tight_layout()
plt.savefig("patho_bounds_comparison")
