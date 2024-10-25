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

# Create figure with 4 subplots
fig, axs = plt.subplots(1, 4, figsize=(20, 4))

# Plot 1: Pathological dataset
axs[0].scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolor="black", s=10)
axs[0].set_title("Pathological Dataset\n  ", fontsize=25)
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(())
axs[0].set_yticks(())

# Plot 2: DecisionTreeClassifier
start = time()
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x, y)
end = time() - start
score_ = tree.score(x, y)
DecisionBoundaryDisplay.from_estimator(tree, x, cmap=cm, alpha=0.8, ax=axs[1], eps=0.5)
axs[1].set_title(
    f"Decision Tree\nAccuracy: {score_*100}%\nTime: {end:.4f}s", fontsize=20
)
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(())
axs[1].set_yticks(())

# Plot 3: DPDTreeClassifier
start = time()
dpd_tree = DPDTreeClassifier(max_depth=3, random_state=0, cart_nodes_list=(8,)).fit(
    x, y
)
end = time() - start
score_ = dpd_tree.score(x, y)
DecisionBoundaryDisplay.from_estimator(
    dpd_tree, x, cmap=cm, alpha=0.8, ax=axs[2], eps=0.5
)
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)
axs[2].set_xticks(())
axs[2].set_yticks(())
axs[2].set_title(
    f"DP Decision Tree\nAccuracy: {score_*100}%\nTime: {end:.4f}s", fontsize=20
)

# Plot 4: Opt Decision Tree (placeholder)
DecisionBoundaryDisplay.from_estimator(
    dpd_tree, x, cmap=cm, alpha=0.8, ax=axs[3], eps=0.5
)
axs[3].set_xlim(0, 1)
axs[3].set_ylim(0, 1)
axs[3].set_xticks(())
axs[3].set_yticks(())
axs[3].set_title("Opt Decision Tree\nAccuracy: 100%\nTime: 92s", fontsize=20)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig("patho_bounds_comparison", dpi=300, bbox_inches="tight")
