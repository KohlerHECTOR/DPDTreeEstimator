import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

from dpdt import DPDTreeClassifier


def create_pathological_dataset(n_samples=1000):
    # Generate random points
    X = np.random.rand(n_samples, 2)
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
x, y = create_pathological_dataset()

# Create meshgrid
feature_1, feature_2 = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max()), np.linspace(x[:, 1].min(), x[:, 1].max())
)
grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

# Plot 1: Pathological dataset
ax1.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolor="black")
ax1.set_title("Pathological Dataset")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks(())
ax1.set_yticks(())

# Plot 2: DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(x, y)
DecisionBoundaryDisplay.from_estimator(tree, x, cmap=cm, alpha=0.8, ax=ax2, eps=0.5)
ax2.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolors="k")
ax2.set_title("Decision Tree")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xticks(())
ax2.set_yticks(())

# Plot 3: DPDTreeClassifier
dpd_tree = DPDTreeClassifier(max_depth=3, random_state=42, cart_nodes_list=(8,)).fit(
    x, y
)
DecisionBoundaryDisplay.from_estimator(dpd_tree, x, cmap=cm, alpha=0.8, ax=ax3, eps=0.5)
ax3.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolors="k")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xticks(())
ax3.set_yticks(())
ax3.set_title("DP Decision Tree")

# Adjust layout and save figure
plt.tight_layout()
plt.savefig("patho_bounds_comparison")
