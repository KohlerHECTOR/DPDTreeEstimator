"""
============================
Benchmarking DPDT
============================
Benchmarking DPDT using datasets from https://arxiv.org/abs/2207.08815.
"""
from time import time

import matplotlib.pyplot as plt
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dpdt import DPDTreeClassifier


def plot_benchmark_results(*benchmark_results, names=None):
    n_results = len(benchmark_results)
    if names is None:
        names = [f"Model {i+1}" for i in range(n_results)]

    n_datasets = len(benchmark_results[0])
    n_cols = 8
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    fig.suptitle("Benchmark Results")

    # Flatten the axes array for easier iteration
    axs = axs.flatten()

    colors = plt.cm.rainbow(np.linspace(0, 1, n_results))

    # Iterate through the datasets
    for i, dataset_name in enumerate(benchmark_results[0].keys()):
        ax = axs[i]

        for j, (result, name) in enumerate(zip(benchmark_results, names)):
            data = result[dataset_name]
            ax.plot(
                data["lengths"],
                data["scores"],
                color=colors[j],
                label=name + " " + str(round(data["time"], 3)) + "s",
            )

        # Set title and labels
        ax.set_title(dataset_name, fontsize=8)
        ax.set_xlabel("Tree Length", fontsize=6)
        ax.set_ylabel("Score", fontsize=6)

        # Set tick label size
        ax.tick_params(axis="both", which="major", labelsize=6)

        # Add legend
        ax.legend(fontsize=6)

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig("benchmark_classif")


def get_pareto_front_cart(clf_kwargs, X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(**clf_kwargs)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, _ = path.ccp_alphas, path.impurities
    scores = []
    lengths = []
    for ccp_alpha in ccp_alphas:
        print(ccp_alpha)
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, **clf_kwargs)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        lengths.append(clf.decision_path(X_test).sum(axis=1).mean() - 1)
    return (scores, lengths)


def benchmark(clf_cls, clf_kwargs):
    benchmark_suite = openml.study.get_suite(337)  # obtain the benchmark suite
    res_dict = {}
    for _, task_id in enumerate(
        benchmark_suite.tasks, start=8
    ):  # iterate over half of tasks  # iterate over half of tasks
        task = openml.tasks.get_task(
            task_id,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )  # download the OpenML task
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(), y.to_numpy(), test_size=0.5, random_state=42
        )

        if clf_cls is DPDTreeClassifier:
            start = time()
            clf = clf_cls(**clf_kwargs)
            clf.fit(X_train, y_train)
            scores, lengths = clf.get_pareto_front(X_test, y_test)
            end = time() - start
        elif clf_cls is DecisionTreeClassifier:
            start = time()
            scores, lengths = get_pareto_front_cart(
                clf_kwargs, X_train, y_train, X_test, y_test
            )
            end = time() - start
        else:
            raise AssertionError(
                "clf_cls should be DecisionTreeClassifier or DPDTreeClassifier"
            )

        res_dict[dataset.name] = {"scores": scores, "lengths": lengths, "time": end}
    return res_dict


dpdt_kwargs = dict(max_depth=4, n_jobs="best")
cart_kwargs = dict(max_depth=4)
res_dpdt = benchmark(DPDTreeClassifier, dpdt_kwargs)
res_cart = benchmark(DecisionTreeClassifier, cart_kwargs)

plot_benchmark_results(
    res_dpdt, res_cart, names=["DPDTreeClassifier", "DecisionTreeClassifier"]
)
