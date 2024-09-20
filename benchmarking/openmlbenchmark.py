"""
============================
Benchmarking DPDT
============================
Benchmarking DPDT using datasets from https://arxiv.org/abs/2207.08815.
"""

import os

import numpy as np
import openml
from sklearn.model_selection import train_test_split

from dpdt import DPDTreeClassifier


def benchmark(results_folder, dpdt_kwargs, nb_seed=3):
    # Create the results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    benchmark_suite = openml.study.get_suite(337)  # obtain the benchmark suite

    for seed in range(nb_seed):
        for task_id in benchmark_suite.tasks:  # iterate over all tasks
            task = openml.tasks.get_task(task_id)  # download the OpenML task
            dataset = task.get_dataset()
            print(seed, dataset.name)
            X, y, _, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=seed
            )
            clf = DPDTreeClassifier(**dpdt_kwargs)
            clf.fit(X_train, y_train)
            scores, lengths = clf.get_pareto_front(
                X_test[: len(X_test) // 2], y_test[: len(X_test) // 2]
            )  # Half the data is left for validation
            np.save(
                results_folder + dataset.name + "_seed_" + str(seed) + "_scores", scores
            )
            np.save(
                results_folder + dataset.name + "_seed_" + str(seed) + "_lengths",
                lengths,
            )

results_folder = "dpdt_depth_5_K5"
dpdt_kwargs = dict(max_depth=5, cart_nodes_list=(16,) * 5, n_jobs="best")
benchmark(results_folder, dpdt_kwargs)
