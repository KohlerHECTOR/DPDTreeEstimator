import pytest
from sklearn.utils.estimator_checks import check_estimator

from dpdt import (
    DPDTreeClassifier,
    DPDTreeRegressor,
    GradientBoostingDPDTClassifier,
    QuantileClassifier,
    QuantileClassifierLight,
    TopKTreeClassifier,
    TopKTreeClassifierLight,
)

# parametrize_with_checks allows to get a generator of check that is more fine-grained
# than check_estimator


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_estimators", [2, 4])
@pytest.mark.parametrize(
    "cart_nodes_list",
    [
        (3,),
        (
            3,
            3,
        ),
    ],
)
@pytest.mark.parametrize("n_jobs_dpdt", [None, 4])
def test_check_estimator_gb(max_depth, n_estimators, cart_nodes_list, n_jobs_dpdt):
    check_estimator(
        GradientBoostingDPDTClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            cart_nodes_list=cart_nodes_list,
            n_jobs_dpdt=n_jobs_dpdt,
        )
    )


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("max_nb_trees", [1, 10])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 3, 1)])
@pytest.mark.parametrize("n_jobs", [None, 4])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list, n_jobs):
    check_estimator(
        DPDTreeClassifier(max_depth, max_nb_trees, cart_nodes_list, n_jobs=n_jobs)
    )
    check_estimator(
        DPDTreeRegressor(max_depth, max_nb_trees, cart_nodes_list, n_jobs=n_jobs)
    )


# @pytest.mark.parametrize("max_depth", [2, 3])
# @pytest.mark.parametrize("max_nb_trees", [1, 10])
# @pytest.mark.parametrize("kmeans_centers_list", [(3,), (3, 3, 1)])
# @pytest.mark.parametrize("n_jobs", [None, 4])
# def test_check_estimator_kmeans(max_depth, max_nb_trees, kmeans_centers_list, n_jobs):
#     check_estimator(
#         DrTTreeClassifier(max_depth, max_nb_trees, kmeans_centers_list, n_jobs=n_jobs)
#     )


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("max_nb_trees", [1, 10])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("n_jobs", [None, 4])
def test_check_estimator_topk(max_depth, max_nb_trees, k, n_jobs):
    check_estimator(TopKTreeClassifier(max_depth, max_nb_trees, k, n_jobs=n_jobs))
    check_estimator(
        TopKTreeClassifier(
            max_depth, max_nb_trees, k, different_feat=False, n_jobs=n_jobs
        )
    )


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("max_nb_trees", [1, 10])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("n_jobs", [None, 4])
def test_check_estimator_quantile(max_depth, max_nb_trees, k, n_jobs):
    check_estimator(QuantileClassifier(max_depth, max_nb_trees, k, n_jobs=n_jobs))


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("max_nb_trees", [1, 10])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("n_jobs", [None, 4])
def test_check_estimator_topk_light(max_depth, max_nb_trees, k, n_jobs):
    check_estimator(TopKTreeClassifierLight(max_depth, max_nb_trees, k, n_jobs=n_jobs))
    check_estimator(
        TopKTreeClassifierLight(
            max_depth, max_nb_trees, k, different_feat=False, n_jobs=n_jobs
        )
    )


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("max_nb_trees", [1, 10])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("n_jobs", [None, 4])
def test_check_estimator_quantile_light(max_depth, max_nb_trees, k, n_jobs):
    check_estimator(QuantileClassifierLight(max_depth, max_nb_trees, k, n_jobs=n_jobs))
