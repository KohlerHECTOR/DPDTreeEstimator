import pytest
from sklearn.utils.estimator_checks import check_estimator

from dpdt import DPDTreeClassifier, DPDTreeRegressor


# parametrize_with_checks allows to get a generator of check that is more fine-grained
# than check_estimator
@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize("max_nb_trees", [1, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1)])
@pytest.mark.parametrize("n_jobs", [None, 4])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list, n_jobs):
    check_estimator(
        DPDTreeClassifier(max_depth, max_nb_trees, cart_nodes_list, n_jobs=n_jobs)
    )
    check_estimator(
        DPDTreeRegressor(max_depth, max_nb_trees, cart_nodes_list, n_jobs=n_jobs)
    )
