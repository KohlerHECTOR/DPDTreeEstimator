import pytest
from dpdt import DPDTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

# parametrize_with_checks allows to get a generator of check that is more fine-grained
# than check_estimator
@pytest.mark.parametrize("max_depth", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("max_nb_trees", [1, 20, 50, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list):
    check_estimator(DPDTreeClassifier(max_depth, max_nb_trees, cart_nodes_list))
