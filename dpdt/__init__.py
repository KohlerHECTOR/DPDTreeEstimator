# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from ._version import __version__
from .boosted_dpdt import AdaBoostDPDT
from .dpdt_classifier import DPDTreeClassifier
from .dpdt_regressor import DPDTreeRegressor

__all__ = [
    "AdaBoostDPDT",
    "DPDTreeClassifier",
    "DPDTreeRegressor",
    "__version__",
]
