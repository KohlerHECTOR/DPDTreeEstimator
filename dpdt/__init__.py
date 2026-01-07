# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"
from .boosted_dpdt import AdaBoostDPDT
from .dpdt_classifier import DPDTreeClassifier
from .dpdt_regressor import DPDTreeRegressor

__all__ = [
    "AdaBoostDPDT",
    "DPDTreeClassifier",
    "DPDTreeRegressor",
    "__version__",
]
