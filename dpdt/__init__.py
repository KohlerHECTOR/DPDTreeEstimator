# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from ._version import __version__
from .dpdt_classifier import DPDTreeClassifier
from .dpdt_regressor import DPDTreeRegressor

__all__ = [
    "DPDTreeClassifier",
    "DPDTreeRegressor",
    "__version__",
]
