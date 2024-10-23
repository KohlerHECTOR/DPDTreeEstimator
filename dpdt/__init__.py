# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from ._version import __version__
from .dpdt_classifier import DPDTreeClassifier
from .dpdt_gb import GradientBoostingDPDTClassifier
from .dpdt_regressor import DPDTreeRegressor
from .quantile_classifier import QuantileClassifier
from .quantile_classifier_light import QuantileClassifier as QuantileClassifierLight
from .topK_classifier import TopKTreeClassifier
from .topK_classifier_light import TopKTreeClassifier as TopKTreeClassifierLight

__all__ = [
    "DPDTreeClassifier",
    "DPDTreeRegressor",
    "TopKTreeClassifier",
    "TopKTreeClassifierLight",
    "QuantileClassifier",
    "QuantileClassifierLight",
    "GradientBoostingDPDTClassifier",
    "__version__",
]
