from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import gini_impurity
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class SplitGenerator(ABC):
    """
    Abstract base class for split generators.

    This class defines the interface for split generators used in decision tree algorithms.
    """

    @abstractmethod
    def split(self, X, y, sample_weight):
        """
        Generate a set of features and thresholds for splitting a dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        list of tuple
            A list of tuples, where each tuple contains:
            - An int representing the feature index to split on.
            - A float representing the threshold value for that feature.
        """
        pass


class CARTSplitter(SplitGenerator):
    """
    CART (Classification and Regression Trees) splitter.

    This class uses scikit-learn's DecisionTreeClassifier or DecisionTreeRegressor
    to generate splits based on the CART algorithm.

    Parameters
    ----------
    cart_kwargs : dict
        Keyword arguments to pass to the DecisionTreeClassifier or DecisionTreeRegressor.
    is_classif : bool, optional
        If True, use DecisionTreeClassifier; otherwise, use DecisionTreeRegressor.
        Default is True.
    """

    def __init__(self, is_classif, cart_kwargs):
        if is_classif:
            self.clf = DecisionTreeClassifier(**cart_kwargs)
        else:
            self.clf = DecisionTreeRegressor(**cart_kwargs)
    
    def split(self, X, y, sample_weight):
        """
        Generate splits using the CART algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        sample_weights : array-like, shape (n_samples,)
            The weights of the samples.

        Returns
        -------
        list of tuple
            A list of tuples containing feature indices and threshold values.
        """
        self.clf.fit(X, y, sample_weight)
        masks = self.clf.tree_.feature >= 0
        valid_features = self.clf.tree_.feature[masks]
        valid_thresholds = self.clf.tree_.threshold[masks]
        return list(zip(valid_features, valid_thresholds))
    

class TopKSplitter(SplitGenerator):
    """
    Top-K splitter based on Gini impurity.

    This class generates splits by selecting the top K splits with the lowest
    weighted Gini impurity.

    Parameters
    ----------
    k : int
        The number of top splits to return.
    """

    def __init__(self, k):
        self.k = k
    
    def split(self, X, y, sample_weight):
        """
        Generate top K splits based on Gini impurity.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        list of tuple
            A list of tuples containing the top K feature indices and threshold values.
        """
        n_features = X.shape[1]
        splits = []

        for feature in range(n_features):
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                left_gini = gini_impurity(y[left_mask])
                right_gini = gini_impurity(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)

                weighted_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

                splits.append((feature, threshold, weighted_gini))

        # Sort splits by gini impurity (ascending) and select top k
        top_k_splits = sorted(splits, key=lambda x: x[2])[:self.k]

        # Return only the feature indices and thresholds
        return [(feature, threshold) for feature, threshold, _ in top_k_splits]
    

class OptSplitter(SplitGenerator):
    """
    Optimal splitter that considers all unique values for each feature.

    This class generates splits by considering all unique values for each feature
    in the dataset.
    """

    def split(self, X, y, sample_weight):
        """
        Generate splits for all unique values of each feature.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        list of tuple
            A list of tuples containing feature indices and threshold values for all
            unique values of each feature.
        """
        splits = []
        for feat in range(X.shape[1]):
            unique_thresh = np.unique(X[:,feat])
            for thresh in unique_thresh:
                splits.append((feat, thresh))
        return splits
        





