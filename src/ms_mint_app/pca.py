
import numpy as np
from scipy import linalg

class SciPyPCA:
    """
    A lightweight PCA implementation using scipy.linalg to replace sklearn.decomposition.PCA.
    Matches the sklearn API for n_components, fit, transform, fit_transform.
    """
    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.random_state = random_state  # Not used in analytical SVD, but kept for API compatibility
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.singular_values_ = None
        self.n_components_ = None  # Set after fit

    def fit(self, X):
        X = np.asarray(X)
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
            
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Compute SVD
        # X_centered = U * S * Vt
        # We use full_matrices=False to get the economy SVD
        try:
             U, S, Vt = linalg.svd(X_centered, full_matrices=False)
        except linalg.LinAlgError:
             # Fallback or robust handling could go here
             raise ValueError("SVD computation failed")
        
        # 3. Calculate explained variance
        # explained_variance_ = (S ** 2) / (n_samples - 1)
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        # Handle zero variance edge case
        if total_var == 0:
             explained_variance_ratio_ = np.zeros_like(explained_variance_)
        else:
             explained_variance_ratio_ = explained_variance_ / total_var
        
        # 4. Resolve n_components
        if self.n_components is None:
            n_comps = min(n_samples, n_features)
        elif isinstance(self.n_components, (int, np.integer)):
             n_comps = int(self.n_components)
        elif 0 < self.n_components < 1:
             # Explain variance ratio
             cumulative_var = np.cumsum(explained_variance_ratio_)
             n_comps = np.searchsorted(cumulative_var, self.n_components) + 1
        else:
            n_comps = min(n_samples, n_features)
        
        if n_comps > min(n_samples, n_features):
             n_comps = min(n_samples, n_features)

        self.components_ = Vt[:n_comps]
        self.explained_variance_ = explained_variance_[:n_comps]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_comps]
        self.singular_values_ = S[:n_comps]
        self.n_components_ = n_comps
        
        return self

    def transform(self, X):
        X = np.asarray(X)
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
            
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class StandardScaler:
    """
    A lightweight StandardScaler implementation to replace sklearn.preprocessing.StandardScaler.
    Standardizes features by removing the mean and scaling to unit variance.
    """
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def fit(self, X):
        X = np.asarray(X)
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        
        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        
        if self.with_mean:
            self.mean_ = np.nanmean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        if self.with_std:
            self.var_ = np.nanvar(X, axis=0, ddof=0)
            self.scale_ = np.sqrt(self.var_)
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1])
            self.var_ = np.ones(X.shape[1])
        
        return self

    def transform(self, X):
        X = np.asarray(X)
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X)
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X
