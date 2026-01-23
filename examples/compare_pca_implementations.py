
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA as SklearnPCA

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

    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Compute SVD
        # X_centered = U * S * Vt
        # We use full_matrices=False to get the economy SVD
        U, S, Vt = linalg.svd(X_centered, full_matrices=False)
        
        # 3. Calculate explained variance
        # explained_variance_ = (S ** 2) / (n_samples - 1)
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        
        # 4. Store results
        # Sklearn stores components as (n_components, n_features)
        # Vt is (min(n_samples, n_features), n_features)
        
        if self.n_components is None:
            n_comps = min(n_samples, n_features)
        elif 0 < self.n_components < 1:
             # Explain variance ratio
             cumulative_var = np.cumsum(explained_variance_ratio_)
             n_comps = np.searchsorted(cumulative_var, self.n_components) + 1
        else:
            n_comps = int(self.n_components)
        
        if n_comps > min(n_samples, n_features):
             n_comps = min(n_samples, n_features)

        self.components_ = Vt[:n_comps]
        self.explained_variance_ = explained_variance_[:n_comps]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_comps]
        self.singular_values_ = S[:n_comps]
        
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def run_comparison():
    print("Running PCA Comparison...")
    
    # Generate random data
    np.random.seed(42)
    n_samples = 50
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    
    # 1. Sklearn PCA
    print(f"Testing with {n_samples} samples, {n_features} features.")
    
    sklearn_pca = SklearnPCA(n_components=5)
    sklearn_trans = sklearn_pca.fit_transform(X)
    
    # 2. SciPy PCA
    scipy_pca = SciPyPCA(n_components=5)
    scipy_trans = scipy_pca.fit_transform(X)
    
    # 3. Compare Components (Handling sign ambiguity)
    # PCA components are unique up to a sign flip.
    # We check if vectors are equal or anti-parallel.
    
    # Check Explained Variance Ratio (should be identical independent of sign)
    var_diff = np.abs(sklearn_pca.explained_variance_ratio_ - scipy_pca.explained_variance_ratio_)
    print(f"Max Variance Ratio Diff: {var_diff.max():.2e}")
    assert np.allclose(sklearn_pca.explained_variance_ratio_, scipy_pca.explained_variance_ratio_), "Variance Ratios do not match!"
    
    # Check Transformed Data
    # For each component, check correlation
    print("\nChecking Component Correlations (should be 1.0 or -1.0):")
    for i in range(5):
        corr = np.corrcoef(sklearn_trans[:, i], scipy_trans[:, i])[0, 1]
        print(f"PC{i+1}: {corr:.4f}")
        assert abs(abs(corr) - 1.0) < 1e-5, f"PC{i+1} does not match!"
        
    print("\nSUCCESS: SciPyPCA implementation matches Sklearn results.")

if __name__ == "__main__":
    run_comparison()
