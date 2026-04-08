import numpy as np
import scipy.sparse as sp
import pandas as pd

class PearsonResidualsScaler:
    """
    Implements Analytic Pearson Residuals for scRNA-seq normalization.
    
    This method computes residuals based on a Negative Binomial null model.
    It is used to identify highly variable genes and to provide a variance-stabilized
    representation of count data without the need for pseudo-counts or log-transformation.
    """

    def __init__(self, theta=100, clip=None, feature_names=None):
        """
        Args:
            theta (float): Overdispersion parameter. As theta -> infinity, 
                          the model converges to a Poisson distribution.
            clip (float, optional): Maximum absolute value for residuals. 
                                   Defaults to sqrt(number of observations).
            feature_names (list-like, optional): Names of genes/features for reporting.
        """
        self.theta = theta
        self.clip = clip
        self.feature_names = np.array(feature_names) if feature_names is not None else None
        self.gene_stats = None
        self.gene_probs = None
        self.n_features = None

    def diagnose(self, X):
        """
        Performs data integrity checks to identify potential numerical issues.
        
        Args:
            X: Input matrix (raw counts).
        Returns:
            dict: Summary of diagnostic results.
        """
        n_obs, n_vars = X.shape
        data_to_check = X.data if sp.issparse(X) else X
        
        # 1. Check for invalid numerical values
        nan_count = np.isnan(data_to_check).sum()
        inf_count = np.isinf(data_to_check).sum()
        neg_count = np.any(data_to_check < 0)
        
        # 2. Check for empty rows/columns
        gene_sums = np.array(X.sum(axis=0)).flatten()
        cell_sums = np.array(X.sum(axis=1)).flatten()
        
        zero_genes = np.sum(gene_sums <= 0)
        zero_cells = np.sum(cell_sums <= 0)
        
        print("-" * 40)
        print("Data Integrity Report")
        print("-" * 40)
        print(f"NaN/Inf check: {n_vars - (nan_count + inf_count)} features passed.")
        if nan_count > 0 or inf_count > 0:
            print(f"  Critical: {nan_count} NaNs and {inf_count} Infs detected.")
            
        print(f"Negative value check: {'Passed (all values >= 0)' if not neg_count else 'Failed (negative values detected)'}.")
        
        print(f"Zero-count gene check: {n_vars - zero_genes} out of {n_vars} genes have non-zero expression.")
        if zero_genes > 0:
            print(f"  Note: {zero_genes} genes have zero total counts and will result in 0.0 residuals.")
            if self.feature_names is not None:
                failed_names = self.feature_names[gene_sums <= 0]
                print(f"  Example zero-count genes: {failed_names[:5].tolist()}")
                
        print(f"Zero-count cell check: {n_obs - zero_cells} out of {n_obs} cells have non-zero total counts.")
        print("-" * 40)
        
        return {
            'has_nan_inf': (nan_count + inf_count) > 0,
            'has_negative': neg_count,
            'zero_genes': zero_genes
        }

    def fit(self, X):
        """
        Fits the Pearson Residual model by calculating gene probabilities 
        and diagnostic statistics.
        """
        self.diagnose(X)
        
        n_obs = X.shape[0]
        sum_total = X.sum()
        if sum_total == 0:
            raise ValueError("The total sum of the input matrix is zero.")

        # Calculate P_j (gene relative abundance)
        sums_genes = np.array(X.sum(axis=0)).flatten()
        self.gene_probs = (sums_genes / sum_total).reshape(1, -1)
        self.n_features = X.shape[1]
        
        # Observed statistics
        obs_mean = sums_genes / n_obs
        avg_depth = sum_total / n_obs
        
        # Expected statistics (at average sequencing depth)
        # mean_expected_mu = avg_n_i * p_j
        mu_avg = self.gene_probs.flatten() * avg_depth
        expected_var = mu_avg + (np.square(mu_avg) / self.theta)

        # Calculate Residual Variance (Key for Scanpy-style plots)
        # z = (X - mu) / sqrt(mu + mu^2/theta)
        sums_cells = np.array(X.sum(axis=1)).reshape(-1, 1)
        mu_matrix = sums_cells @ self.gene_probs
        variation_matrix = mu_matrix + (np.square(mu_matrix) / self.theta)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            X_dense = X.toarray() if sp.issparse(X) else X
            # Compute unclipped residuals for variance calculation
            z_raw = (X_dense - mu_matrix) / np.sqrt(variation_matrix)
            z_raw = np.nan_to_num(z_raw, nan=0.0)
            
        residual_variance = np.var(z_raw, axis=0)

        # Store statistics for later retrieval
        self.gene_stats = pd.DataFrame({
            'mean_counts': obs_mean,
            'mean_expected_mu': mu_avg,
            'expected_variance': expected_var,
            'residual_variance': residual_variance,
            'gene_probability': self.gene_probs.flatten(),
            'is_zero_count': sums_genes <= 0
        })

        if self.feature_names is not None:
            self.gene_stats.index = self.feature_names
            
        return self

    def transform(self, X):
        """
        Transforms raw counts into clipped Pearson residuals.
        
        Math: z_ij = (x_ij - mu_ij) / sqrt(mu_ij + mu_ij^2 / theta)
        """
        if self.gene_probs is None:
            raise ValueError("The scaler must be fitted before calling transform.")
            
        sums_cells = np.array(X.sum(axis=1)).reshape(-1, 1)
        mu = sums_cells @ self.gene_probs
        
        # Negative Binomial variance formula
        variation = mu + (np.square(mu) / self.theta)
        std_dev = np.sqrt(variation)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            X_dense = X.toarray() if sp.issparse(X) else X
            residuals = (X_dense - mu) / std_dev
            # Handle cases where mu=0 (div by zero) by replacing with 0.0
            residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply clipping to reduce influence of extreme outliers
        c = self.clip if self.clip is not None else np.sqrt(X.shape[0])
        return np.clip(residuals, -c, c)

    def get_statistics(self):
        """
        Returns a DataFrame containing gene-wise fitting parameters and 
        diagnostic statistics.
        """
        return self.gene_stats

    def fit_transform(self, X):
        """Fit the model and return the transformed residuals."""
        return self.fit(X).transform(X)