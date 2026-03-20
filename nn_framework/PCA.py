import numpy as np
import matplotlib.pyplot as plt
from nn_framework.feature_scaling import StandardScaler

class PCA:
    def __init__(self, num_components):
        self.num_components = num_components
        self.principal_components = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None
        self.standard_scaler = StandardScaler()

    def __str__(self):
        return f"PCA({self.num_components})"
    
    def fit(self, X):
        # Fit and Standardize the data with StandardScaler
        self.standard_scaler.fit(X)
        X_standardized = self.standard_scaler.transform(X)
        
        # Compute the covariance matrix
        covariance_matrix = np.cov(X_standardized, rowvar=False)
        
        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the top num_components eigenvectors
        self.principal_components = eigenvectors[:, :self.num_components]
        self.eigenvalues = eigenvalues
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        self.explained_variance_ratio_ = explained_variance_ratio
    
    def transform(self, X):
        # Standardize the data
        X_standardized = self.standard_scaler.transform(X)
        
        # Project the data onto the principal components
        projected_data = np.dot(X_standardized, self.principal_components)
        
        return projected_data
    
    def plot_explained_variance_ratio(self):
        plt.plot(np.cumsum(self.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Number of Components')
        plt.grid(True)
        plt.show()
