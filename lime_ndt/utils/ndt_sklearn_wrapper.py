import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RTF.nrf.Neural_Decision_Tree import NDTRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class NDTRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, D, gammas=[10, 1], tree_id=None, sigma=0, 
                 gamma_activation=True, max_depth=5, random_state=42):
        self.D = D
        self.gammas = gammas
        self.tree_id = tree_id
        self.sigma = sigma
        self.gamma_activation = gamma_activation
        self.max_depth = max_depth
        self.random_state = random_state
        self.ndt = None

    def fit(self, X, y, sample_weight=None):
        self.ndt = NDTRegressor(D=self.D, gammas=self.gammas, tree_id=self.tree_id, 
                                sigma=self.sigma, gamma_activation=self.gamma_activation)
        # Ici, il faut fournir un arbre de décision pour initialiser les poids
        tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                     random_state=self.random_state).fit(X, y)
        self.ndt.compute_matrices_and_biases(tree)
        self.ndt.to_keras(loss='mean_squared_error')
        self.ndt.fit(X, y, epochs=10)
        self.intercept_ = np.mean(y)
        return self

    def predict(self, X):
        return self.ndt.predict(X).flatten()

    @property
    def coef_(self):
        # À adapter : retourner des coefficients pour compatibilité LIME
        return self.ndt.W_in_nodes.values.mean(axis=1)