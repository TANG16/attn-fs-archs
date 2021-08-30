from sklearn import base, cluster, tree, metrics
import numpy as np

class ClusterMultitree(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__( self, n_features, n_clusters=2, max_depth=15, random_state=None):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.random_state = random_state
                
        self._clust = None
        self._trees = None
        
        self.build_submodels()
        
    def build_submodels(self):
        self._clust = cluster.KMeans(self.n_clusters, random_state=self.random_state)
        self._trees = [
            tree.DecisionTreeClassifier(
                max_depth=self.max_depth, 
                random_state=self.random_state
            ) for _ in range(self.n_clusters)
        ]
        
    def get_params(self, deep=True):
        return {
            'n_features': self.n_features,
            'n_clusters': self.n_clusters, 
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        self.build_submodels()
        
        return self
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):

        
        attn_matrices = X[:, self.n_features:]
        original_examples = X[:, :self.n_features]
        clust_labels = self._clust.fit_predict(attn_matrices)
        for label in range(self.n_clusters):
            indices = np.where(clust_labels == label)[0]
            self._trees[label].fit(original_examples[indices], y[indices])        
        
        return self
    
    def predict(self, X):

        attn_matrices = X[:, self.n_features:]
        original_examples = X[:, :self.n_features]

        predictions = np.zeros((original_examples.shape[0],))
        
        clust_labels = self._clust.predict(attn_matrices)
        
        for label in range(self.n_clusters):
            indices = np.where(clust_labels == label)[0]
            if indices.shape[0] > 0:
                predictions[indices] = self._trees[label].predict(original_examples[indices])    
        
        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        score_val = metrics.accuracy_score(y, preds)
        return score_val

