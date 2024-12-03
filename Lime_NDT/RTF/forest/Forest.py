# import sys

import numpy as np
from sklearn.metrics import accuracy_score

# sys.path.append("../../utils")

from .common import *
from .DT_grouping import *
from .Subforest import *
# from comparison_functions import fleiss_score
from ..utils.comparison_functions import fleiss_score


class Forest:
	'''
	Ensemble of subforests from a random forest
	'''

	def __init__(self, cl, n_trees=None, X=None):
		trees = cl.estimators_

		if n_trees is not None:
			if n_trees < len(cl.estimators_):
				
				trees_ = []
				for t in cl.estimators_:
					if len(np.unique(t.predict(X))) == t.n_classes_:
						trees_.append(t)
				
				if len(trees_) >= n_trees:
					trees = np.random.choice(trees_, size=n_trees, replace=False)
			
		self.estimators_ = trees
		self.n_estimators = len(trees)
		self.classes_ = cl.classes_
		self.n_features = cl.n_features_
		self.n_subforests = 1
		self.subforests = [Subforest(self.estimators_, self.classes_)]
		self.weights = []
		self.clusters = []
		self.representatives = None
	
	def clustering(self, function, **params):
		self.n_subforests, subforests, self.clusters,\
			self.weights, representatives = function(self.estimators_, **params)

		self.subforests = []
		for sf in subforests:
			self.subforests.append(Subforest(sf, self.classes_))

		self.representatives = Subforest(representatives, self.classes_)
	
	def sampling(self, function, **params):
		self.n_subforests, subforests = function(self.estimators_, **params)

		self.subforests = []
		for sf in subforests:
			self.subforests.append(Subforest(sf, self.classes_))
	
	def subforestSampling(self, function, **params):
		self.weights = []
		self.n_subforests, self.subforests = function(self.subforests, self.clusters, **params)

	def predict(self, X, subforests=False):
		'''
		Weighted vote using the subforests as learners
		'''

		prediction = [] 
		subforest_prediction = np.zeros((self.n_subforests, X.shape[0]))
		proba = np.zeros((X.shape[0], len(self.classes_)))

		if np.asarray(self.weights).size == 0:
			
			for i, sf in enumerate(self.subforests):
				subforest_prediction[i, :] = sf.predict(X)
				proba = proba+sf.predict_proba(X)
			
			proba = proba/self.n_subforests
		
		else:

			for i, sf in enumerate(self.subforests):
				subforest_prediction[i, :] = sf.predict(X)
				proba = proba+sf.predict_proba(X)*self.weights[i]
			
			proba = proba/sum(self.weights)

		prediction = np.take(self.classes_, np.argmax(proba, axis=1))

		if subforests:
			return prediction, subforest_prediction
		else:
			return prediction

	def score(self, X, y, subforests=False):
		'''
		Accuracy using subforests as learners
		'''
		
		if subforests:
			y_p, y_sf = self.predict(X, subforests)

			sf_accuracy = []
			for pred in y_sf:
				sf_accuracy.append(accuracy_score(y, pred))
			
			return accuracy_score(y, y_p), sf_accuracy
		
		else:
			y_p = self.predict(X, subforests)
			return accuracy_score(y, y_p)

	def fleiss(self, X):
		'''
		Fleiss score for each subforest
		'''

		f = []

		for sf in self.subforests:
			f.append(sf.fleiss(X))

		return f

	def mean_fleiss(self, X):
		if np.asarray(self.weights).size != 0:
			return np.average(self.fleiss(X), weights=self.weights)
		else:
			return np.average(self.fleiss(X), weights=None)
	
	def sf_fleiss(self, X):
		return fleiss_score(self.subforests, X)

	def feature_importances(self):
		importances = np.zeros((self.n_subforests, self.n_features))

		for i, sf in enumerate(self.subforests):
			importances[i, :] = sf.feature_importances()
		
		return importances, np.mean(importances, axis=0)

	def newTrees(self, X, y, ratio, replace_labels, by_class=True, random_state=0):
		tree_list = []

		for sf in self.subforests:
			t = sf.newTree(X, y, ratio, replace_labels, by_class, random_state=random_state)
			tree_list.append(t)
		
		return(Subforest(tree_list, self.classes_))

	def score_representatives(self, X, y):
		
		if self.representatives is not None:
			return self.representatives.score(X, y)
