"""
Subforest class
"""

# import sys
from itertools import combinations

import numpy as np

# sys.path.append("../utils")

from .common import *
from ..utils.comparison_functions import fleiss_score
from ..DT_extraction import generateNewDataClassification

########## Subforests and Subforest ensembles


class Subforest:
	'''
	Methods for subforests (groups of DTS)
	'''

	def __init__(self, tree_list, classes):
		self.estimators_ = tree_list
		self.n_estimators = len(tree_list)
		self.n_features = tree_list[0].n_features_
		self.classes_ = classes

	def predict_proba(self, X):
		return predict_trees_proba(self.estimators_, X, len(self.classes_))

	def predict(self, X):
		return predict_trees(self.estimators_, X, self.classes_)

	def score(self, X, y):
		return score_trees(self.estimators_, X, y)

	def fleiss(self, X):
		return fleiss_score(self.estimators_, X)

	def feature_importances(self):
		importances = np.zeros((self.n_estimators, self.n_features))

		for i, t in enumerate(self.estimators_):
			importances[i, :] = t.feature_importances_

		return np.mean(importances, axis=0)

	def data_augmentation(self, X, y, ratio, replace_labels, by_class):
		return generateNewDataClassification(self, X, y, ratio, replace_labels, by_class)

	def newTree(self, X, y, ratio, replace_labels, by_class=True, random_state=0):
		'''
		Generate a new tree trained with extra data generated using the subforest
		as an oracle
		'''
		newX, newY = generateNewDataClassification(self, X, y, ratio, replace_labels, by_class)

		# feat_imp = self.feature_importances()
		# feat_imp = np.floor(feat_imp/sum(feat_imp)*100).astype(int)

		# newX = scale(newX)
		# print(feat_imp)
		# newX = newX*feat_imp
		# print("newY :", np.unique(newY))
		# tmp = np.zeros((newX.shape[0], sum(feat_imp)))

		# start = 0
		# end = 0
		# print(feat_imp)
		# for i, fi in enumerate(feat_imp):
		# 	end = start+fi
		# 	tmp[:, start:end] = np.tile(newX[:,i], (fi,1)).T
		# 	start = end
		# newX = tmp

		tree = DecisionTreeClassifier(random_state=random_state)
		tree.fit(newX, newY)

		return tree  #, feat_imp


def SFSamplingCombinations(groups, clusters, combination_length, n_trees, n_sf):
	'''
	Create comb(n_clusters, combination_length) subforests with
	n_trees each by radomly selecting trees from a combination of
	clusters. Size of the clusters is taken into account in the
	sampling process.
	'''

	combs = list(combinations(range(len(groups)), combination_length))
	idx = np.random.choice(len(combs), size=n_sf, replace=False)
	combs = [combs[i] for i in idx]

	subforests = []

	for c in combs:
		probas = [len(groups[k].estimators_) for k in c]  ### create the probability
		probas = [p/sum(probas) for p in probas]    ## distribution using the
												    ## sizes of the clusters

		r = np.random.choice(c, size=n_trees, p=probas)  ### choose n_trees
		subforests.append(Subforest([np.random.choice(groups[k].estimators_) for k in r], groups[0].classes_))

	return n_sf, subforests


def SFSamplingDistance(groups, clusters, distance_matrix, n_trees, n_sf):

	def single_linkage(clusters, distance_matrix):
		labs = set(clusters)
		cluster_dists = np.zeros((len(labs), len(labs)))

		for i, c1 in enumerate(labs):
			for j, c2 in enumerate(labs):
				if i != j:
					D = distance_matrix[clusters == c1, :]
					D = D[:, clusters == c2]

					cluster_dists[i, j] = np.min(D)
					cluster_dists[j, i] = cluster_dists[i, j]

		return cluster_dists

	labs = set(clusters)
	cluster_dists = single_linkage(clusters, distance_matrix)

	subforests = []

	select_groups = np.random.choice(len(groups), size=n_sf)

	for i in select_groups:
		d = 1-cluster_dists[i, :]
		probas = d/sum(d)
		# print(1-d)
		# print(probas)
		trees_cluster = np.random.choice(len(labs), size=n_trees, p=probas)

		trees_cluster, counts = np.unique(trees_cluster, return_counts=True)
		# print(trees_cluster)
		# print(counts)

		trees = []
		for j, cl in enumerate(trees_cluster):
			trees.extend(np.random.choice(groups[cl].estimators_, size=counts[j]))

		subforests.append(Subforest(trees, groups[i].classes_))

	return n_sf, subforests
