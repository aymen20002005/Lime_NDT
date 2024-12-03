
import numpy as np
from sklearn.metrics import accuracy_score

######### Functions for DT lists


def predict_trees_proba(tree_list, X, n_classes, weights=[]):
	'''
	Weighted probability predictions for DT lists (classification)
	'''

	probas = np.zeros((X.shape[0], n_classes))

	if np.asarray(weights).size == 0:

		for t in tree_list:
			# print(t.classes_)
			probas = probas+t.predict_proba(X)

		return probas/len(tree_list)

	else:
		for i, t in enumerate(tree_list):
			probas = probas+t.predict_proba(X)*weights[i]

		return probas/sum(weights)


def predict_trees(tree_list, X, classes, weights=[]):
	'''
	Weighted predictions for DT lists (classification)
	'''

	proba = predict_trees_proba(tree_list, X, len(classes), weights)

	prediction = np.take(classes, np.argmax(proba, axis=1))

	return prediction


def score_trees(tree_list, X, y, weights=[]):
	'''
	Weighted accuracy for a group of DT
	'''

	classes = np.unique(y)
	
	y_p = predict_trees(tree_list, X, classes, weights)
		
	return accuracy_score(y, y_p)
