
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from kmodes.kmodes import KModes
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, AffinityPropagation, SpectralClustering

from .common import *

######### Forming groups of DTs


def avg_distance_to_clusters(tree, clusters, distance_matrix, n_clusters):
		'''
		Average distance from a tree to other clusters
		'''

		avg = 0

		for k in range(1, n_clusters+1):
			cluster_k = np.nonzero(clusters == k)[0]

			if tree not in cluster_k:
				distances = distance_matrix[tree, cluster_k]
				avg = avg+max(distances)
			
		if n_clusters > 1:					
			return avg/(n_clusters-1)
		else:
			return avg


def DTHAC(trees, distance_matrix, link_method="complete", n_clusters=None, X=None, y=None, 
		  plot=True, plot_dendrogram=False):
	'''
	Hierarchical Agglomerative Clustering for DTs
	as described in Design of Effective Multiple Classifier Systems by 
	Clustering of Classifiers; Giorgio Giacinto, Fabio Roli, and Giorgio Fumera
	'''

	def select_trees(distance_matrix, clusters, n_clusters):
		'''
		Selection of the trees that will not be pruned.
		From eah cluster, the tree with the maximum average distance to the other 
		clusters is selected.
		'''

		selected_trees = []
		for i in range(1, n_clusters+1):
			trees_in_cluster = np.nonzero(clusters == i)[0]
			max_dist = 0
			keep = -1
			
			for t in trees_in_cluster:
				d = avg_distance_to_clusters(t, clusters, distance_matrix, n_clusters)
				
				if d > max_dist:
					max_dist = d
					keep = t
	
			selected_trees.append(keep)
				
		return selected_trees
	
	dist_matrix = squareform(distance_matrix)
	link = linkage(dist_matrix, method=link_method)   ### calculate HAC tree

	if plot_dendrogram:
		plt.figure()
		dendrogram(link)
		
	if n_clusters:

		clusters = fcluster(link, n_clusters, criterion='maxclust')  ### form a maximum of n_clusters clusters
																	 ### can be inferior to n_clusters
		labs, weights = np.unique(clusters, return_counts=True)

		n_clusters_ = len(labs)
		representatives_idx = select_trees(distance_matrix, clusters, n_clusters_)
		representatives = [trees[i] for i in representatives_idx]
		
		clusters = clusters-1

		clusters_ = []
		for _ in range(n_clusters_):
			clusters_.append([])
		for i, c in enumerate(clusters):
			clusters_[c].append(trees[i])

		return n_clusters_, clusters_, clusters, weights, representatives

	else:   # choose the "best" nb clusters

		best_k = 0
		best_accuracy = -1
		accuracies = []
		nb_clusters = []

		for k in range(2, len(rf.estimators_)+1):
			# print("#########k:", k)
				
			clusters = fcluster(link, k, criterion='maxclust')
			labs, weights = np.unique(clusters, return_counts=True)
			w = weights

			if len(labs) == k:
				nb_clusters.append(k)
				selected_trees = select_trees(distance_matrix, clusters, len(labs))
				
				accuracy = score_trees([trees[j] for j in selected_trees], X, y, weights=w)
				accuracies.append(accuracy)

				if accuracy > best_accuracy:
					best_k = k
					best_accuracy = accuracy
					representatives_idx = selected_trees
					best_clusters = clusters
					best_weights = w		
		
		best_clusters = best_clusters-1
		
		clusters_ = [] 
		for _ in range(best_k):
			clusters_.append([])
		for i, c in enumerate(best_clusters):
			clusters_[c].append(trees[i])

		representatives = [trees[i] for i in representatives_idx]

		if plot:
			plt.plot(nb_clusters, accuracies, ".-", label="Clusters")
			plt.plot(best_k, best_accuracy, "*", label=("k = "+str(best_k) +
					"  acc = "+str(np.round(best_accuracy, decimals=2))))
			plt.ylim((0, 1.1))
			plt.xticks(nb_clusters[::2], nb_clusters[::2], rotation=45)
			plt.ylabel("Accuracy on validation set")
			plt.xlabel("Number of clusters")

		return best_k, clusters_, best_clusters, best_weights, representatives


def DTKmedoids(trees, distance_matrix, max_iter=100, n_clusters=None, X=None, y=None,  plot=True):
	'''
	K-medoids (Voronoi) algorithm for DTs
	Based on the implementation by Timo Erkkilä and Antti Lehmussola in
	scikit-learn's github repository
	'''

	def main_alg(n_clust, distance_matrix, max_iter):
		n_iter = 0
		medoids = np.random.permutation(distance_matrix.shape[0])[:n_clust]
		old_medoids = np.zeros((n_clust,))

		while not np.all(old_medoids == medoids) and n_iter < max_iter:
			n_iter = n_iter+1

			old_medoids = np.copy(medoids)
			clusters = np.argmin(distance_matrix[medoids, :], axis=0)

			for i in range(n_clust):
				current_cost = np.sum(distance_matrix[medoids[i], clusters == i])
					
				D = distance_matrix[clusters == i, :]
				D = D[:, clusters == i]

				all_costs = np.sum(D, axis=1)
				min_cost_idx = np.argmin(all_costs)
				min_cost = all_costs[min_cost_idx]

				if min_cost < current_cost:
					medoids[i] = np.where(clusters == i)[0][min_cost_idx]
			
		return medoids, clusters

	if n_clusters:

		representatives_idx, clusters = main_alg(n_clusters, distance_matrix, max_iter)
		_, weights = np.unique(clusters, return_counts=True)

		# clusters = clusters+1

		clusters_ = [] 
		for _ in range(n_clusters):
			clusters_.append([])
		for i, c in enumerate(clusters):
			clusters_[c].append(trees[i])

		representatives = [trees[i] for i in representatives_idx]
								
		return n_clusters, clusters_, clusters,  weights, representatives

	else:

		best_k = 0
		best_accuracy = -1
		# not_pruned = []
		best_clusters = []
		accuracies = []

		# distance_matrix = compare_trees_from_forest(rf, distance_function, **params)

		for k in range(2, len(trees)+1):
			# print("#########k:", k)

			medoids, clusters = main_alg(k, distance_matrix, max_iter)
				
			_, weights = np.unique(clusters, return_counts=True)
			
			accuracy = score_trees([trees[m] for m in medoids], X, y, weights=weights)
			accuracies.append(accuracy)

			if accuracy > best_accuracy:
				best_k = k
				best_accuracy = accuracy
				representatives_idx = medoids
				best_clusters = clusters
				best_weights = weights
		
		# best_clusters = best_clusters+1
		
		clusters_ = [] 
		for _ in range(best_k):
			clusters_.append([])
		for i, c in enumerate(best_clusters):
			clusters_[c].append(trees[i])

		representatives = [trees[i] for i in representatives_idx]

		if plot:
			plt.plot(range(2, len(trees)+1), accuracies, ".-", label="Subforests")
			plt.plot(best_k, best_accuracy, "*", 
					label=("k = "+str(best_k)+"  acc = "+str(np.round(best_accuracy, decimals=2))))
			plt.ylim((0, 1.1))
			plt.xticks(range(2, len(trees)+1)[::2], 
					   range(2, len(rf.estimators_)+1)[::2], rotation=45)	
			plt.ylabel("Accuracy on validation set")
			plt.xlabel("Number of clusters")

		return best_k, clusters_, best_clusters, best_weights, representatives


def DTDBSCAN(trees, distance_matrix, **params):

	db = DBSCAN(metric="precomputed", **params)
	db.fit(distance_matrix)

	clusters = db.labels_
	# print(db.labels_)

	n_clusters = len(set(clusters))  #- (1 if -1 in clusters else 0)
	n_noise_ = len(np.nonzero(clusters == -1)[0])
	print("n_clusters: ", n_clusters)
	print("%_noise: ", n_noise_/len(trees)*100)

	if n_noise_ > 0:
		clusters = clusters+1
	# else:
	# 	clusters = clusters+1
	_, weights = np.unique(clusters, return_counts=True)

	clusters_ = [] 
	for _ in range(n_clusters):
		clusters_.append([])
	for i, c in enumerate(clusters):
		clusters_[c].append(trees[i])

	if len(db.core_sample_indices_) == 0:
		representatives = [trees[0]]
	else:
		representatives = []
		for i in db.core_sample_indices_:
			representatives.append(trees[i])

	return n_clusters, clusters_, clusters,  weights, representatives


def DTAffinityPropagation(trees, distance_matrix, **params):
	affinity_matrix = 1-distance_matrix

	af = AffinityPropagation(affinity="precomputed", **params)
	af.fit(affinity_matrix)

	clusters = af.labels_
	# print(db.labels_)

	n_clusters = len(set(clusters))  #- (1 if -1 in clusters else 0)
	print("n_clusters: ", n_clusters)
	_, weights = np.unique(clusters, return_counts=True)

	# clusters = clusters+1

	clusters_ = [] 
	for _ in range(n_clusters):
		clusters_.append([])
	for i, c in enumerate(clusters):
		clusters_[c].append(trees[i])

	representatives = []
	for i in af.cluster_centers_indices_:
		representatives.append(trees[i])

	return n_clusters, clusters_, clusters,  weights, representatives


def DTSpectralClustering(trees, distance_matrix, **params):
	affinity_matrix = 1-distance_matrix

	sc = SpectralClustering(affinity="precomputed", **params)
	sc.fit(affinity_matrix)

	clusters = sc.labels_
	# print(db.labels_)

	n_clusters = len(set(clusters))  #- (1 if -1 in clusters else 0)
	print("n_clusters: ", n_clusters)
	_, weights = np.unique(clusters, return_counts=True)

	# clusters = clusters+1

	clusters_ = [] 
	for _ in range(n_clusters):
		clusters_.append([])
	for i, c in enumerate(clusters):
		clusters_[c].append(trees[i])

	representatives = [trees[0]]  # for compatibility

	return n_clusters, clusters_, clusters,  weights, representatives


def DTSamplingFeatureImp(trees, n_trees, n_groups):
	'''
	Generate groups of trees by sampling. Each group corresponds to a feature, and the
	trees inside each group are chosen randomly using their feature importances as probabilities
	'''

	groups = []

	distributions = np.zeros((len(trees), trees[0].n_features_))
	group_dists = [np.random.randint(trees[0].n_features_) for i in range(n_groups)]
	
	### each group corresponds to the distribution of one feature

	for i, t in enumerate(trees):
		distributions[i, :] = t.feature_importances_
		
	distributions = distributions/np.sum(distributions, axis=0)  ### feature importances to probability 
																 ### distributions

	for i in range(n_groups):
		idx = np.random.choice(len(trees), size=n_trees, 
			                   p=distributions[:, group_dists[i]])  ### choose  n_trees according to the 
							   									   ### distribution of the group
		groups.append([trees[k] for k in idx])
		
	return n_groups, groups
	
	
def DTSamplingUnif(trees, n_trees, n_groups):
	'''
	Generate groups of trees by sampling. Each group randomly chooses trees 
	with a uniform distribution.
	'''
	groups = []

	for _ in range(n_groups):
		idx = np.random.choice(len(trees), size=n_trees)  ### choose n_trees randomly
		groups.append([trees[k] for k in idx])
	
	return n_groups, groups


def DTCombinations(trees, combination_length):
	'''
	Generate groups of trees by combinations
	'''

	if combination_length == 0 or combination_length == len(trees):
		return 1, [trees]

	groups = []
	combs = list(combinations(trees, combination_length))

	for c in combs:
		groups.append(list(c))
	
	n_groups = len(groups)

	return n_groups, groups


########### Pruning


def keep_k_trees(tree_list, k, X, y):
	'''
	Order the trees according to their accuracies on X and keep the k
	most accurate.
	'''

	acc = []

	for t in tree_list:
		acc.append(t.score(X, y))
	
	idx = np.argsort(acc)

	keep = [tree_list[i] for i in idx[-k:]]

	return 1, [keep]


def CLUBDRF(tree_list, n_clusters, Xtr, Xval, yval):
	'''
	Implementation of the CLUBDRF algorithm introduced in 
	On Extreme Pruning of Random Forest Ensembles
	for Real-time Predictive Applications by
	Khaled Fawagreh, Mohamed Medhat Gaber, and Eyad Elyan, Member, IEEE
	'''

	def diversity(labels1, labels2, **_):  ## ratio of matching decisions to total decisions
		return np.sum(labels1 != labels2, axis=1)/labels1.shape[1]
	
	class_labels = np.zeros((len(tree_list), Xtr.shape[0]))
	for i, t in enumerate(tree_list):
		class_labels[i, :] = t.predict(Xtr)
	
	km = KModes(n_clusters=n_clusters, cat_dissim=diversity, n_init=1)  ### clustering by k-modes
	km.fit(class_labels)

	weights = np.unique(km.labels_, return_counts=True)[1]

	keep = []

	for c in range(n_clusters):  #### the representatives are the most accurate trees from each cluster
		best_score = 0
		best_tree = None
		
		idx = np.nonzero(km.labels_ == c)[0]
		trees = [tree_list[i] for i in idx]

		for t in trees:
			if t.score(Xval, yval) > best_score:
				best_tree = t
		
		keep.append(best_tree)

	# print(km.labels_)
	clusters = [] 
	for _ in range(n_clusters):
		clusters.append([])
	for i, c in enumerate(km.labels_):
		clusters[c].append(tree_list[i])

	return n_clusters, clusters, km.labels_, weights, keep

########## Clustering evaluation


def silhouette(clusters, distance_matrix, plot=True):
	silhouettes = silhouette_samples(distance_matrix, clusters, metric="precomputed")
	score = np.mean(silhouettes)

	if plot:
		plt.figure()

		n_clusters = len(set(clusters))

		y_lower = 10

		for i in range(n_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort thems
			ith_cluster_silhouette_values = silhouettes[clusters == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.nipy_spectral(float(i) / n_clusters)
			plt.fill_betweenx(np.arange(y_lower, y_upper),
								0, ith_cluster_silhouette_values,
								facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			# plt.text(-0.075, y_lower + 0.5 * size_cluster_i, classes[i])

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		plt.axvline(x=score, color="red", linestyle="--", label="Mean silhouette")
		plt.axvline(x=0, color="black", linestyle="-")

		plt.yticks([])  # Clear the yaxis labels / ticks
		plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
		plt.xlabel("Silhouette")
		plt.suptitle("Sample silhouettes")
		plt.legend()

	return score


def averageDistanceRatio(distance_matrix, clusters):
	d = 0
	l, counts = np.unique(clusters, return_counts=True)
	n_clusters = len(l)
	
	idx = np.triu_indices_from(distance_matrix, k=1)
	upper = distance_matrix[idx]
	d_tot = upper[upper.nonzero()].mean()
		
	for c in range(n_clusters):
		if counts[c] > 1:
			D = distance_matrix[clusters == c, :]
			D = D[:, clusters == c]
			
			idx = np.triu_indices_from(D, k=1)
			upper = D[idx]
			d = d+upper[upper.nonzero()].mean()
	
	d = d/n_clusters
	
	return d/d_tot
