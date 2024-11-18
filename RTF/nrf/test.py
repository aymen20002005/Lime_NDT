from Neural_Decision_Tree import NDTClassifier
from sklearn.tree import DecisionTreeClassifier  #, export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from keras.regularizers import l1
sys.path.append("../data_mngmt")
from data import load_letter
from keras.utils import to_categorical

X_s, X_t_train, X_t_test, y_s, y_t_train, y_t_test, = load_letter()
X_t = np.concatenate([X_t_train, X_t_test])
y_t = np.concatenate([y_t_train, y_t_test])
y_t_cat = to_categorical(y_t)
y_s_cat = to_categorical(y_s)


results = []

for i in range(110, 210, 10):
	print(i)
	X_t_train = X_t[:i, :]
	X_t_test = X_t[i:, :]
	y_t_train = y_t[:i]
	y_t_test = y_t[i:]
	y_t_train_cat = y_t_cat[:i, :]
	y_t_test_cat = y_t_cat[i:, :]
	for depth in range(2,  5):
		print(depth)
		for rep in range(3):
			print(rep)
			# create tree
			clf = DecisionTreeClassifier(max_depth=depth)
			clf = clf.fit(X_s, y_s)
			# create NDT
			a = NDTClassifier(D=X_s.shape[1], gammas=[1, 1, 1], tree_id=0)
			a.compute_matrices_and_biases(clf)
			a.to_keras(dropouts=[0., 0., 0.], kernel_regularizer=[l1(1e-10), l1(1e-10), l1(1e-10)])
			# compute scores
			t_source_only_tree = clf.score(X_t_test, y_t_test)
			t_source_only_NN_before_training = a.score(X_t_test, y_t_test)
			s_tree = clf.score(X_s, y_s)
			s_NN_before_training = a.score(X_s, y_s)
			# Fit NN
			errors = a.fit(X_s, y_s_cat, epochs=100, to_categorical_conversion=False)
			# compute new scores
			t_source_only_NN_after_training = a.score(X_t_test, y_t_test)
			s_NN_after_training = a.score(X_s, y_s)
			# refine on target
			errors = a.fit(X_t_train, y_t_train_cat, epochs=100, to_categorical_conversion=False)
			clf_t = DecisionTreeClassifier(max_depth=depth)
			clf_t = clf_t.fit(X_t_train,  y_t_train)
			target_only_tree = clf_t.score(X_t_test, y_t_test)
			clf_sl = DecisionTreeClassifier(max_depth=depth)
			clf_sl = clf_sl.fit(X_t_test,  y_t_test)
			skyline = clf_sl.score(X_t_test,  y_t_test)
			transfered_NN = a.score(X_t_test, y_t_test)
			transfered_NN_source = a.score(X_s, y_s)
			current_result = [i, 
							depth, 
							s_tree, 
							s_NN_before_training, 
							s_NN_after_training,
							t_source_only_tree,
							t_source_only_NN_before_training,
							t_source_only_NN_after_training,
							target_only_tree,
							skyline,
							transfered_NN,
							transfered_NN_source
							]
			results.append(current_result)
			print(current_result)
res = pd.DataFrame(results, columns=["size",
										"depth",
										"s_tree",
										"s_NN_before_training",
										"s_NN_after_training",
										"t_source_only_tree",
										"t_source_only_NN_before_training",
										"t_source_only_NN_after_training",
										"target_only_tree",
										"skyline",
										"transfered_NN",
										"transfered_NN_source"])

res_d_2 = res[res["depth"] == 2]
res_d_3 = res[res["depth"] == 3]
res_d_4 = res[res["depth"] == 4]


plt.plot(res_d_2["size"], res_d_2["t_source_only_tree"], "yo--")
plt.plot(res_d_2["size"], res_d_2["t_source_only_NN_after_training"], "yo-")
plt.plot(res_d_2["size"], res_d_2["target_only_tree"], "ro--")
plt.plot(res_d_2["size"], res_d_2["skyline"], "ko-")
plt.plot(res_d_2["size"], res_d_2["transfered_NN"], "go-")
plt.show()


plt.plot(res_d_3["size"], res_d_3["t_source_only_tree"], "yo--")
plt.plot(res_d_3["size"], res_d_3["t_source_only_NN_after_training"], "yo-")
plt.plot(res_d_3["size"], res_d_3["target_only_tree"], "ro--")
plt.plot(res_d_3["size"], res_d_3["skyline"], "ko-")
plt.plot(res_d_3["size"], res_d_3["transfered_NN"], "go-")
plt.show()


plt.plot(res_d_4["size"], res_d_4["t_source_only_tree"], "yo--")
plt.plot(res_d_4["size"], res_d_4["t_source_only_NN_after_training"], "yo-")
plt.plot(res_d_4["size"], res_d_4["target_only_tree"], "ro--")
plt.plot(res_d_4["size"], res_d_4["skyline"], "ko-")
plt.plot(res_d_4["size"], res_d_4["transfered_NN"], "go-")
plt.show()
