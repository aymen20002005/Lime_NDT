# -*- coding: utf-8 -*-
"""
Neural Random Forest Implementation
"""

# import sys

# sys.path.append("../forest")

from .Neural_Decision_Tree import ndt, NDTClassifier, NDTRegressor

# from ..forest.Forest import Forest
from ..forest.DT_grouping import *
from ..forest.Subforest import *

import numpy as np
import pandas as pd
from keras import optimizers
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.layers import Concatenate, Dense, Input
from keras.models import Model
from keras.regularizers import l1  # , l2
# from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, export_graphviz

# sys.path.append("../utils")
# from common_functions import (find_parent, get_list_split_phi,
# 							  get_list_split_phi_forest,
# 							  get_parents_nodes_leaves_dic, leaves_id)


class nrf_fully_connected(ndt):
    def __init__(self, D, gammas=[10, 1], random_forest_id=None, sigma=0,
                 gamma_activation=True, is_classifier=False):
        """
        Creates and neural random forest

        Args:
                D (int): Dataset dimensionality (nb. features)
                gammas (list): Metaparameter for each layer of the neural decision tree (slope of the tanh function).
                               High gamma -> behavior of NN is closer to tree (and also harder to change).
                random_forest_id (str or int): id for the neural random forest
                sigma (float): STD for the initial weights
        """
        ndt.__init__(self, D=D, gammas=gammas, tree_id=random_forest_id,
                     sigma=sigma, gamma_activation=gamma_activation,
                     is_classifier=is_classifier)

    def compute_matrices_and_biases(self, random_forest):
        """
        Computes the weights and biases of the NRF.
        One huge NN gathering all the trees is created.
        """
        self.rf = random_forest
        if self.is_classifier:
            self.classes = self.rf.classes_
        self.ndts = []
        # Compute weights and biases of individual trees
        for i, dtree in enumerate(random_forest.estimators_):
            self.ndts.append(ndt(D=self.D, gammas=self.gammas, tree_id=i,
                                 sigma=self.sigma, gamma_activation=self.use_gamma_activation,
                                 is_classifier=self.is_classifier))
            self.ndts[i].compute_matrices_and_biases(dtree)
        # Concatenate input to nodes W
        self.W_in_nodes = pd.concat([t.W_in_nodes for t in self.ndts], axis=1)
        # Concatenate input to nodes b
        self.b_nodes = pd.concat([t.b_nodes for t in self.ndts], axis=0)
        # Concatenate nodes to leaves W
        self.W_nodes_leaves = pd.concat([t.W_nodes_leaves for t in self.ndts], axis=1, sort=False)
        self.W_nodes_leaves = self.W_nodes_leaves.fillna(0)
        # Concatenate nodes to leaves b
        self.b_leaves = pd.concat([t.b_leaves for t in self.ndts], axis=0)
        # Concatenate leaves to out W
        self.W_leaves_out = pd.concat([t.W_leaves_out for t in self.ndts], axis=0)
        self.W_leaves_out = self.W_leaves_out/self.rf.n_estimators
        # Concatenate leaves to out b
        self.b_out = pd.concat([t.b_out for t in self.ndts], axis=0)
        self.b_out = self.b_out.groupby(self.b_out.index).mean()
        # Set other parameters
        self.N = self.b_nodes.size
        self.L = self.b_leaves.size
        self.C = self.b_out.size
        # print("self.C:", self.C)
        # print("n_classes", len(self.classes))


class NRFfullClassifier(nrf_fully_connected, NDTClassifier):
    def __init__(self, D, gammas=[10, 1], random_forest_id=None, sigma=0,
                 gamma_activation=True):
        super().__init__(D=D, gammas=gammas, random_forest_id=random_forest_id,
                         sigma=sigma, gamma_activation=gamma_activation,
                         is_classifier=True)


class NRFfullRegressor(nrf_fully_connected, NDTRegressor):
    def __init__(self, D, gammas=[10, 1], random_forest_id=None, sigma=0,
                 gamma_activation=True):
        super().__init__(D=D, gammas=gammas, random_forest_id=random_forest_id,
                         sigma=sigma, gamma_activation=gamma_activation,
                         is_classifier=False)


class nrf_independent_ndt(ndt):
    def __init__(self, D, gammas=[100, 1], sigma=0, gamma_activation=True,
                 is_classifier=False):
        """
        Creates a NRF were each tree is independent of each other.
        The last decision is a weighted voting of each tree

        Args:
        D (int): dimensionality of the dataset (nb. features)
        gammas (list): Metaparameter for each layer of the neural decision tree
        (slope of the tanh function).
        High gamma -> behavior of NN is closer to tree (and also harder to change).
        """
        ndt.__init__(self, D=D, gammas=gammas, sigma=sigma, gamma_activation=gamma_activation,
                     is_classifier=is_classifier)

    def compute_matrices_and_biases(self, random_forest):
        """
        Computes the weights and biases of the NRF.
        Each DT from the RF is mapped to a NN, and a last layer is created for
        the weighted voting.

        Args:
        random_forest (sklearn.ensemble.RandomForestClassifier): Sklearn random forest
        """
        self.rf = random_forest
        if self.is_classifier:
            self.classes = self.rf.classes_
        self.ndts = []
        for i, dtree in enumerate(random_forest.estimators_):
            self.ndts.append(ndt(D=self.D, gammas=self.gammas, tree_id=i,
                                 sigma=self.sigma, gamma_activation=self.use_gamma_activation,
                                 is_classifier=self.is_classifier))
            self.ndts[i].compute_matrices_and_biases(dtree)
        # # Define averaging layer
        # if self.is_classifier:
        # 	self.W_outputs_to_output = pd.DataFrame(np.concatenate([np.eye(tree.C)
        # 															for tree in self.ndts],
        # 															axis=0),
        # 											index=[str(t.tree_id)+"_"+str(t_class)
        # 												for t in self.ndts
        # 												for t_class in t.classes],
        # 											columns=self.classes)
        # 	# print(self.W_outputs_to_output)
        # 	# print(self.W_outputs_to_output.shape)
        # 	self.b_out = pd.DataFrame(np.zeros(len(self.classes)),
        # 								index=self.classes,
        # 								columns=["CLASS_BIASES"])
        # else:
        # 	self.W_outputs_to_output = pd.DataFrame(np.concatenate([np.eye(tree.C)
        # 															for tree in self.ndts],
        # 															axis=0),
        # 											index=[str(t.tree_id)
        # 												   for t in self.ndts],
        # 											columns=["MEAN VALUES"])
        # 	print(self.W_outputs_to_output)
        # 	# print(self.W_outputs_to_output.shape)
        # 	self.b_out = pd.DataFrame(np.zeros(1),
        # 							  columns=["BIAS"])

    def to_keras(self, loss,
                 metrics=[], optimizer=optimizers.Adam,
                 kernel_regularizer=[None, None, None],
                 optimizer_params={"lr": 0.001, "beta_1": 0.9,
                                   "beta_2": 0.999, "epsilon": 1e-8,
                                   "decay": 1e-6}):
        """
        Creates keras NN model
        """

        self.input_layer = Input(shape=(self.D,))
        self.count_ops = 0

        for tree in self.ndts:
            tree.to_keras(loss=loss, metrics=metrics,
                          optimizer=optimizer, kernel_regularizer=kernel_regularizer,
                          optimizer_params=optimizer_params)
            self.count_ops = self.count_ops+tree.count_ops

        # # Define the averaging model
        # if len(tree_models) > 1:
        # 	self.concatenation_output_layers = Concatenate()([output
        # 											for output in tree_models])
        # else:
        # 	self.concatenation_output_layers = tree_models[0]
        # self.dropouts_output_layers = Dropout(dropouts[2])(self.concatenation_output_layers)
        # if self.is_classifier:
        # 	self.output_layer = Dense(len(self.classes),
        # 							  activation='softmax')(self.dropouts_output_layers)
        # else:
        # 	self.output_layer = Dense(1)(self.dropouts_output_layers)

        # self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        # self.sgd = optimizer(**optimizer_params)
        # self.model.compile(loss=loss, optimizer=self.sgd, metrics=metrics)
        # # print(self.model.summary())
        # # plot_model(self.model, to_file='model.png')
        # # print(self.model.layers[2].layers[2])

        # self.count_ops = self.count_ops + \
        # 	self.model.layers[-1].input_shape[1]*self.model.layers[-1].output_shape[1]
        # if self.is_classifier:
        # 	self.count_ops = self.count_ops+self.model.layers[-1].output_shape[1] *\
        # 						(11+10*len(self.classes))
        # else:
        # 	self.count_ops = self.count_ops+self.model.layers[-1].output_shape[1]

        # self.model.layers[-1].set_weights(weights=[self.W_outputs_to_output,
        # 								  self.b_out.values.flatten()])

    def fit(self, X, y, sparse=False, epochs=100, min_delta=0, patience=10,
            earlyStopping=True, monitor="loss", validation_data=None,
            restore_best_weights=True, verbose=0, **fit_params):

        h = []

        for nn in self.ndts:
            h.append(nn.fit(X, y, sparse=sparse, epochs=epochs, min_delta=min_delta,
                     patience=patience, earlyStopping=earlyStopping, monitor=monitor,
                     validation_data=validation_data, restore_best_weights=restore_best_weights,
                     verbose=verbose, **fit_params))

        return h

    def predict(self, X, **kwargs):

        preds = np.zeros((X.shape[0], len(self.ndts)))
        for i, nn in enumerate(self.ndts):
            p = nn.predict(X, **kwargs)
            preds[:, i] = p.reshape((p.shape[0],))
        preds = np.mean(preds, axis=1)

        return preds

    def get_activations(self,
                        X,
                        y=None):
        """
        Get activation of each layer of the NRF

        Args:
                X (numpy.array or pandas.DataFrame): Training set
                y (numpy.array or pandas.Series): training set labels

        Returns:
                list: activations for each neural decision tree and for the last layer
                of the neural random forest
        """
        output_a = pd.DataFrame(self.model.predict(X),
                                columns=self.b_out.columns)
        return [t.get_activations(X, y) for t in self.ndts]+[[output_a]]

    def get_weights_from_NN(self):
        """
        Get the NRF weights from Keras object
        """
        for tree in self.ndts:
            tree.get_weights_from_NN()
        # print(len(self.model.layers))
        # w_9 = self.model.layers[-1].get_weights()
        # self.W_outputs_to_output_nn = pd.DataFrame(w_9[0],
        # 								index=[str(t.tree_id)+"_"+str(t_class)
        # 									   for t in self.ndts for t_class in t.classes],
        # 								columns=self.classes)
        # self.b_out_nn = pd.DataFrame(w_9[1],
        # 							 index=self.classes,
        # 							 columns=["CLASS_BIASES"])

    def compute_weights_differences(self):
        """
        Compute the difference between the weights mapped from the original RF,
        and the NRF weights

        Returns:
                list: list of pandas.DataFrames containing the weight differences
        """
        # TODO
        self.get_weights_from_NN()
        # for tree in self.ndts:
        # 	tree.compute_weights_differences()
        # self.diff_W_outputs_to_output_nn = self.W_outputs_to_output -\
        # 								   self.W_outputs_to_output_nn
        # self.diff_b_out_nn = self.b_out_nn - self.b_out

        return [t.compute_weights_differences() for t in self.ndts]  # +\
        #    [[diff_W_outputs_to_output_nn, diff_b_out_nn]]

    def print_tree_weights(self):
        """
        Print the weights mapped from the original RF
        """
        for t in self.ndts:
            t.print_tree_weights()
        # print("W: outputs -> output")
        # print(self.W_outputs_to_output)
        # print("b: output")
        # print(self.b_out)

    def print_nn_weights(self):
        """
        Print the NRF weights
        """
        for t in self.ndts:
            t.print_nn_weights()
        # print("W: outputs -> output")
        # print(self.W_outputs_to_output_nn)
        # print("b: output")
        # print(self.b_out_nn)


class NRFindClassifier(nrf_independent_ndt, NDTClassifier):
    def __init__(self, D, gammas=[100, 1], sigma=0, gamma_activation=True):
        super().__init__(D=D, gammas=gammas, sigma=sigma, gamma_activation=gamma_activation,
                         is_classifier=True)

    def predict_probas(self, X, **kwargs):
        return super().predict(X, **kwargs)


class NRFindRegressor(nrf_independent_ndt, NDTRegressor):
    def __init__(self, D, gammas=[100, 1], sigma=0, gamma_activation=True):
        super().__init__(D=D, gammas=gammas, sigma=sigma, gamma_activation=gamma_activation,
                         is_classifier=False)


#### nrf_sf_ind
class nrf_sf_ind(nrf_independent_ndt):

    def __init__(self, D, gammas=[10, 1]):
        nrf_independent_ndt.__init__(self, D, gammas)

    def compute_matrices_and_biases(self, forest):
        """
        Computes the weights and biases of the NRF_SF.
        Each SF from the Forest is mapped to a NRF, and a last layer is created for the weighted voting.

        Args:
        forest (Forest): Forest generated from an RF or a Bagging classifier
        """
        self.f = forest
        self.classes = self.f.classes_
        self.C = len(self.classes)
        self.nrfs = []
        for i, sf in enumerate(forest.subforests):
            self.nrfs.append(nrf_independent_ndt(self.D, self.gammas))
            self.nrfs[i].compute_matrices_and_biases(sf)
        # Define averaging layer
        self.W_outputs_to_output = pd.DataFrame(np.concatenate([np.eye(self.C)*self.gammas[-1] for subf in self.nrfs], axis=0),
                                                # index=[str(s.tree_id)+"_"+str(c) for s in self.nrfs for s_class in s.classes],
                                                columns=self.classes)
        self.b_out = pd.DataFrame(np.zeros(self.C),
                                    index=self.classes,
                                    columns=["CLASS_BIASES"])

    def to_keras(self, loss='categorical_crossentropy',
                 metrics=[], optimizer=optimizers.Adam, kernel_regularizer=[l1(0), l1(0), l1(0)],
                 optimizer_params={"lr": 0.001, "beta_1": 0.9,
                                   "beta_2": 0.999, "epsilon": 1e-8,
                                   "decay": 1e-6}):
        """
        Creates keras NN model
        """

        self.input_layer = Input(shape=(self.D,))
        self.count_ops = 0

        sf_models = []
        for sf in self.nrfs:
            sf.to_keras(dloss, metrics, optimizer, kernel_regularizer, optimizer_params)
            sf_models.append(sf.model(self.input_layer))
            self.count_ops = self.count_ops+sf.count_ops

        # Define the averaging model
        if len(sf_models) > 1:
            self.concatenation_output_layers = Concatenate()([output for output in sf_models])
        else:
            self.concatenation_output_layers = sf_models[0]

        self.output_layer = Dense(self.C, activation='softmax')(self.concatenation_output_layers)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)

        plot_model(self.model)
        # print(self.model.summary())

        self.sgd = optimizer(**optimizer_params)
        self.model.compile(loss=loss, optimizer=self.sgd, metrics=metrics)

        self.count_ops = self.count_ops + \
            self.model.layers[-1].input_shape[1]*self.model.layers[-1].output_shape[1]
        self.count_ops = self.count_ops+self.model.layers[-1].output_shape[1]*(11+10*len(self.classes))

        self.model.layers[-1].set_weights(weights=[self.W_outputs_to_output, self.b_out.values.flatten()])

    def get_activations(self,
                        X,
                        y=None):
        """
        Get activation of each layer of the NRF

        Args:
                X (numpy.array or pandas.DataFrame): Training set
                y (numpy.array or pandas.Series): training set labels

        Returns:
                list: activations for each neural decision tree and for the last layer of the neural random forest
        """
        output_a = pd.DataFrame(self.model.predict(X), columns=self.b_out.columns)
        return [sf.get_activations(X, y) for sf in self.nrfs]+[[output_a]]

    def get_weights_from_NN(self):
        """
        Get the NRF weights from Keras object
        """
        for tree in self.ndts:
            tree.get_weights_from_NN()
        w_9 = self.model.layers[9].get_weights()
        self.W_outputs_to_output_nn = pd.DataFrame(w_9[0],
                                        index=[str(t.tree_id)+"_"+str(c) for t in self.ndts for t_class in t.classes],
                                        columns=self.classes)
        self.b_out_nn = pd.DataFrame(w_9[1],
                                    index=self.classes,
                                    columns=["CLASS_BIASES"])


class nrf_sf_full(nrf_independent_ndt):
    def __init__(self, D, gammas=[10, 1]):
        nrf_independent_ndt.__init__(self, D, gammas)

    def compute_matrices_and_biases(self, forest):

        self.f = forest
        self.classes = self.f.classes_
        self.nrfs = []
        for i, sf in enumerate(forest.subforests):
            self.nrfs.append(nrf_fully_connected(self.D, self.gammas, i))
            self.nrfs[i].compute_matrices_and_biases(sf)
        # Define averaging layer
        # self.W_outputs_to_output = pd.DataFrame(np.concatenate([np.eye(sf.C)*self.gammas[-1] for sf in self.nrfs], axis=0),
        # 										  index=[str(sf.tree_id)+"_"+str(sf_class) for sf in self.nrfs for sf_class in sf.classes],
        # 										  columns=self.classes)

        self.W_outputs_to_output = pd.DataFrame(np.concatenate([np.eye(sf.C) for sf in self.nrfs], axis=0),
                                                index=[str(sf.tree_id)+"_"+str(sf_class) for sf in self.nrfs for sf_class in sf.classes],
                                                columns=self.classes)

        self.b_out = pd.DataFrame(np.zeros(len(self.classes)),
                                    index=self.classes,
                                    columns=["CLASS_BIASES"])

    def to_keras(self, loss='categorical_crossentropy',
                metrics=[], optimizer=optimizers.Adam, kernel_regularizer=[l1(0), l1(0), l1(0)],
                optimizer_params={"lr": 0.001, "beta_1": 0.9,
                                  "beta_2": 0.999, "epsilon": 1e-8,
                                  "decay": 1e-6}):
        """
        Creates keras NN model
        """

        self.input_layer = Input(shape=(self.D,))
        self.count_ops = 0

        sf_models = []
        for sf in self.nrfs:
            sf.to_keras(loss, metrics, optimizer, kernel_regularizer, optimizer_params)
            sf_models.append(sf.model(self.input_layer))
            self.count_ops = self.count_ops+sf.count_ops

        # Define the averaging model

        if len(sf_models) > 1:
            self.concatenation_output_layers = Concatenate()([output for output in sf_models])
        else:
            self.concatenation_output_layers = sf_models[0]
        self.output_layer = Dense(len(self.classes), activation='softmax')(self.concatenation_output_layers)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        self.sgd = optimizer(**optimizer_params)
        self.model.compile(loss=loss, optimizer=self.sgd, metrics=metrics)

        self.count_ops = self.count_ops + \
            self.model.layers[-1].input_shape[1]*self.model.layers[-1].output_shape[1]
        self.count_ops = self.count_ops+self.model.layers[-1].output_shape[1]*(11+10*len(self.classes))

        # print(self.model.summary())
        self.model.layers[-1].set_weights(weights=[self.W_outputs_to_output,
                                                   self.b_out.values.flatten()])

    def get_weights_from_NN(self):
        """
        Get the NRF weights from Keras object
        """
        for sf in self.nrfs:
            sf.get_weights_from_NN()
        w_9 = self.model.layers[-1].layers[-1].get_weights()
        self.W_outputs_to_output_nn = pd.DataFrame(w_9[0],
                                        index=[str(sf.tree_id)+"_"+str(t_class) for sf in self.nrfs for t_class in sf.classes],
                                        columns=self.classes)
        self.b_out_nn = pd.DataFrame(w_9[1],
                                       index=self.classes,
                                       columns=["CLASS_BIASES"])

    def compute_weights_differences(self):
        """
        Compute the difference between the weights mapped from the original RF, and the NRF weights

        Returns:
                list: list of pandas.DataFrames containing the weight differences
        """
        self.get_weights_from_NN()
        diff_W_outputs_to_output_nn = self.W_outputs_to_output - self.W_outputs_to_output_nn
        diff_b_out_nn = self.b_out_nn - self.b_out
        return [sf.compute_weights_differences() for sf in self.nrfs] + [[diff_W_outputs_to_output_nn, diff_b_out_nn]]
if __name__ == "__main__":
	np.random.seed(0)
	from keras.regularizers import l1  #, l2
	import matplotlib.pyplot as plt
	dataset_length = 1000
	D = 2
	X = np.random.randn(dataset_length, D)*0.1
	X[0:dataset_length//2, 0] += 0.1
	X[0:dataset_length//2, 0] += 0.2
	Y = np.ones(dataset_length)
	Y[0:dataset_length//2] *= 0

	X_test = np.random.randn(dataset_length, D)*0.1
	X_test[0:dataset_length//2, 0] += 0.1
	X_test[0:dataset_length//2, 0] += 0.2
	Y_test = np.ones(dataset_length)
	Y_test[0:dataset_length//2] *= 0
	# Train a Tree
	rf = RandomForestClassifier(max_depth=5, n_estimators=20)
	rf.fit(X, Y)

	f = Forest(rf)
	f.sampling(DTSamplingFeatureImp, n_trees=5, n_groups=4)

	# a = nrf_fully_connected(D=2, gammas=[10, 1], sigma=0.)
	# a = nrf_independent_ndt(D=2, gammas=[10, 1])
	a = nrf_sf_full(D=2, gammas=[0.01, 0.01])
	# a = nrf_sf_ind(D=2, gammas=[10, 1])
	b = nrf_fully_connected(D=2, gammas=[10, 1], sigma=0.)
	a.compute_matrices_and_biases(f)
	b.compute_matrices_and_biases(rf)
	l1_coef = -4
	a.to_keras(kernel_regularizer=[l1(10**l1_coef), l1(10**l1_coef), l1(10**l1_coef)],
						 dropouts=[0.1, 0.1, 0.1])
	b.to_keras(kernel_regularizer=[l1(10**l1_coef), l1(10**l1_coef), l1(10**l1_coef)],
						 dropouts=[0.1, 0.1, 0.1])
	print(a.count_ops)
	print(b.count_ops)

	print("scores before training")
	print("SF test:", a.score(X_test, Y_test))
	print("SF train:", a.score(X, Y))
	print("RF test:", b.score(X_test, Y_test))
	print("RF train:", b.score(X, Y))
	errors_a = a.fit(X, Y, epochs=100)["loss"]
	# print(errors_a)
	plt.figure()
	plt.plot(errors_a)
	plt.title("errors SF")
	errors_b = b.fit(X, Y, epochs=100)["loss"]
	plt.figure()
	plt.plot(errors_b)
	plt.title("errors RF")

	print("scores after training")
	print("SF test:", a.score(X_test, Y_test))
	print("SF train:", a.score(X, Y))
	print("RF test:", b.score(X_test, Y_test))
	print("RF train:", b.score(X, Y))

	print("scores forest")
	print("SF test:", a.f.score(X_test, Y_test))
	print("SF train:", a.f.score(X, Y))
	print("RF test:", b.rf.score(X_test, Y_test))
	print("RF train:", b.rf.score(X, Y))
	# a.get_weights_from_NN()

	# print "Tree weights"
	# a.print_tree_weights()
	# print "NN weights"
	# a.print_nn_weights()
	# print "activations"
	# print a.get_activations(X)
	# differences = a.compute_weights_differences()
	# print(differences)
	# a.plot_old_new_network()
	# a.plot_differences()
	# a.plot_W_nn_quantiles()
	plt.show()
