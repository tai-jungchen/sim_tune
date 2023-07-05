"""
File: utils.py
Name:Alex
----------------------------------------
TODO:
All the helper functions you would need!
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, \
	precision_recall_fscore_support, cohen_kappa_score, roc_curve, auc, precision_recall_curve, average_precision_score, \
	balanced_accuracy_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math
import statistics
import random
import statistics as stat
from my_logistic_regression import MyLogisticRegression


class Tuning:
	def __init__(self, penalty, solver, max_iter, classifier, hyper_metric):
		self.penalty = penalty
		self.solver = solver
		self.max_iter = max_iter
		self.classifier = classifier
		self.hyper_metric = hyper_metric

	def cross_val(self, X, y, param_1s, param_2s):
		"""
		Perform stratified 10-fold cross validation and return the best pair of hyper-parameters.
		hyper-parameter selection criteria can be accuracy or f1 score
		:param hyper_metric: <str> the hyper-param selection metric
		:param X: <np array> features
		:param y: <np array> labels
		:param param_1s: <list> a list containing the set of hyper-parameters 1
		:param param_2s: <list> a list containing the set of hyper-parameters 2
		:return: <tuple> the best pair of (hyper parameter 1, hyper-parameter 2)
		"""
		skf = StratifiedKFold(n_splits=10)
		metrics = []
		param_pair = []

		# trying different lambdas
		for param_1 in param_1s:
			for param_2 in param_2s:
				if self.classifier == 'logistic_regression':
					model = LogisticRegression(penalty=self.penalty, C=param_1, solver=self.solver, max_iter=self.max_iter)
				elif self.classifier == 'RF':
					model = RandomForestClassifier(n_estimators=param_1)
				elif self.classifier == 'mylr':
					model = MyLogisticRegression(penalty=self.penalty, C=param_1, max_iter=self.max_iter)

				metrics_among_10_folds = []
				# 10 fold cross validating
				for train_idx, test_idx in skf.split(X, y):
					X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
					model.fit(X_train, y_train)
					y_pred_test = self.tuned_prediction(model, X_test, param_2)

					# print('cross val: ')
					# print(confusion_matrix(y_test, y_pred_test, labels=[1, 0]))
					# print(classification_report(y_test, y_pred_test))

					"*** hyper param criteria ***"
					# use accuracy as criteria
					if self.hyper_metric is 'acc':
						metrics_among_10_folds.append(accuracy_score(y_test, y_pred_test))

					# use f1 score as criteria
					elif self.hyper_metric is 'f1':
						precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test)
						metrics_among_10_folds.append(f1[1])

					elif self.hyper_metric is 'f1_star':
						f1 = self.new_f1_score(y_test, y_pred_test)
						metrics_among_10_folds.append(f1)

					# use kappa as criteria
					elif self.hyper_metric is 'kappa':
						kappa = cohen_kappa_score(y_test, y_pred_test)
						metrics_among_10_folds.append(kappa)

					# use balanced accuracy as criteria
					elif self.hyper_metric is 'bacc':
						bacc = balanced_accuracy_score(y_test, y_pred_test)
						metrics_among_10_folds.append(bacc)

				"*** estimator ***"
				metrics.append(stat.mean(metrics_among_10_folds))
				# metrics.append(stat.median(metrics_among_10_folds))
				param_pair.append((param_1, param_2))
				"*** estimator ***"

		# plt.plot(list(map(str, param_1s)), metrics, 'o-')
		# plt.title('CV accuracy to $\\frac{1}{\lambda}$ plot')
		# plt.xlabel('$\\frac{1}{\lambda}$')
		# plt.ylabel('CV accuracy')
		# plt.show()

		# print(hyper_metric)
		# print(metrics)
		best_idx = metrics.index(max(metrics))
		return param_pair[best_idx]

	def ghost(self, cls, X_train, y_train, thresholds, N_subsets=100, subsets_size=0.2):
		# seeding
		# np.random.seed(random_seed)
		# random_seeds = np.random.randint(N_subsets * 10, size=N_subsets)

		# calculate prediction probability for the training set
		probs_train = cls.predict_proba(X_train)[:, 1]
		labels_train_thresh = {'labels': y_train}
		labels_train_thresh.update({'probs': probs_train})
		# recalculate the predictions for the training set using different thresholds and
		# store the predictions in a dataframe
		for thresh in thresholds:
			labels_train_thresh.update({str(thresh): [1 if x >= thresh else 0 for x in probs_train]})

		df_preds = pd.DataFrame(labels_train_thresh)
		# Optmize the decision threshold based on the f1 score / Cohen's Kappa coefficient
		# pick N_subsets training subsets and determine the threshold that provides the highest kappa on each
		# of the subsets

		kappa_accum = []
		for i in range(N_subsets):
			df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds, y_train, test_size=subsets_size,
																			stratify=y_train)
			probs_subset = list(df_subset['probs'])
			thresh_names = [x for x in df_preds.columns if (x != 'labels' and x != 'probs')]
			kappa_train_subset = []
			for col1 in thresh_names:
				precision, recall, f1, _ = precision_recall_fscore_support(labels_subset, list(df_subset[col1]))
				if self.hyper_metric == 'f1':
					kappa_train_subset.append(f1[1])
				elif self.hyper_metric == 'f1_star':
					f1 = self.new_f1_score(labels_subset, list(df_subset[col1]))
					kappa_train_subset.append(f1)
				elif self.hyper_metric == 'bacc':
					kappa_train_subset.append(balanced_accuracy_score(labels_subset, list(df_subset[col1])))
				elif self.hyper_metric == 'acc':
					kappa_train_subset.append(accuracy_score(labels_subset, list(df_subset[col1])))
				elif self.hyper_metric == 'kappa':
					kappa_train_subset.append(cohen_kappa_score(labels_subset, list(df_subset[col1])))
			kappa_accum.append(kappa_train_subset)

		# determine the threshold that provides the best results on the training subsets
		y_values_median, y_values_std = self.helper_calc_median_std(kappa_accum)
		print(f'{self.hyper_metric}: {y_values_median}')
		# opt_thresh_idx = metrics_selection(y_values_median.tolist())
		# opt_thresh = thresholds[opt_thresh_idx]
		opt_thresh = thresholds[np.argmax(y_values_median)]

		return opt_thresh

	@staticmethod
	def helper_calc_median_std(specificity):
		# Calculate median and std of the columns of a pandas dataframe
		arr = np.array(specificity)
		y_values_median = np.median(arr, axis=0)
		y_values_std = np.std(arr, axis=0)
		return y_values_median, y_values_std

	@staticmethod
	def new_f1_score(y_true, y_pred):
		r1 = sum(y_true == 1) / len(y_true)
		r0 = 1 - r1

		tn, fp, fn, tp = np.ravel(confusion_matrix(y_true, y_pred))
		p1 = (tp + fp) / (tp + fp + tn + fn)
		p0 = 1 - p1

		f1 = f1_score(y_true, y_pred)
		baseline = (2 * p1 * r1) / (p1 + r1)
		new_f1 = (f1 - baseline) / (1 - baseline)

		# acc = accuracy_score(y_true, y_pred)
		# kappa = (acc - p1*r1 - p0*r0) / (1 - p1*r1 - p0*r0)
		#
		# precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
		# new_prec = (precision[1] - r1) / (1 - r1)
		#
		# new_recall = (recall[1] - p1) / (1 - p1)
		return new_f1

	def tuned_prediction(self, model, X, decision_threshold):
		"""
		Return the predction based on the given decision threshold
		:param model: <obj> The given model
		:param X: <np array> Features
		:param decision_threshold: <float> The given decision threshold
		:return: <np array> The predicted labels based on the given decision threshold
		"""
		if self.classifier == 'logistic_regression':
			coef = model.coef_
			intercept = model.intercept_
			x = np.dot(X, coef.T) + intercept
			y_prob = Plot.logistic_func(x)
			y_pred = np.where(y_prob < decision_threshold, 0, 1)
			return y_pred.flatten()
		elif self.classifier == 'RF':
			y_prob = model.predict_proba(X)[:, 1]
			y_pred = np.where(y_prob < decision_threshold, 0, 1)
			return y_pred.flatten()
		elif self.classifier == 'svm':
			y_prob = model.predict_proba(X)[:, 1]
			y_pred = np.where(y_prob < decision_threshold, 0, 1)
			return y_pred.flatten()
		elif self.classifier == 'mylr':
			coef = model.coef_
			intercept = model.intercept_
			x = np.dot(X, coef.T) + intercept
			y_prob = Plot.logistic_func(x)
			y_pred = np.where(y_prob < decision_threshold, 0, 1)
			return y_pred.flatten()
		else:
			return 0

	@staticmethod
	def se(lst):
		"""
		Return standard error of a list of numbers
		:param lst: <list> the list of data to be calculated
		:return: the standard error of the data in the list
		"""
		return statistics.pstdev(lst) / math.sqrt(len(lst))


class Kappa:
	def __init__(self):
		pass

	@staticmethod
	def maj_kappa(y_pred, y_test):
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		p1 = (tp + fp) / (tn + fp + fn + tp)
		p0 = (tn + fn) / (tn + fp + fn + tp)
		r1 = (tp + fn) / (tn + fp + fn + tp)
		r0 = (tn + fp) / (tn + fp + fn + tp)

		acc = (tp + tn) / len(y_test)
		return (acc - p0*r0) / (1-p0*r0)

	@staticmethod
	def kappa_baseline_helper(y_pred, y_test):
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		p1 = (tp + fp) / (tn + fp + fn + tp)
		p0 = (tn + fn) / (tn + fp + fn + tp)
		r1 = (tp + fn) / (tn + fp + fn + tp)
		r0 = (tn + fp) / (tn + fp + fn + tp)

		tp_prime = (fn + tp) * p1
		fn_prime = (fn + tp) * p0
		fp_prime = (fp + tn) * p1
		tn_prime = (fp + tn) * p0
		# print(f'tp, fn, fp, tn:{tp, fn, fp, tn}')
		print(f'tp_prime, fn_prime, fp_prime, tn_prime:{tp_prime, fn_prime, fp_prime, tn_prime}')

		tp_prime = round(tp_prime)
		fn_prime = round(fn_prime)
		fp_prime = round(fp_prime)
		tn_prime = round(tn_prime)
		print(f'round up tp_prime, fn_prime, fp_prime, tn_prime:{tp_prime, fn_prime, fp_prime, tn_prime}')

		baseline = r1*p1 + r0*p0
		print(f'baseline: {round(baseline, 4)}')
		return baseline

	@staticmethod
	def non_zero_kappa_selection(kappas):
		"""
		Output the optimized metric' s index.
		If the maximum of the metrics is 0 (naive prediction), select the non-zero kappa as the optimized metric
		If all elements of the metrics is 0, select 0.
		:param kappas: <list> the metrics output by cross validation
		:return: <int> index of the optimized metric
		"""
		if sum([item == 0.0 for item in kappas]) == len(kappas):
			print('Warning: kappa is all zero.')
			return 0
		if sum([item <= 0.0 for item in kappas]) == len(kappas):
			list2 = list(set(kappas))  # remove duplicates
			list2.sort()  # sort
			opt = list2[-2]
			opt_idx = kappas.index(opt)
			return opt_idx
		return kappas.index(max(kappas))

	@staticmethod
	def naive_pred(y_test, threshold):
		"""
		Conduct naive prediction here
		:param y_test: <np array> the array of testing data
		:param threshold: <float> the decision threshold
		:return: the naive predicted array
		"""
		pred = np.ones(y_test.shape)
		for i in range(len(pred)):
			rand_num = random.random()
			if rand_num <= threshold:
				pred[i] = 0
			else:
				pred[i] = 1
		return pred

	@staticmethod
	def derived(pr0s, y_test):
		"""
		Variation of the y_test won't matter due to stratified sampling
		:param pr0s:
		:param y_test:
		:return:
		"""
		de_accs = []
		de_f1s = []
		de_precs = []
		de_recalls = []
		de_specs = []
		de_kappas = []

		for pr0 in pr0s:
			r0 = sum(y_test == 0) / len(y_test)
			r1 = sum(y_test == 1) / len(y_test)
			pr1 = 1 - pr0

			acc = pr1 * r1 + pr0 * r0
			f1 = (2 * pr1 * r1) / (pr1 + r1)
			prec = r1
			recall = pr1
			spec = pr0
			kappa = 0

			de_accs.append(acc)
			de_f1s.append(f1)
			de_precs.append(prec)
			de_recalls.append(recall)
			de_specs.append(spec)
			de_kappas.append(kappa)

		# plot the curves #
		plt.plot(pr0s, de_accs, 'o-')
		plt.plot(pr0s, de_f1s, 'o-')
		plt.plot(pr0s, de_precs, 'o-')
		plt.plot(pr0s, de_recalls, 'o-')
		plt.plot(pr0s, de_specs, 'o-')
		plt.plot(pr0s, de_kappas, 'o-')
		plt.title('Metrics-$pr_0$ relationship (derived)')
		plt.xlabel('$pr_0$')
		plt.ylabel('Metrics')
		plt.legend(['Accuracy', 'F1-score', 'Precision', 'Recall', 'Specificity', 'Kappa'])
		plt.grid()
		plt.show()


class Plot:
	def __init__(self):
		pass

	@staticmethod
	def plot_feature_importance(model, features):
		"""
		Plot out the feature importance plot
		:param model: <obj> The model that is being used to perform the prediction
		:param features: <list> The names of the features
		:return: feature importance
		"""
		important_features = []
		importance = model.coef_[0]

		# summarize feature importance
		for i, v in enumerate(importance):
			print('Feature: %0d, Score: %.5f' % (i, v))
			if v != 0:
				important_features.append(features[i])
		# plot feature importance
		plt.bar([x for x in range(len(importance))], importance)
		plt.title('Feature importance plot')
		plt.xlabel('features')
		plt.ylabel('feature importance')
		plt.show()
		return important_features

	@staticmethod
	def logistic_func(x):
		"""
		Return calculated logistic function value
		:param x: <float> The given x
		:return: <float> The calculated logistic function value
		"""
		"""Stable implementation of the sigmoid function"""
		pos_mask = (x >= 0)  # Mask for positive values
		neg_mask = (x < 0)  # Mask for negative values

		# Compute sigmoid values for positive and negative values separately
		pos_exp = np.exp(-x[pos_mask])
		neg_exp = np.exp(x[neg_mask])

		# Combine the results
		sigmoid_vals = np.zeros_like(x)
		sigmoid_vals[pos_mask] = 1 / (1 + pos_exp)
		sigmoid_vals[neg_mask] = neg_exp / (1 + neg_exp)

		return sigmoid_vals

	@staticmethod
	def inverse_sigmoid(z):
		return np.log(z / (1 - z))

	def logistic_plot(self, model, X_train, y_train, X_test, y_test, thre, eval='tr'):
		coef = model.coef_
		intercept = model.intercept_
		z_train = np.dot(X_train, coef.T) + intercept
		z_train = np.squeeze(z_train)
		z_test = np.dot(X_test, coef.T) + intercept
		z_test = np.squeeze(z_test)

		if max(z_train) < 1:
			x_curve = np.linspace(min(z_train), -min(z_train), 100)
		else:
			x_curve = np.linspace(min(z_train), max(z_train), 100)
		thre_x = self.inverse_sigmoid(thre)
		thre_y = [thre] * len(x_curve)
		y_curve = self.logistic_func(x_curve)

		if eval == 'tr':
			plt.scatter(z_train, y_train, label='Truth')
		else:
			plt.scatter(z_test, y_test, label='Truth')

		# Plot the fitted sigmoid curve
		plt.plot(x_curve, y_curve, label='Fitted Curve', color='r')
		# plot threshold lines
		plt.plot(x_curve, thre_y, linestyle='--', color='g')
		plt.plot([thre_x]*11, np.linspace(0.0, 1.0, 11), linestyle='--', color='g')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('Logistic Regression Plot')
		plt.legend()
		plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
		yticks = np.linspace(0.0, 1.0, 11)  # Specify the positions of the ticks
		plt.yticks(yticks)
		plt.show()

	@staticmethod
	def roc_auc_plot(y_true, y_pred_prob, plot='N'):
		# Compute the false positive rate (FPR), true positive rate (TPR),
		# and corresponding thresholds
		fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, drop_intermediate=True)

		# Calculate the AUC
		roc_auc = auc(fpr, tpr)

		if plot != 'N':
			# Plot the ROC curve
			plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
			plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate (FPR)')
			plt.ylabel('True Positive Rate (TPR)')
			plt.title('Receiver Operating Characteristic (ROC) Curve')
			plt.legend(loc="lower right")
			plt.show()
		return roc_auc

	@staticmethod
	def pr_curve_plot(y_true, y_pred_prob, plot='N'):
		precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

		# Calculate the average precision score
		avg_precision = average_precision_score(y_true, y_pred_prob)

		if plot != 'N':
			# Plot the precision-recall curve
			plt.plot(recall, precision, label='Precision-Recall curve (AP = {:.2f})'.format(avg_precision))
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.title('Precision-Recall Curve')
			plt.legend(loc="lower right")
			plt.show()
		return avg_precision
