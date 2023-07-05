"""
File: new_kappa_tuned_hmeq.py
Name:Alex
----------------------------------------
TODO:
Use the new kappa method (sim tune) to perform classification
Dataset: Home Equity dataset
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from my_logistic_regression import MyLogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, \
    precision_recall_fscore_support, cohen_kappa_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import sys
import statistics as stat
from matplotlib import pyplot as plt
import utils

# hyps = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3]
hyps = [10, 100, 300]    # n estimator
# thresholds = np.arange(0, 0.55, 0.05)
thresholds = [0.5]

# CLASSIFIER = 'logistic_regression'
# CLASSIFIER = 'mylr'
CLASSIFIER = 'RF'
ran_states = np.arange(10)  # number of train-test splits
SOLVER = 'liblinear'
HYPER_METRIC = 'acc'
PENALTY = 'l1'
MAX_ITER = 1000

# DATA_X = 'dataset/pickles/hab_X.pickle'
# DATA_Y = 'dataset/pickles/hab_Y.pickle'
DATA_X = 'dataset/pickles/hmeq_X.pickle'
DATA_Y = 'dataset/pickles/hmeq_Y.pickle'

LOG_FILENAME = 'log/hmeq_0609_05thre_acc_rf.txt'
CSV_FILENAME = 'results/hmeq_0609_05thre_acc_rf.csv'

tune = utils.Tuning(PENALTY, SOLVER, MAX_ITER, CLASSIFIER, HYPER_METRIC)
plot = utils.Plot()


def main():
    sys.stdout = open(LOG_FILENAME, "w")

    with open(DATA_X, 'rb') as f:
        X = pickle.load(f)
    with open(DATA_Y, 'rb') as f:
        y = pickle.load(f)

    metrics = {'kappas': [],'maj_kappas': [],  'f1_stars': [], 'baccs': [], 'f1s': [], 'precs': [], 'recalls': [], 'specs': [], 'aucs': [],
               'aprs': [], 'accs': [], 'opt_thres': [], 'opt_hyps': []}

    for ran_state in tqdm(ran_states):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=ran_state)
        best_hyp, best_threshold = tune.cross_val(X_train, y_train, hyps, thresholds)

        if CLASSIFIER == 'logistic_regression':
            final_clf = LogisticRegression(penalty=PENALTY, C=best_hyp, solver=SOLVER, max_iter=MAX_ITER).fit(X_train,
                                                                                                           y_train)
        elif CLASSIFIER == 'RF':
            final_clf = RandomForestClassifier(n_estimators=best_hyp).fit(X_train, y_train)

        elif CLASSIFIER == 'mylr':
            final_clf = MyLogisticRegression(penalty=PENALTY, C=best_hyp, max_iter=MAX_ITER).fit(X_train, y_train)

        # plot.logistic_plot(final_clf, X_train, y_train, X_test, y_test, best_threshold, eval='test')
        y_pred = tune.tuned_prediction(final_clf, X_test, best_threshold)
        print(confusion_matrix(y_test, y_pred, labels=[1, 0]))
        print(classification_report(y_test, y_pred))
        print(f'kappa: {cohen_kappa_score(y_pred, y_test)}')
        print(f'f1 stat: {tune.new_f1_score(y_pred, y_test)}')
        auc = utils.Plot.roc_auc_plot(y_test, final_clf.predict_proba(X_test)[:, 1])
        apr = utils.Plot.pr_curve_plot(y_test, final_clf.predict_proba(X_test)[:, 1])

        # prediction performance #
        print(
            f'Final model: {final_clf}, best_lamb = {best_hyp}, best_threshold = {best_threshold}, 'f'random_state = {ran_state}')
        # print(f'Final model: {final_clf}, best_n_estimator = {best_lamb}, best_threshold = {best_threshold}, random_state = {ran_state}')
        # print(f'Final model coefficient: {final_clf.coef_}, and intercept: {final_clf.intercept_}\n')
        # print(f'Final model features: {final_clf.feature_names_in_}')

        # metrics #
        tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred))
        spec = tn / (tn + fp)
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        f1_star = tune.new_f1_score(y_test, y_pred)

        metrics['accs'].append(round(acc, 4))
        metrics['baccs'].append(round(bacc, 4))
        metrics['f1s'].append(round(f1[1], 4))
        metrics['f1_stars'].append(round(f1_star, 4))
        metrics['precs'].append(round(precision[1], 4))
        metrics['recalls'].append(round(recall[1], 4))
        metrics['specs'].append(round(spec, 4))
        metrics['opt_thres'].append(best_threshold)
        metrics['opt_hyps'].append(best_hyp)
        metrics['kappas'].append(round(cohen_kappa_score(y_test, y_pred), 4))
        metrics['maj_kappas'].append(round(utils.Kappa.maj_kappa(y_test, y_pred), 4))
        metrics['aucs'].append(auc)
        metrics['aprs'].append(apr)

    # averaging performance
    print(f"mean accuracy: {np.mean(metrics['accs'])}, standard error: {tune.se(metrics['accs'])}")
    print(f"mean balanced accuracy: {np.mean(metrics['baccs'])}, standard error: {tune.se(metrics['baccs'])}")
    print(f"mean f1 score: {np.mean(metrics['f1s'])}, standard error: {tune.se(metrics['f1s'])}")
    print(f"mean f1 star score: {np.mean(metrics['f1_stars'])}, standard error: {tune.se(metrics['f1_stars'])}")
    print(f"mean kappa score: {np.mean(metrics['kappas'])}, standard error: {tune.se(metrics['kappas'])}")
    print(f"mean maj kappa score: {np.mean(metrics['maj_kappas'])}, standard error: {tune.se(metrics['maj_kappas'])}")
    print(f"mean precision: {np.mean(metrics['precs'])}, standard error: {tune.se(metrics['precs'])}")
    print(f"mean recall: {np.mean(metrics['recalls'])}, standard error: {tune.se(metrics['recalls'])}")
    print(f"mean specificity: {np.mean(metrics['specs'])}, standard error: {tune.se(metrics['specs'])}")
    print(f"mean auc: {np.mean(metrics['aucs'])}, standard error: {tune.se(metrics['aucs'])}")
    print(f"mean apr: {np.mean(metrics['aprs'])}, standard error: {tune.se(metrics['aprs'])}")

    # save output to
    results_df = pd.DataFrame()
    for key, value in metrics.items():
        results_df[key] = value

    # summary stats
    for key, value in metrics.items():
        results_df.at[len(ran_states), key] = round(float(np.mean(value)), 4)
        results_df.at[len(ran_states) + 1, key] = round(tune.se(value), 4)

    results_df.to_csv(CSV_FILENAME, index=False)
    sys.stdout.close()


if __name__ == '__main__':
    main()
