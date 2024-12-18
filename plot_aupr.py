import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import csv
import sys
import numpy as np
from scipy.interpolate import interp1d



def ro_curve(precision, recall, aupr, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''


    plt.plot(recall, precision, lw=2,
             label=method_name + ' (area = %0.5f)' %aupr)
    fontsize = 14
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    # plt.savefig(figure_file)
    return


def col_pic():
    mean_precision = []
    mean_recall = []
    mean_average_precision = []
    for i in range(5):
        precision = np.load('save_p_test_ogb' + "Fold" + str(i + 1) + '.npy')
        recall = np.load('save_r_test_ogb' + "Fold" + str(i + 1) + '.npy')
        aupr = np.load('save_aupr_test_ogb' + "Fold" + str(i + 1) + '.npy')
        mean_average_precision.append(aupr)
        mean_precision.append(precision)
        mean_recall.append(recall)


        ro_curve(precision, recall, aupr, "aupr_val_1", "Fold" + str(i + 1))
    # mean_precision=np.sum(mean_precision,axis=0)/5
    # mean_recall=np.sum(mean_recall,axis=0)/5

    # This is what the actual MAP score should be
    # mean_average_precision = sum(mean_average_precision) / len(mean_average_precision)

    # Code for plotting the mean average precision curve across folds
    # plt.plot(mean_recall, mean_precision, lw=2,
    #          label='Mean aupr' + ' (area = %0.5f)' % mean_average_precision)
    plt.title('AUPR curve on OGB-biokg')
    plt.savefig('aupr.png')


def main():
    col_pic()


if __name__ == "__main__":
    main()