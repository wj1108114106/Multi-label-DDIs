
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import csv
import sys
import numpy as np


def ro_curve(fpr, tpr,roc_auc, figure_file, method_name):

    lw = 2
    plt.plot(fpr, tpr,
             lw=lw, label=method_name + ' (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title('ROC Curve on OGB-biokg', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(figure_file + ".png")
    return


def col_pic():
    for i in range(5):
        fpr=np.load('save_fpr_test_ogb'+"Fold" + str(i + 1)+'.npy')
        tpr=np.load('save_tpr_test_ogb'+"Fold" + str(i + 1)+'.npy')
        roc_auc=np.load('save_auc_test_ogb'+"Fold" + str(i + 1)+'.npy')

        ro_curve(fpr, tpr,roc_auc, "auc_val_1", "Fold" + str(i + 1))


def main():
    col_pic()


if __name__ == "__main__":
    main()