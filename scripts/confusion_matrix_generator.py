import itertools

import numpy as np
import matplotlib.pyplot as plt

from tweets_processing_config import model_classes


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar().ax.tick_params(labelsize=20)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tick_params(labelsize=18)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.tight_layout()

    return plt


cm_logistic_regression = np.array([[9.90315747e-01, 7.06143448e-03, 9.07898719e-04, 1.41228690e-03, 3.02632906e-04],
                                   [1.04895105e-01, 8.91608392e-01, 0.00000000e+00, 1.74825175e-03, 1.74825175e-03],
                                   [2.36842105e-01, 0.00000000e+00, 7.00657895e-01, 6.08552632e-02, 1.64473684e-03],
                                   [1.67962675e-01, 1.55520995e-03, 2.17729393e-02, 8.08709176e-01, 0.00000000e+00],
                                   [3.41805434e-02, 2.62927257e-03, 0.00000000e+00, 3.50569676e-03, 9.59684487e-01]])

plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_logistic_regression, classes=model_classes, normalize=True,
                      title='Logistic Regression Confusion Matrix').show()

cm_decision_tree = np.array([[9.82447291e-01, 9.48249773e-03, 3.37940079e-03, 3.88378896e-03, 8.07021083e-04],
                             [1.03146853e-01, 8.91608392e-01, 0.00000000e+00, 2.33100233e-03, 2.91375291e-03],
                             [1.49671053e-01, 3.28947368e-03, 7.87828947e-01, 5.92105263e-02, 0.00000000e+00],
                             [1.27527216e-01, 6.22083981e-03, 3.88802488e-02, 8.27371695e-01, 0.00000000e+00],
                             [2.45398773e-02, 6.13496933e-03, 0.00000000e+00, 1.75284838e-03, 9.67572305e-01]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_decision_tree, classes=model_classes, normalize=True,
                      title='Decision Tree Confusion Matrix').show()

cm_mnb = np.array([[9.86885907e-01, 1.10965399e-02, 1.00877635e-04, 3.53071724e-04, 1.56360335e-03],
                   [7.75058275e-02, 8.62470862e-01, 0.00000000e+00, 5.82750583e-03, 5.41958042e-02],
                   [4.75328947e-01, 0.00000000e+00, 4.19407895e-01, 8.88157895e-02, 1.64473684e-02],
                   [4.01244168e-01, 0.00000000e+00, 1.86625194e-02, 5.80093313e-01, 0.00000000e+00],
                   [3.41805434e-02, 5.25854514e-03, 0.00000000e+00, 4.38212095e-03, 9.56178791e-01]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_mnb, classes=model_classes, normalize=True,
                      title='Multinomial Naive Bayes Confusion Matrix').show()

cm_gnb = np.array([[0.91153031, 0.02284878, 0.01835973, 0.04004842, 0.00721275],
                   [0.10431235, 0.76923077, 0.0034965, 0.00815851, 0.11480186],
                   [0.14309211, 0.00164474, 0.67105263, 0.13322368, 0.05098684],
                   [0.12130638, 0.00933126, 0.0311042, 0.79471229, 0.04354588],
                   [0.06134969, 0.03680982, 0.00438212, 0.0035057, 0.89395267]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_gnb, classes=model_classes, normalize=True,
                      title='Gaussian Naive Bayes Confusion Matrix').show()

cm_svm = np.array([[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0.]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_svm, classes=model_classes, normalize=True, title='SVM Confusion Matrix').show()

cm_rf = np.array([[9.88046000e-01, 8.57459901e-03, 1.46272571e-03, 1.66448098e-03, 2.52194089e-04],
                  [8.33333333e-02, 9.09673660e-01, 5.82750583e-04, 2.91375291e-03, 3.49650350e-03],
                  [1.64473684e-01, 0.00000000e+00, 7.76315789e-01, 5.75657895e-02, 1.64473684e-03],
                  [1.18195956e-01, 1.55520995e-03, 4.35458787e-02, 8.36702955e-01, 0.00000000e+00],
                  [2.62927257e-02, 9.64066608e-03, 1.75284838e-03, 2.62927257e-03, 9.59684487e-01]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_rf, classes=model_classes, normalize=True, title='Random Forest Confusion Matrix').show()

cm_knn = np.array([[9.81640270e-01, 1.17018057e-02, 3.32896197e-03, 2.77413497e-03, 5.54826995e-04],
                   [5.12820513e-02, 9.31818182e-01, 2.91375291e-03, 6.41025641e-03, 7.57575758e-03],
                   [1.16776316e-01, 1.64473684e-03, 7.99342105e-01, 6.90789474e-02, 1.31578947e-02],
                   [9.02021773e-02, 0.00000000e+00, 2.33281493e-02, 8.86469673e-01, 0.00000000e+00],
                   [2.01577564e-02, 1.31463628e-02, 8.76424189e-04, 4.38212095e-03, 9.61437336e-01]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_knn, classes=model_classes, normalize=True,
                      title='K-Nearest Neighbors Confusion Matrix').show()

cm_mlp = np.array([[9.86482397e-01, 7.91889438e-03, 2.37062443e-03, 2.77413497e-03, 4.53949359e-04],
                   [8.91608392e-02, 9.07342657e-01, 0.00000000e+00, 5.82750583e-04, 2.91375291e-03],
                   [1.06907895e-01, 1.64473684e-03, 8.65131579e-01, 2.63157895e-02, 0.00000000e+00],
                   [9.79782271e-02, 3.11041991e-03, 2.64385692e-02, 8.72472784e-01, 0.00000000e+00],
                   [1.75284838e-02, 3.50569676e-03, 0.00000000e+00, 0.00000000e+00, 9.78965819e-01]])
plt.figure(figsize=(13, 13))
plot_confusion_matrix(cm_mlp, classes=model_classes, normalize=True,
                      title='Multi-layer Perceptron Confusion Matrix').show()
