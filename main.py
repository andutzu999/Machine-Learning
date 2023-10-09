import numpy as np
from sktime.datasets import *
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectPercentile, chi2, VarianceThreshold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


def read_datasets(test_filepath, train_filepath):
    test_absolute_path = os.path.abspath(test_filepath)
    train_absolute_path = os.path.abspath(train_filepath)
    testDF_x, testDF_y = load_from_tsfile_to_dataframe(test_absolute_path, replace_missing_vals_with='NaN')
    trainDF_x, trainDF_y = load_from_tsfile_to_dataframe(train_absolute_path, replace_missing_vals_with='NaN')
    defrag_testDF_x, defrag_testDF_y = testDF_x.copy(), testDF_y.copy()  # increase performance
    defrag_trainDF_x, defrag_trainDF_y = trainDF_x.copy(), trainDF_y.copy()
    return defrag_testDF_x, defrag_testDF_y, defrag_trainDF_x, defrag_trainDF_y

# vector de criterii
def add_criterias(df):
    criteria = [np.max, np.min, np.mean, np.median, np.std]
    # create a dataframe
    data = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    # apply criteria on every columns
    for col, _ in df.iteritems():
        for i in range(len(criteria)):
            data[i][col] = df[col].apply(criteria[i])
    return data


def select_cols(df_x, df_y, percentile, dataset_selection):
    if dataset_selection == "uwave":
        # selecteaza coloanele care au gradul de varianta mai mare decat limita setat; default 0, selecteaza toate feature-urile cu valori diferite
        selection = VarianceThreshold()
        #selection = SelectPercentile(chi2, percentile=percentile)
    else:
        # selecteaza cele mai bune feature determinate cu ajutorul functiei de evaluare chi2
        selection = SelectPercentile(chi2, percentile=percentile)
    selection.fit(df_x, df_y)
    return selection.get_support(indices=True)


def select_df(train_df_x, train_df_y, test_df_x, percentile, dataset_selection):
    criteria = [np.max, np.min, np.mean, np.median, np.std]
    selected = []
    train_df_crit_x = add_criterias(train_df_x.copy())
    print("Train_df_crit_x")
    print(train_df_crit_x)
    for idx in range(len(criteria)):
        tmp = select_cols(train_df_crit_x[idx], train_df_y, percentile, dataset_selection)
        selected.append(tmp)
    # toate col distincte
    selected = np.unique(selected)
    print(selected)
    print(len(selected))
    train_df_x_sel = train_df_x.copy().iloc[:, selected]
    test_df_x_sel = test_df_x.copy().iloc[:, selected]
    # add_criterias for initial train and test
    df_train = add_criterias(train_df_x_sel)
    df_test = add_criterias(test_df_x_sel)
    return df_train, df_test, selected


def run_algorithm(train_df_x, train_df_y, test_df_x, test_df_y, percentile, dataset_selection):
    names = ['max', 'min', 'mean', 'median', 'std']
    #select_df
    df_train, df_test, selected = select_df(train_df_x, train_df_y, test_df_x, percentile, dataset_selection)
    print("df_train: {}".format(df_train))
    print("df_test: {}".format(df_test))
    print("df_selected: {}".format(selected))
    keys = [list(map(lambda x: str(x) + names[i], selected)) for i in range(5)]
    print("keys: {}".format(keys))
    for i in range(5):
        df_train[i].columns = keys[i]
        df_test[i].columns = keys[i]
    train_final_x = pd.concat(df_train, axis=1)
    test_final_x = pd.concat(df_test, axis=1)
    print("train_final_x: {}".format(train_final_x))
    print("test_final_x: {}".format(test_final_x))
    random_forest(test_df_y, test_final_x, train_df_y, train_final_x, dataset_selection)
    svm_alg(test_df_y, test_final_x, train_df_y, train_final_x, dataset_selection)
    gradient_boosted_trees(test_df_y, test_final_x, train_df_y, train_final_x, dataset_selection)


def random_forest(test_df_y, test_final_x, train_df_y, train_final_x, dataset_selection):
    # posibilitatile de selectie pentru hiperparametrii
    parameters = {
        'n_estimators': [50, 100, 150, 200, 250],  # numarul de estimatori folositi
        'max_features': ['auto', 'sqrt', 'log2'],  # algoritmul folosit pentru evaluare
        'max_depth': [1, 2, 5, 9, 10]  # adancimea arborilor de clasificare
    }
    # alegerea clasificatorului
    rfc = RandomForestClassifier()
    print("rfc: {}".format(rfc))
    # cauta cea mai buna combinatie (cea mai mare precizie) de hiperparametrii pentru clasificatorul folosit
    clf = GridSearchCV(rfc, parameters)
    clf.fit(train_final_x, train_df_y)
    predicted = clf.predict(test_final_x)
    print("predicted: {}".format(predicted))
    # din print-urile astea 2 facem tabelele
    print(clf.best_params_)
    print(classification_report(test_df_y, predicted))
    ConfusionMatrixDisplay.from_predictions(test_df_y, predicted)
    plt.show()


def svm_alg(test_df_y, test_final_x, train_df_y, train_final_x, dataset_selection):
    if dataset_selection == "pems":
        parameters = {
            'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),  # tipul de kernel
            'C': [0.1, 1, 10, 50, 100, 500, 1000],  # parametrul de regularizare
            'gamma': [0.1, 0.5, 1, 5, 10, 100, 1000]  # factorul de scalare
        }
    elif dataset_selection == "uwave":
        parameters = {
            'kernel': ('linear', 'rbf', 'sigmoid'),
            'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 50, 100, 500, 1000, 5000],
            'gamma': [0.1, 0.5, 1, 5, 10, 100, 1000, 5000]
        }
    else:
        parameters = {
        }
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(train_final_x, train_df_y)
    predicted = clf.predict(test_final_x)
    print(clf.best_params_)
    print(classification_report(test_df_y, predicted))
    ConfusionMatrixDisplay.from_predictions(test_df_y, predicted)
    plt.show()


def gradient_boosted_trees(test_df_y, test_final_x, train_df_y, train_final_x, dataset_selection):
    parameters = {
        'n_estimators': [50, 100, 150, 200, 250],  # numarul de estimatori folositi
        'max_depth': [1, 2, 5, 9, 10],  # adancimea maxima a arborilor
    }
    model = GradientBoostingClassifier()
    clf = GridSearchCV(model, parameters, verbose=2)
    clf.fit(train_final_x, train_df_y)
    predicted = clf.predict(test_final_x)
    print(clf.best_params_)
    print(classification_report(test_df_y, predicted))
    ConfusionMatrixDisplay.from_predictions(test_df_y, predicted)
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    start_time = time.time_ns()
    #pems_testDF_x, pems_testDF_y, pems_trainDF_x, pems_trainDF_y = read_datasets("PEMS/PEMS-SF_TEST.ts",
                                                                                 #"PEMS/PEMS-SF_TRAIN.ts")

    # print(pems_testDF_x)
    # print(pems_testDF_y)
    # print(pems_trainDF_x)
    # print(pems_trainDF_y)
    uwave_testDF_x, uwave_testDF_y, uwave_trainDF_x, uwave_trainDF_y = read_datasets(
        "UWaveGesture/UWaveGestureLibrary_TEST.ts", "UWaveGesture/UWaveGestureLibrary_TRAIN.ts")
    #run_algorithm(pems_trainDF_x, pems_trainDF_y, pems_testDF_x, pems_testDF_y, 1, "pems")
    run_algorithm(uwave_trainDF_x, uwave_trainDF_y, uwave_testDF_x, uwave_testDF_y, 1, "uwave")
    end_time = time.time_ns()
    print("Elapsed time: {} ms".format((end_time - start_time) / 1000000))
