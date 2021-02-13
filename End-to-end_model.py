# -*- coding: utf-8 -*-
"""
This file contains all steps from reading in the data to the final model.

"""

# import the needed libraries
import pandas as pd
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier

#Functions

def plot_precision_vs_recall(precision_scores, recall_scores, figsize=(8,4)):
    """
    Plots a simple precision-recall curve.
    
    Parameters
    ----------
    precision_scores : precision scores as given by the precision_recall_curve function
    recall_scores : recall scores as given by the precision_recall_curve function

    Returns
    -------
    Plot of the precision-recall curve.

    """
    plt.plot(recall_scores, precision_scores, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.show()

def evaluate_classification(y, y_pred):
    """
    Evaluates the predictions of a classifier with accuracy, precision and recall
    score, confusion matrix as well as a precision-recall plot.
    
    Parameters
    ----------
    y : pd.Series or array of target values from the training, validation or test set
    y_pred : pd.Series or array of target values as predicted by the classifier

    Returns
    -------
    Accuracy score
    Precision score
    Recall score
    Confusion Matrix
    Plot of the precision-recall curve.
    """
    print("Accuracy: ", accuracy_score(y, y_pred))
    print("Precision: ", precision_score(y,y_pred))
    print("Recall: ", recall_score(y,y_pred))
    print("   TN       FP")
    print(confusion_matrix(y,y_pred)) 
    print("   FN       TP")
    precisions, recalls, thresholds = precision_recall_curve(y,y_pred)
    return plot_precision_vs_recall(precisions,recalls, figsize=(10,5)) #plot precision recall curve
 

#Load the data
path = "C:/Users/ms101/OneDrive/datasets"
data = pd.read_csv(path + "/creditcard.csv")


#Clean the data and reduce it to the most important features
data.drop_duplicates(inplace = True)
data.reset_index(inplace = True, drop = True)
assert data.shape == (283726,31)#check if duplicates are removed correctly
data_red = data[["V17","V14","V12","V10","V16","V3","V7","V11","Class"]].copy()

#split into X and y
X = data_red.drop("Class", axis = 1)
y = data_red["Class"]
assert X.shape == (283726,8) #check dimensions for X
assert y.shape == (283726,) #check dimensions for y

#Stratified Split into Training and Test set(20% of full data)
strat_split = StratifiedShuffleSplit(test_size = 0.2, random_state = 13)
for train_index, test_index in strat_split.split(X,y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    


#define the model
forest_clf = RandomForestClassifier(bootstrap=False, max_depth=82, min_samples_split=10,
                       n_estimators=233, n_jobs=-1, random_state=13)

#fitting the model
forest_clf.fit(X_train,y_train)
#getting predictions for the test set
y_pred_forest = forest_clf.predict(X_test)
#evaluating the performance on the test set
evaluate_classification(y_test, y_pred_forest)

