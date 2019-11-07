
#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image

from sklearn.datasets.base import Bunch
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


import pydot, io
import time

#######################End imports###################################


####################Do not change anything below
#Load MNIST data. fetch_mldata will download the dataset and put it in a folder called mldata.
#Some things to be aware of:
#   The folder mldata will be created in the folder in which you started the notebook
#   So to make your life easy, always start IPython notebook from same folder.
#   Else the following code will keep downloading MNIST data
mnist_tf = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist_tf.load_data()
rows = np.concatenate((x_train, x_test))
labels = np.concatenate((y_train, y_test))
mnist = Bunch(data=rows, target=labels)

#The data is organized as follows:
#  Each row corresponds to an image
#  Each image has 28*28 pixels which is then linearized to a vector of size 784 (ie. 28*28)
# mnist.data gives the image information while mnist.target gives the number in the image
print("#Images = %d and #Pixel per image = %s" % (mnist.data.shape[0], mnist.data.shape[1]))

#Print first row of the dataset
img = mnist.data[0]
print("First image shows %d" % (mnist.target[0]))
print("The corresponding matrix version of image is \n" , img)
print("The image in grey shape is ")
plt.imshow(img.reshape(28, 28), cmap="Greys")
plt.show()
#First 60K images are for training and last 10K are for testing
all_train_data = mnist.data[:60000]
all_test_data = mnist.data[60000:]
all_train_labels = mnist.target[:60000]
all_test_labels = mnist.target[60000:]

#For the first task, we will be doing binary classification and focus  on two pairs of
#  numbers: 7 and 9 which are known to be hard to distinguish
#Get all the seven images
sevens_data = mnist.data[mnist.target==7]
#Get all the none images
nines_data = mnist.data[mnist.target==9]
#Merge them to create a new dataset
binary_class_data = np.vstack([sevens_data, nines_data])
binary_class_labels = np.hstack([np.repeat(7, sevens_data.shape[0]), np.repeat(9, nines_data.shape[0])])

#In order to make the experiments repeatable, we will seed the random number generator to a known value
# That way the results of the experiments will always be same
np.random.seed(1234)
#randomly shuffle the data
binary_class_data, binary_class_labels = shuffle(binary_class_data, binary_class_labels)
print("Shape of data and labels are :" , binary_class_data.shape, binary_class_labels.shape)

#There are approximately 14K images of 7 and 9.
#Let us take the first 5000 as training and remaining as test data
orig_binary_class_training_data = binary_class_data[:5000]
binary_class_training_labels = binary_class_labels[:5000]
orig_binary_class_testing_data = binary_class_data[5000:]
binary_class_testing_labels = binary_class_labels[5000:]

#The images are in grey scale where each number is between 0 to 255
# Now let us normalize them so that the values are between 0 and 1.
# This will be the only modification we will make to the image
binary_class_training_data = orig_binary_class_training_data / 255.0
binary_class_testing_data = orig_binary_class_testing_data / 255.0
scaled_training_data = all_train_data / 255.0
scaled_testing_data = all_test_data / 255.0

print(binary_class_training_data[0,:])

###########Make sure that you remember the variable names and their meaning
#binary_class_training_data, binary_class_training_labels: Normalized images of 7 and 9 and the correct labels for training
#binary_class_testing_data, binary_class_testing_labels : Normalized images of 7 and 9 and correct labels for testing
#orig_binary_class_training_data, orig_binary_class_testing_data: Unnormalized images of 7 and 9
#all_train_data, all_test_data: un normalized images of all digits
#all_train_labels, all_test_labels: labels for all digits
#scaled_training_data, scaled_testing_data: Normalized version of all_train_data, all_test_data for all digits


###Do not make any change below
def plot_dtree(model,fileName):
    #You would have to install a Python package pydot
    #You would also have to install graphviz for your system - see http://www.graphviz.org/Download..php
    #If you get any pydot error, see url
    # http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will
    dot_tree_data = io.StringIO()
    tree.export_graphviz(model, out_file = dot_tree_data)
    (dtree_graph,) = pydot.graph_from_dot_data(dot_tree_data.getvalue())
    dtree_graph.write_png(fileName)


flat_training_data = binary_class_training_data.reshape(len(binary_class_training_data), -1)
flat_test_data = binary_class_testing_data.reshape(len(binary_class_testing_data), -1)

# Exercise 1 (10 marks)
# Create a CART decision tree with splitting criterion as entropy
# Remember to set the random state to 1234
dtree_clf = DecisionTreeClassifier(criterion="entropy", random_state=1234)
dtree_clf.fit(flat_training_data, binary_class_training_labels)
dtree_preds = dtree_clf.predict(flat_test_data)

acc = dtree_clf.score(flat_test_data, binary_class_testing_labels)
print("Accuracy of decision tree is {:.2%}".format(acc))
plot_dtree(dtree_clf, "DecisionTree.png")

# Exercise 2 (10 marks)
# Create multinomial NB
nb_clf = MultinomialNB()
nb_clf.fit(flat_training_data, binary_class_training_labels)
nb_preds = nb_clf.predict(flat_test_data)

acc = nb_clf.score(flat_test_data, binary_class_testing_labels)
print("Accuracy of Multinomial NaiveBayes is {:.2%}".format(acc))

# Exercise 3 (10 marks)
# Create a model with default parameters. Remember to set random state to 1234
lg_clf = LogisticRegression(random_state=1234, solver="liblinear")
lg_clf.fit(flat_training_data, binary_class_training_labels)
lg_preds = lg_clf.predict(flat_test_data)

acc = lg_clf.score(flat_test_data, binary_class_testing_labels)
print("Accuracy of Logistic Regression is {:.2%}".format(acc))

# Exercise 4 (10 marks)
# Create a random forest classifier with Default parameters
rf_clf = RandomForestClassifier(n_estimators=10)
rf_clf.fit(flat_training_data, binary_class_training_labels)
rf_preds = rf_clf.predict(flat_test_data)

acc = rf_clf.score(flat_test_data, binary_class_testing_labels)
print("Accuracy of Random Forests is {:.2%}".format(acc))


# task t5a (5 marks)
# Print the classification report and confusion matrix for each of the models above
# Write code here
print("Classification report and confusion matrix for Decision Tree:")
print(metrics.classification_report(binary_class_testing_labels, dtree_preds))
print(metrics.confusion_matrix(binary_class_testing_labels, dtree_preds))

print("\nClassification report and confusion matrix for Multinomial NaiveBayes:")
print(metrics.classification_report(binary_class_testing_labels, nb_preds))
print(metrics.confusion_matrix(binary_class_testing_labels, nb_preds))

print("\nClassification report and confusion matrix for Logistic Regression:")
print(metrics.classification_report(binary_class_testing_labels, lg_preds))
print(metrics.confusion_matrix(binary_class_testing_labels, lg_preds))

print("\nClassification report and confusion matrix for Random Forests:")
print(metrics.classification_report(binary_class_testing_labels, rf_preds))
print(metrics.confusion_matrix(binary_class_testing_labels, rf_preds))

# task t5b (5 marks)
# Each of the model above has some probabilistic interpretation
# So sklearn allows you to get the probability values as part of classification
# Using this information, you can print roc_curve
# See http://nbviewer.ipython.org/github/datadave/GADS9-NYC-Spring2014-Lectures/blob/master/lessons/lesson09_decision_trees_random_forests/sklearn_decision_trees.ipynb
# Write code here
dtree_probs = dtree_clf.predict_proba(flat_test_data)
nb_probs = nb_clf.predict_proba(flat_test_data)
lg_probs = lg_clf.predict_proba(flat_test_data)
rf_probs = rf_clf.predict_proba(flat_test_data)

def plot_roc(probs, name):
    fpr_p, tpr_p, thresholds_p = metrics.roc_curve(binary_class_testing_labels, probs, pos_label=9)

    plt.plot(fpr_p, tpr_p)
    plt.title('ROC curve for {} Classifier'.format(name))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()

plot_roc(dtree_probs[:,1], "Decision Tree")
plot_roc(nb_probs[:,1], "Multinomial Naive Bayes")
plot_roc(lg_probs[:,1], "Logistic Regression")
plot_roc(rf_probs[:,1], "Random Forests")

# task t5c (5 marks)
# Print the AUC value for each of the models above
# Write code here
print("AUC value for Decision Tree classifier: {:.5f}".format(metrics.roc_auc_score(binary_class_testing_labels, dtree_probs[:,1])))
      
print("AUC value for Multinomial Naive Bayes classifier: {:.5f}".format(metrics.roc_auc_score(binary_class_testing_labels, nb_probs[:,1])))

print("AUC value for Logisitic Regression classifier: {:.5f}".format(metrics.roc_auc_score(binary_class_testing_labels, lg_probs[:,1])))

print("AUC value for Random Forests classifier: {:.5f}".format(metrics.roc_auc_score(binary_class_testing_labels, rf_probs[:,1])))


# task t5d (5 marks)
# Print the precision recall curve for each of the models above
# print the curve based on http://scikit-learn.org/stable/auto_examples/plot_precision_recall.html   
# Write code here
from sklearn.utils.fixes import signature

def plot_pr(probs, name):
    precision, recall, _ = metrics.precision_recall_curve(binary_class_testing_labels, probs, pos_label=9)
    step_kwargs = ({'step': 'post'}
    if 'step' in signature(plt.fill_between).parameters
    else {})
    plt.step(recall, precision, color='b', alpha=0.2,
    where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall curve for {} Classifier'.format(name))
    plt.show()

plot_pr(dtree_probs[:,1], "Decision Tree")
plot_pr(nb_probs[:,1], "Multinomial Naive Bayes")
plot_pr(lg_probs[:,1], "Logistic Regression")
plot_pr(rf_probs[:,1], "Random Forests")



###Do not make any change below
all_scaled_data = binary_class_data / 255.0
all_scaled_target = binary_class_labels

# Exercise 6 (15 marks)
# Tuning Random Forest for MNIST
tuned_parameters = [{'max_features': ['sqrt', 'log2'], 'n_estimators': [1000, 1500]}] 

# Write code here
grid_rf = GridSearchCV(estimator=rf_clf, cv=3, param_grid=tuned_parameters, verbose=2, n_jobs=-1)
flat_scaled_data = all_scaled_data.reshape(len(all_scaled_data), -1)
grid_rf.fit(flat_scaled_data, all_scaled_target)

# print the details of the best model and its accuracy
# Write code here
print("Best parameters:\n{}".format(grid_rf.best_params_))
print("Accuracy of best model: {:.2%}".format(grid_rf.best_score_))
