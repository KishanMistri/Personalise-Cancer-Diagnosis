
#!/usr/bin/env python
# coding: utf-8

# <h1 align=Center>Personalized cancer diagnosis</h1>

# In[1]:


import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import re
import random
import joblib
import math
from collections import Counter, defaultdict
from pprint import pprint
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Importing utility functions from parent directory. Can't keep in pages as it creates menu in sidebar here.
import sys 
sys.path.insert(0, '../')
from utilities import *

# Setting seed for reproducibility
random_state = 42
np.random.RandomState(random_state)
np.random.seed(random_state)
random.seed(random_state)


# In[2]:


# Noting the time of start of execution
start_time = time.time()


# In[3]:


# Environment Variables & Data imports
TRAIN_DATA_DIR = "../train_data/" 
TEST_DATA_DIR = "../test_data/"
PROCESSED_DATA_DIR = "../preprocessed/"
MODELS_DIR = "../models/"

try:
    train_df = joblib.load(PROCESSED_DATA_DIR+"train_df.pkl")
    cv_df = joblib.load(PROCESSED_DATA_DIR+"cv_df.pkl")
    test_df = joblib.load(PROCESSED_DATA_DIR+"test_df.pkl")
    y_train = joblib.load(PROCESSED_DATA_DIR+"y_train.pkl")
    y_cv = joblib.load(PROCESSED_DATA_DIR+"y_cv.pkl")
    y_test = joblib.load(PROCESSED_DATA_DIR+"y_test.pkl")
    train_x_onehotCoding = joblib.load(PROCESSED_DATA_DIR+"train_x_onehotCoding.pkl")
    test_x_onehotCoding = joblib.load(PROCESSED_DATA_DIR+"test_x_onehotCoding.pkl")
    cv_x_onehotCoding = joblib.load(PROCESSED_DATA_DIR+"cv_x_onehotCoding.pkl")
    train_x_responseCoding = joblib.load(PROCESSED_DATA_DIR+"train_x_responseCoding.pkl")
    test_x_responseCoding = joblib.load(PROCESSED_DATA_DIR+"test_x_responseCoding.pkl")
    cv_x_responseCoding = joblib.load(PROCESSED_DATA_DIR+"cv_x_responseCoding.pkl")
    train_y = joblib.load(PROCESSED_DATA_DIR+"train_y.pkl")
    test_y = joblib.load(PROCESSED_DATA_DIR+"test_y.pkl")
    cv_y = joblib.load(PROCESSED_DATA_DIR+"cv_y.pkl")
    
except FileNotFoundError:
    print("Please check {0} path to validate if required file is present".format(PROCESSED_DATA_DIR))

all_model_metadata={}


# In[4]:


# Utility
def get_impfeature_names(indices, text, gene, var, no_features, train_df):
    """
    Usage: get_impfeature_names(indices, text, gene, var, no_features)
    Returns: None
    >> for the given indices, we will print the name of the features and we will check whether the feature present in the test point text or not
    """
    gene_count_vec = CountVectorizer()
    var_count_vec = CountVectorizer()
    text_count_vec = CountVectorizer(min_df=3)
    
    gene_vec = gene_count_vec.fit(train_df['Gene'])
    var_vec  = var_count_vec.fit(train_df['Variation'])
    text_vec = text_count_vec.fit(train_df['Text'])
    
    fea1_len = len(gene_vec.get_feature_names())
    fea2_len = len(var_count_vec.get_feature_names())
    
    word_present = 0
    feature_presence_map = {"Gene": {},"Variantion": {},"Text": {}}
    for i,v in enumerate(indices):
        if (v < fea1_len):
            word = gene_vec.get_feature_names()[v]
            yes_no = True if word == gene else False
            if yes_no:
                word_present += 1
                feature_presence_map['Gene'][i] = word
        elif (v < fea1_len+fea2_len):
            word = var_vec.get_feature_names()[v-(fea1_len)]
            yes_no = True if word == var else False
            if yes_no:
                word_present += 1
                feature_presence_map['Variantion'][i] = word
        else:
            word = text_vec.get_feature_names()[v-(fea1_len+fea2_len)]
            yes_no = True if word in text.split() else False
            if yes_no:
                word_present += 1
                feature_presence_map['Text'][i] = word
    
    pprint(">>> Out of the top {0} features {1} are present in query point".format(no_features,word_present))
    if word_present:
        pprint("Features present in Top {}".format(no_features),compact=True)
        pprint(feature_presence_map,compact=True)
        pprint("Numuber of features from Gene features: {0} | Variation: {1} | Text: {2}".               format(len(feature_presence_map['Gene']),len(feature_presence_map['Variantion']),len(feature_presence_map['Text'])))
    print("-"*50)
    # st.write(feature_presence_map)
    # st.caption("Out of the top {0} features {1} are present in query point".format(no_features,word_present))


# <h3>1. Naive Bayes</h3>

# <h5>1.1. Hyper parameter tuning</h5>

# In[5]:


multinomial_dict = {}
alpha = [10**power for power in range(-5,4)]
cv_log_error_array = []
alpha_dict = {}

for i in alpha:
    clf = MultinomialNB(alpha=i)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    alpha_dict[i] = log_loss(cv_y, sig_clf_probs)
multinomial_dict["Tuning"] = alpha_dict

fig, ax = plt.subplots()
ax.plot(np.log10(alpha), cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error measure")
plt.show()

best_alpha = alpha[np.argmin(cv_log_error_array)]

clf = MultinomialNB(alpha=best_alpha)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_onehotCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

multinomial_dict["LogLoss"] = loss_dict 
pprint(multinomial_dict,compact=True)


# <h5>1.2. Testing the model with best hyper paramters</h5>

# In[6]:


# Refer: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
best_parameters = {}
best_parameters['Alpha'] = best_alpha

clf = MultinomialNB(alpha=best_alpha)

# # Keep an eye for train_x & test_x for perticular problem
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding,                                                                                            train_y,                                                                                            test_x_onehotCoding,                                                                                            test_y,                                                                                            clf)
plot_confusion_matrix(test_y, pred_y)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
multinomial_dict["best_parameters"] = best_parameters


# <h5>1.1.4. Feature Importance</h5>

# In[7]:


# Final Model Training with best params
clf = MultinomialNB(alpha=best_alpha)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

# Get classification points separately
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
multinomial_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h6>Correctly classified point</h6>

# In[8]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.feature_log_prob_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# <h6>Incorrectly classified point</h6>

# In[9]:


test_point_index = random.choice(misclassified_point_idxs)
no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class :", test_y[test_point_index])
indices = np.argsort(-1*abs(clf.feature_log_prob_))[predicted_cls-1][:,:no_feature]
print("-"*50)
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                     train_df)


# In[10]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_MultinomialNB.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_MultinomialNB.pkl")

multinomial_dict["Models"] = {"base_classifier" : MODELS_DIR[1:]+"base_MultinomialNB.pkl",                              "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_MultinomialNB.pkl"}
all_model_metadata["MultinomialNB"] = multinomial_dict


# In[11]:


pprint(multinomial_dict,compact=True)


# <h3>2. K-Nearest Neighbour Classification</h3>

# <h5>2.1. Hyper parameter tuning</h5>

# In[12]:


knn_dict = {}
neighbors = [5, 11, 15, 21, 31, 41, 51, 99]
cv_log_error_array = []
hyperparam_dict = {}

for i in neighbors:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(train_x_responseCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_responseCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    hyperparam_dict[i] = log_loss(cv_y, sig_clf_probs)
knn_dict["Tuning"] = hyperparam_dict

fig, ax = plt.subplots()
ax.plot(neighbors, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((neighbors[i],str(txt)), (neighbors[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each neighbors")
plt.xlabel("k neighbors")
plt.ylabel("Error measure")
plt.show()


best_neighbors = neighbors[np.argmin(cv_log_error_array)]

clf = KNeighborsClassifier(n_neighbors=best_neighbors)
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)


loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_responseCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_responseCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_responseCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

knn_dict["LogLoss"] = loss_dict 
pprint(knn_dict,compact=True)


# <h5>2.2. Testing the model with best hyper paramters</h5>

# In[13]:


# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
best_parameters = {}
best_parameters['n_neighbors'] = best_neighbors

clf = KNeighborsClassifier(n_neighbors=best_neighbors)

# Keep an eye for train_x & test_x for perticular problem
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_responseCoding,                                                                                            train_y,                                                                                            test_x_responseCoding,                                                                                            test_y,                                                                                            clf)
plot_confusion_matrix(test_y, pred_y)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
knn_dict["best_parameters"] = best_parameters


# <h5>2.3.Classification details</h5>

# In[14]:


# Model fitting
clf = KNeighborsClassifier(n_neighbors=best_neighbors)
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_responseCoding)
total_test_points = test_x_responseCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
knn_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h6>Sampling correct point</h6>

# In[15]:


test_point_index = random.choice(correctly_classified_point_idxs)
predicted_cls = sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))
neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), best_neighbors)

print("Optimal value for nearest neighbors knn is {0} \nThe nearest neighbours of the test points belongs to classes \n>>> {1}".      format(best_neighbors,Counter(train_y[neighbors[1][0]])))
print("Predicted Class : {0}".format(predicted_cls[0]))
print("Actual Class : {0}".format(test_y[test_point_index]))


# <h5>Sampling incorrect point </h5>

# In[16]:


test_point_index = random.choice(misclassified_point_idxs)

predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), best_neighbors)
print("Optimal value for nearest neighbors knn is {0} \nThe nearest neighbours of the test points belongs to classes \n>>> {1}".      format(best_neighbors,Counter(train_y[neighbors[1][0]])))
print("Predicted Class : {0}".format(predicted_cls[0]))
print("Actual Class : {0}".format(test_y[test_point_index]))


# In[17]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_KNN.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_KNN.pkl")

knn_dict["Models"] = {"base_classifier"       : MODELS_DIR[1:]+"base_KNN.pkl",                      "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_KNN.pkl"}

all_model_metadata["KNN"] = knn_dict


# In[18]:


pprint(knn_dict,compact=True)


# <h3>3. Logistic Regression</h3>

# <h5>3.1. With Class balancing</h5>

# <h6>3.1.1. Hyper paramter tuning</h6>

# In[19]:


# Logistic regression = SGD with LOGLOSS
balanced_logreg_dict = {}
alpha = [10 ** x for x in range(-6, 3)]
cv_log_error_array = []
hyperparam_dict = {}
# Refer: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# Weights associated with classes. If not given, all classes are supposed to have weight one.
# The “balanced” mode uses the values of y to automatically adjust weights inversely proportional
# to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

for i in alpha:
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log_loss', random_state=random_state)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    hyperparam_dict[i] = log_loss(cv_y, sig_clf_probs)
balanced_logreg_dict["Tuning"] = hyperparam_dict

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error measure")
plt.show()

best_alpha = alpha[np.argmin(cv_log_error_array)]

clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='log_loss', random_state=random_state)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_onehotCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

balanced_logreg_dict["LogLoss"] = loss_dict 
pprint(balanced_logreg_dict,compact=True)


# <h6>3.1.2. Testing the model with best hyper paramters</h6>

# In[20]:


best_parameters = {}
best_parameters['Alpha'] = best_alpha

# Refer SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='log_loss', random_state=random_state)
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding,                                                                                            train_y,                                                                                            test_x_onehotCoding,                                                                                            test_y,                                                                                            clf)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
balanced_logreg_dict["best_parameters"] = best_parameters

plot_confusion_matrix(test_y, pred_y)


# <h6>3.1.3. Feature Importance</h6>

# In[21]:


# Model fitting
clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='log_loss', random_state=random_state)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
balanced_logreg_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h6>Correctly Classified point</h6>

# In[22]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# <h6>Mis-Classified point</h6>

# In[23]:


test_point_index = random.choice(misclassified_point_idxs)
no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# In[24]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_LOGREG_BALANCED.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_LOGREG_BALANCED.pkl")

balanced_logreg_dict["Models"] = {"base_classifier"       : MODELS_DIR[1:]+"base_LOGREG_BALANCED.pkl",                      "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_LOGREG_BALANCED.pkl"}

all_model_metadata["LOGREG_BALANCED"] = balanced_logreg_dict


# In[25]:


pprint(balanced_logreg_dict,compact=True)


# <h5>3.2. Without Class balancing</h5>

# <h6>3.2.1. Hyper paramter tuning</h6>

# In[26]:


unbalanced_logreg_dict = {}
alpha = [10 ** x for x in range(-6, 3)]
cv_log_error_array = []
hyperparam_dict = {}
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log_loss', random_state=random_state)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    hyperparam_dict[i] = log_loss(cv_y, sig_clf_probs)
unbalanced_logreg_dict["Tuning"] = hyperparam_dict

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error measure")
plt.show()

best_alpha = alpha[np.argmin(cv_log_error_array)]

clf = SGDClassifier(alpha=best_alpha, penalty='l2', loss='log', random_state=random_state)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_onehotCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

unbalanced_logreg_dict["LogLoss"] = loss_dict 
pprint(unbalanced_logreg_dict,compact=True)


# <h6>3.2.2. Testing model with best hyper parameters</h6>

# In[27]:


best_parameters = {}
best_parameters['Alpha'] = best_alpha

clf = SGDClassifier(alpha=best_alpha, penalty='l2', loss='log_loss', random_state=random_state)
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding,                                                                                            train_y,                                                                                            test_x_onehotCoding,                                                                                            test_y,                                                                                            clf)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
unbalanced_logreg_dict["best_parameters"] = best_parameters

plot_confusion_matrix(test_y, pred_y)


# <h6>3.1.3. Feature Importance</h6>

# In[28]:


correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
unbalanced_logreg_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h6>Correctly Classified point</h6>

# In[29]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# <h6>Inorrectly Classified point</h6>

# In[30]:


test_point_index = random.choice(misclassified_point_idxs)
no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# In[31]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_LOGREG_UNBALANCED.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_LOGREG_UNBALANCED.pkl")

unbalanced_logreg_dict["Models"] = {"base_classifier"       : MODELS_DIR[1:]+"base_LOGREG_UNBALANCED.pkl",                                    "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_LOGREG_UNBALANCED.pkl"}

all_model_metadata["LOGREG_UNBALANCED"] = unbalanced_logreg_dict


# In[32]:


pprint(unbalanced_logreg_dict,compact=True)


# <h3>4. Support Vector Machines</h3>

# <h5>4.1. Hyper paramter tuning</h5>

# In[33]:


# REFER: SVM with linear kernals here http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# SVM = SGD with hinge loss [not always]
svm_dict = {}
alpha = [10 ** x for x in range(-5, 3)]
cv_log_error_array = []
hyperparam_dict = {}
for i in alpha:
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=random_state)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    hyperparam_dict[i] = log_loss(cv_y, sig_clf_probs)
svm_dict["Tuning"] = hyperparam_dict

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error measure")
plt.show()


best_alpha = alpha[np.argmin(cv_log_error_array)]

clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='hinge', random_state=random_state)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_onehotCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

svm_dict["LogLoss"] = loss_dict 
pprint(svm_dict,compact=True)


# <h5>4.2. Testing model with best hyper parameters</h5>

# In[34]:


best_parameters = {}
best_parameters['Alpha'] = best_alpha

clf = SGDClassifier(alpha=best_alpha, penalty='l2', loss='hinge', random_state=random_state,class_weight='balanced')
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding,                                                                                            train_y,                                                                                            test_x_onehotCoding,                                                                                            test_y,                                                                                            clf)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
svm_dict["best_parameters"] = best_parameters

plot_confusion_matrix(test_y, pred_y)


# <h6>4.3. Feature Importance</h6>

# In[35]:


# Model fitting
clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='hinge', random_state=random_state)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
svm_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h5>4.3.1. For Correctly classified point</h5>

# In[36]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# <h5>4.3.2. For Mis-classified point</h5>

# In[37]:


test_point_index = random.choice(misclassified_point_idxs)
no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
get_impfeature_names(indices[0],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# In[38]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_SVM.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_SVM.pkl")

svm_dict["Models"] = {"base_classifier"       : MODELS_DIR[1:]+"base_SVM.pkl",                      "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_SVM.pkl"}

all_model_metadata["SVM"] = svm_dict


# In[39]:


pprint(svm_dict,compact=True)


# <h3>5. Random Forest Classifier</h3>

# <h4>With One hot Encoding</h4>

# In[40]:


# Issues with pandas version 0.23 with Parellelism
# https://github.com/scikit-learn/scikit-learn/issues/15851#issuecomment-564015275
# Setting n_jobs=1 will fix it for now.


# <h5>5.1. Hyper paramter tuning</h5>

# In[41]:


rf_ohe_dict = {}
estimators = [10,50,100,200,500,1000,2000]
max_depth = [2,3,5,10]
cv_log_error_array = []
hyperparam_dict = {}

for i in estimators:
    for j in max_depth:
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=random_state,n_jobs=1)
        clf.fit(train_x_onehotCoding, train_y)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_x_onehotCoding, train_y)
        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        hyperparam_dict[(i,j)] = log_loss(cv_y, sig_clf_probs)
rf_ohe_dict["Tuning"] = hyperparam_dict

# Note here it is just idx not param so while saving keep an eye
min_loss_value = min(cv_log_error_array)

minloss_pairs = [tup for tup in hyperparam_dict if hyperparam_dict[tup] == min_loss_value]
best_estimators = minloss_pairs[0][0] 
best_max_depth = minloss_pairs[0][1] 

clf = RandomForestClassifier(n_estimators=best_estimators, criterion='gini', max_depth=best_max_depth, random_state=random_state,n_jobs=1)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_onehotCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

rf_ohe_dict["LogLoss"] = loss_dict 
pprint(rf_ohe_dict,compact=True)


# <h5>5.2. Testing model with best hyper parameters (One Hot Encoding)</h5>

# In[42]:


best_parameters = {}
best_parameters['n_estimators'] = best_estimators
best_parameters['max_depth'] = best_max_depth

clf = RandomForestClassifier(n_estimators=best_estimators, criterion='gini', max_depth=best_max_depth, random_state=random_state, n_jobs=1)
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding,                                                                                            train_y,                                                                                            test_x_onehotCoding,                                                                                            test_y,                                                                                            clf)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
rf_ohe_dict["best_parameters"] = best_parameters

plot_confusion_matrix(test_y, pred_y)


# <h5>5.3. Feature Importance</h5>

# In[43]:


# Model fitting
clf = RandomForestClassifier(n_estimators=best_estimators, criterion='gini', max_depth=best_max_depth, random_state=random_state,n_jobs=1)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
rf_ohe_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h6>Correctly Classified point</h6>

# In[44]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-clf.feature_importances_)

get_impfeature_names(indices[:no_feature],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# <h6>Mis-Classified point</h6>

# In[45]:


test_point_index = random.choice(misclassified_point_idxs)
no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices=np.argsort(-clf.feature_importances_)

get_impfeature_names(indices[:no_feature],                      test_df['Text'].iloc[test_point_index],                     test_df['Gene'].iloc[test_point_index],                     test_df['Variation'].iloc[test_point_index],                      no_feature,                      train_df)


# In[46]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_RF_OHE.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_RF_OHE.pkl")

rf_ohe_dict["Models"] = {"base_classifier"       : MODELS_DIR[1:]+"base_RF_OHE.pkl",                         "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_RF_OHE.pkl"}

all_model_metadata["Random-Forest_with_OHE"] = rf_ohe_dict


# In[47]:


pprint(rf_ohe_dict,compact=True)


# <h4>With Response Coding</h4>

# <h5>5.1. Hyper paramter tuning</h5>

# In[48]:


rf_rc_dict = {}
estimators = [10,50,100,200,500,1000]
max_depth = [2,3,5,10]
cv_log_error_array = []
hyperparam_dict = {}

for i in estimators:
    for j in max_depth:
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=random_state, n_jobs=1)
        clf.fit(train_x_responseCoding, train_y)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_x_responseCoding, train_y)
        sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)
        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        hyperparam_dict[(i,j)] = log_loss(cv_y, sig_clf_probs)
rf_rc_dict["Tuning"] = hyperparam_dict

# Note here it is just idx not param so while saving keep an eye
min_loss_value = min(cv_log_error_array)

minloss_pairs = [tup for tup in hyperparam_dict if hyperparam_dict[tup] == min_loss_value]
best_estimators = minloss_pairs[0][0] 
best_max_depth = minloss_pairs[0][1] 

clf = RandomForestClassifier(n_estimators=best_estimators, criterion='gini', max_depth=best_max_depth, random_state=random_state, n_jobs=1)
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)


loss_dict = {}
predict_y = sig_clf.predict_proba(train_x_responseCoding)
train_loss = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(cv_x_responseCoding)
cv_loss = log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

predict_y = sig_clf.predict_proba(test_x_responseCoding)
test_loss = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

rf_rc_dict["LogLoss"] = loss_dict 
pprint(rf_rc_dict,compact=True)


# <h5>5.2. Testing model with best hyper parameters</h5>

# In[49]:


best_parameters = {}
best_parameters['n_estimators'] = best_estimators
best_parameters['max_depth'] = best_max_depth

clf = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_estimators, criterion='gini', max_features='auto',random_state=random_state,n_jobs=1)
logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_responseCoding,                                                                                            train_y,                                                                                            test_x_responseCoding,                                                                                            test_y,                                                                                            clf)

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points
rf_rc_dict["best_parameters"] = best_parameters

plot_confusion_matrix(test_y, pred_y)


# <h5>5.3. Feature Importance</h5>

# In[50]:


# Model fitting
clf = RandomForestClassifier(n_estimators=best_estimators, criterion='gini', max_depth=best_max_depth, random_state=random_state, n_jobs=1)
clf.fit(train_x_responseCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_responseCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sig_clf.predict(test_x_responseCoding)
total_test_points = test_x_responseCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
rf_rc_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h5>Correctly Classified point</h5>

# In[51]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 27   # 9 classes * 3 number of features

predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index].reshape(1,-1)),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices = np.argsort(-clf.feature_importances_)
print("-"*50)
for i in indices:
    if i<9:
        print("At {0} Gene is important feature".format(i))
    elif i<18:
        print("At {0} Variation is important feature".format(i))
    else:
        print("At {0} Text is important feature".format(i))


# <h5> Misclassified point</h5>

# In[52]:


test_point_index = random.choice(misclassified_point_idxs)

predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index].reshape(1,-1)),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))
indices = np.argsort(-clf.feature_importances_)
print("-"*50)
for i in indices:
    if i<9:
        print("At {0} Gene is important feature".format(i))
    elif i<18:
        print("At {0} Variation is important feature".format(i))
    else:
        print("At {0} Text is important feature".format(i))
        


# In[53]:


# Saving metadata and model
joblib.dump(clf,MODELS_DIR+"base_RF_RC.pkl")
joblib.dump(sig_clf,MODELS_DIR+"calibrated_RF_RC.pkl")

rf_rc_dict["Models"] = {"base_classifier"       : MODELS_DIR[1:]+"base_RF_RC.pkl",                        "calibrated_classifier" : MODELS_DIR[1:]+"calibrated_RF_RC.pkl"}

all_model_metadata["Random-Forest_with_RC"] = rf_rc_dict


# In[54]:


pprint(rf_rc_dict,compact=True)


# <h3>6. Models Stacking </h3>

# <h4>6.1 Testing with hyper parameter tuning</h4>

# In[55]:


stacked_dict = {}

clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log_loss', class_weight='balanced', random_state=random_state)
clf1.fit(train_x_onehotCoding, train_y)
sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=random_state)
clf2.fit(train_x_onehotCoding, train_y)
sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")

clf3 = MultinomialNB(alpha=0.001)
clf3.fit(train_x_onehotCoding, train_y)
sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")

sig_clf1.fit(train_x_onehotCoding, train_y)
LR_logloss = log_loss(cv_y, sig_clf1.predict_proba(cv_x_onehotCoding))

sig_clf2.fit(train_x_onehotCoding, train_y)
SVM_logloss = log_loss(cv_y, sig_clf2.predict_proba(cv_x_onehotCoding))

sig_clf3.fit(train_x_onehotCoding, train_y)
NB_logloss = log_loss(cv_y, sig_clf3.predict_proba(cv_x_onehotCoding))

stacked_dict["base_classifiers"] = {    'clf1' : {'alpha':0.001, 'name':'Logistic Regression', 'logloss': LR_logloss},    'clf2' : {'alpha':1, 'name':'SVM', 'logloss': SVM_logloss},    'clf3' : {'alpha':0.001, 'name':'MultinomialNB', 'logloss': NB_logloss}                            
}

# Tuning the stacking of all three models with alpha where meta classifier will be LR.
alpha = [0.0001,0.001,0.01,0.1,1,10] 
minloss = float('inf')
best_alpha = alpha[0]
hyperparam_dict = {}
for i in alpha:
    lr = LogisticRegression(C=i)
    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)
    sclf.fit(train_x_onehotCoding, train_y)
    stacked_logloss =log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))
    hyperparam_dict[(i,j)] = stacked_logloss


    if stacked_logloss < best_alpha:
        minloss = stacked_logloss
        best_alpha = i
        
stacked_dict["Tuning"] = hyperparam_dict
pprint(stacked_dict,compact=True)


# <h4>6.2 testing the model with the best hyper parameters</h4>

# In[56]:


lr = LogisticRegression(C=best_alpha)
sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)
sclf.fit(train_x_onehotCoding, train_y)

loss_dict = {}
train_loss = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))
cv_loss = log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))
test_loss = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))

loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss

stacked_dict["LogLoss"] = loss_dict 


# In[57]:


best_parameters = {}
best_parameters['Alpha'] = best_alpha

# logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
#                                                                                            train_y, \
#                                                                                            test_x_onehotCoding, \
#                                                                                            test_y, \
#                                                                                            sclf)

sig_clf_probs = sclf.predict_proba(test_x_onehotCoding)
pred_y = sclf.predict(test_x_onehotCoding)

logloss = log_loss(test_y, sig_clf_probs, eps=1e-15)
misclassfied_points = np.count_nonzero((pred_y - test_y))/test_y.shape[0]

best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points

stacked_dict['best_parameters'] = best_parameters

plot_confusion_matrix(test_y, pred_y)
# print(test_y)
# print(pred_y)


# <h5>5.3. Feature Importance</h5>
# For ensemble type models, interpretability is little bit difficult to get.

# In[58]:


# Model fitting
lr = LogisticRegression(C=best_alpha)
sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)
sclf.fit(train_x_onehotCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = sclf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)
stacked_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# <h5>Correctly Classified point</h5>

# In[59]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 100

predicted_cls = sclf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sclf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))

# Ensemble so no feature_importances_
# indices=np.argsort(-sclf.feature_importances_)
# get_impfeature_names(indices[:no_feature], \
#                      test_df['Text'].iloc[test_point_index],\
#                      test_df['Gene'].iloc[test_point_index],\
#                      test_df['Variation'].iloc[test_point_index], \
#                      no_feature, \
#                      train_df)


# <h5> Misclassified point</h5>

# In[60]:


test_point_index = random.choice(misclassified_point_idxs)
no_feature = 100

predicted_cls = sclf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sclf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))

# Ensemble so no feature_importances_
# indices=np.argsort(-sclf.feature_importances_)
# get_impfeature_names(indices[:no_feature], \
#                      test_df['Text'].iloc[test_point_index],\
#                      test_df['Gene'].iloc[test_point_index],\
#                      test_df['Variation'].iloc[test_point_index], \
#                      no_feature, \
#                      train_df)


# In[61]:


# Saving metadata and model
joblib.dump(sclf,MODELS_DIR+"base_STACKED_CLASSIFIER.pkl")

stacked_dict["Models"] = {"base_classifier" : MODELS_DIR[1:]+"base_STACKED_CLASSIFIER.pkl"}

all_model_metadata["Stacked-Classifier"] = stacked_dict


# In[62]:


pprint(stacked_dict,compact=True)


# <h4>6.3 Maximum Voting classifier </h4>

# In[63]:


#Refer:http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
voting_dict = {}

vclf = VotingClassifier(estimators=[('lr', sig_clf1), ('svc', sig_clf2), ('rf', sig_clf3)], voting='soft')
vclf.fit(train_x_onehotCoding, train_y)

train_loss = log_loss(train_y, vclf.predict_proba(train_x_onehotCoding))
cv_loss = log_loss(cv_y, vclf.predict_proba(cv_x_onehotCoding))
test_loss = log_loss(test_y, vclf.predict_proba(test_x_onehotCoding))

loss_dict = {}
loss_dict['Train-loss'] = train_loss
loss_dict['CV-loss'] = cv_loss
loss_dict['Test-loss'] = test_loss


logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding,                                                                                            train_y,                                                                                            test_x_onehotCoding,                                                                                            test_y,                                                                                            vclf)
best_parameters = {}
best_parameters["LogLoss"] = logloss
best_parameters["Misclassified-Percent"] = misclassfied_points

voting_dict['LogLoss'] = loss_dict
voting_dict['best_parameters'] = best_parameters

plot_confusion_matrix(test_y, pred_y)


# In[64]:


# Model fitting
vclf = VotingClassifier(estimators=[('lr', sig_clf1), ('svc', sig_clf2), ('rf', sig_clf3)], voting='soft')
vclf.fit(train_x_onehotCoding, train_y)

# Getting correct & incorrect classifications
correctly_classified_point_idxs = []
misclassified_point_idxs = []

predictions = vclf.predict(test_x_onehotCoding)
total_test_points = test_x_onehotCoding.shape[0]
for i in range(total_test_points):
    if test_y[i] == predictions[i]:
        correctly_classified_point_idxs.append(i)
    else:
        misclassified_point_idxs.append(i)

voting_dict["Classified-Test-Datapoints"] ={ "Correct": correctly_classified_point_idxs, "Misclassified":  misclassified_point_idxs}


# In[65]:


test_point_index = random.choice(correctly_classified_point_idxs)
no_feature = 100

predicted_cls = sclf.predict(test_x_onehotCoding[test_point_index])
print("Predicted Class :{0}".format(predicted_cls[0]))

pred_class_probabilties = np.round(sclf.predict_proba(test_x_onehotCoding[test_point_index]),4)
prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],                                  "Prob":pred_class_probabilties[0]})
print(prob_df.set_index('Class').T)
print("Actual Class : {0}".format(test_y[test_point_index]))

# Ensemble so no feature_importances_
# indices=np.argsort(-sclf.feature_importances_)
# get_impfeature_names(indices[:no_feature], \
#                      test_df['Text'].iloc[test_point_index],\
#                      test_df['Gene'].iloc[test_point_index],\
#                      test_df['Variation'].iloc[test_point_index], \
#                      no_feature, \
#                      train_df)


# In[66]:


# Saving metadata and model
joblib.dump(vclf,MODELS_DIR+"base_VOTING_CLASSIFIER.pkl")

voting_dict["Models"] = {"base_classifier" : MODELS_DIR[1:]+"base_VOTING_CLASSIFIER.pkl"}

all_model_metadata["VOTING-Classifier"] = voting_dict


# In[67]:


pprint(voting_dict,compact=True)


# <h3>Final Metadata</h3>

# In[68]:


pprint(all_model_metadata,compact=True)


# In[69]:


# import joblib
# from pprint import pprint
# MODELS_DIR = "../models/"
# all_model_metadata = joblib.load(MODELS_DIR+"models_metadata.pkl")
# pprint(all_model_metadata, compact=True)


# In[70]:


# Fixing issue of tuple being in key position and convert it to [hp1, hp2, logloss].

def fix_tuple_as_keys(list_model_tuning_tobe_fixed,original_metadata_dict):
    for model_key_name in list_model_tuning_tobe_fixed:
        d = original_metadata_dict[model_key_name]['Tuning']
        tuning_updated = []
        for tup in d:
            hp1, hp2 = tup
            logloss = d[tup]
            tuning_updated.append((hp1, hp2, logloss))
        
        original_metadata_dict[model_key_name]['Tuning'] = tuning_updated
    return original_metadata_dict
        
fix_list = ["Random-Forest_with_OHE", "Random-Forest_with_RC","Stacked-Classifier"]
all_model_metadata = fix_tuple_as_keys(fix_list, all_model_metadata)
pprint(all_model_metadata,compact=True)


# In[71]:


joblib.dump(all_model_metadata, MODELS_DIR+"models_metadata.pkl")


# In[72]:


pprint(all_model_metadata,compact=True)


# In[73]:


time_taken_mins = round((time.time() - start_time)/60,2)
print("Total time taken for executing this whole notebook is : {0} mins".format(time_taken_mins))


# In[ ]:




