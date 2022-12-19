import streamlit as st
import os
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import time
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
import random
import joblib
import math
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")

random_state = 42
np.random.seed(random_state)
random.seed(random_state)
from collections import Counter, defaultdict
from scipy.sparse import hstack

# Utility Functions
# Plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    """
    Usage: plot_confusion_matrix(test_y, predict_y)
    Returns: 3 confusion matrix plots in given order [Confusion Matrix, Recall Matrix, Precision Matrix]
    """
    sns.set(font_scale = 0.7)
    sns.color_palette("rocket")
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10,10))
    
    labels = [1,2,3,4,5,6,7,8,9]
    confusion_mat = confusion_matrix(test_y, predict_y)
    sns.heatmap(confusion_mat, annot=True, fmt=".0f", xticklabels=labels, yticklabels=labels,ax = axes[0])
    axes[0].set_title("Confusion matrix (points)")
    axes[0].set_ylabel('Original Class')
    
    recall_mat =(((confusion_mat.T)/(confusion_mat.sum(axis=1))).T)*100
    sns.heatmap(recall_mat, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,ax = axes[1])
    axes[1].set_title("Recall matrix (%)")
    axes[1].set_ylabel('Original Class')
    
    precision_mat =(confusion_mat/confusion_mat.sum(axis=0))*100
    sns.heatmap(precision_mat, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,ax = axes[2])
    axes[2].set_title("Precision matrix(%)")
    axes[2].set_xlabel('Predicted Class')
    axes[2].set_ylabel('Original Class')
    st.pyplot(fig)



# get_gv_fea_dict: Get Gene varaition Feature Dict
def get_gv_fea_dict(alpha, feature, df):
    """
    # code for response coding with Laplace smoothing.
    # alpha : used for laplace smoothing
    # feature: ['gene', 'variation']
    # df: ['train_df', 'test_df', 'cv_df']
    # algorithm
    # ----------
    # Consider all unique values and the number of occurances of given feature in train data dataframe
    # build a vector (1*9) , the first element = (number of times it occured in class1 + 10*alpha / number of time it occurred in total data+90*alpha)
    # gv_dict is like a look up table, for every gene it store a (1*9) representation of it
    # for a value of feature in df:
    # if it is in train data:
    # we add the vector that was stored in 'gv_dict' look up table to 'gv_fea'
    # if it is not there is train:
    # we add [1/9, 1/9, 1/9, 1/9,1/9, 1/9, 1/9, 1/9, 1/9] to 'gv_fea'
    # return 'gv_fea'
    """
    value_count = df[feature].value_counts()
    
    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation
    gv_dict = dict()
    
    # denominator will contain the number of time that particular feature occured in whole data
    for i, denominator in value_count.items():
        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class
        # vec is 9 diamensional vector
        vec = []
        for k in range(1,10):
            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])
            #         ID   Gene             Variation  Class  
            # 2470  2470  BRCA1                S1715C      1   
            # 2486  2486  BRCA1                S1841R      1   
            # 2614  2614  BRCA1                   M1R      1   
            # 2432  2432  BRCA1                L1657P      1   
            # 2567  2567  BRCA1                T1685A      1   
            # 2583  2583  BRCA1                E1660G      1   
            # 2634  2634  BRCA1                W1718L      1   
            # cls_cnt.shape[0] will return the number of rows

            cls_cnt = df.loc[(df['Class']==k) & (df[feature]==i)]
            
            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data
            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))

        # we are adding the gene/variation to the dict as key and vec as value
        gv_dict[i]=vec
    return gv_dict

# Get Gene variation feature
def get_gv_feature(alpha, feature, df):
    gv_dict = get_gv_fea_dict(alpha, feature, df)
    # value_count is similar in get_gv_fea_dict
    value_count = df[feature].value_counts()
    
    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_fea = []
    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea
    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea
    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
#             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])
    return gv_fea


def extract_dictionary_paddle(cls_text):
    """
    Usage: extract_dictionary_paddle(cls_text)
    returns: dictionary
    Details
    cls_text is a data frame
    for every row in data fram consider the 'Text'
    split the words by space
    make a dict with those words
    increment its count whenever we see that word
    """
    dictionary = defaultdict(int)
    for index, row in cls_text.iterrows():
        for word in row['Text'].split():
            dictionary[word] +=1
    return dictionary

    
def report_log_loss_and_misclassified_points_and_pred_y(train_x, train_y, test_x, test_y, clf):
    """
    Note: Keep an eye for train_x & test_x for perticular problem
    
    Usage: report_log_loss_and_misclassified_points_and_pred_y(train_x, train_y, test_x, test_y, clf)
    Outputs: logloss, misclassfied_points, pred_y
    """
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    sig_clf_probs = sig_clf.predict_proba(test_x)
    
    pred_y = sig_clf.predict(test_x)
    
    logloss = log_loss(test_y, sig_clf_probs, eps=1e-15)
    
    misclassfied_points = np.count_nonzero((pred_y - test_y))/test_y.shape[0]
    
    return logloss, misclassfied_points, pred_y

    
    
def predict_and_plot_confusion_matrix(train_x, train_y, test_x, test_y, clf):
    logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x, train_y, test_x, test_y, clf)
    plot_confusion_matrix(test_y, pred_y)

def get_impfeature_names(indices, text, gene, var, no_features, train_df):
    """
    Usage: get_impfeature_names(indices, text, gene, var, no_features)
    Returns: None
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
    
    st.write("Out of the top {0} features {1} are present in query point".format(no_features,word_present))
    
    if word_present:
        st.write("Features present in Top {0}".format(no_features))
        st.write("\t Gene: {0} | Variation: {1} | Text: {2}".\
               format(len(feature_presence_map['Gene']),len(feature_presence_map['Variantion']),len(feature_presence_map['Text'])))
        st.json(feature_presence_map,expanded=False)
        
    st.write("""---""")
    
