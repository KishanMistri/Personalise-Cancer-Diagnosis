import streamlit as st
import os
import pandas as pd
import numpy as np
import nltk
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
from collections import Counter, defaultdict
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")
random_state = 42
np.random.seed(random_state)
random.seed(random_state)
# Importing utility functions from parent directory. Can't keep in pages as it creates menu in sidebar here.
import sys 
sys.path.insert(0, '../')
from utilities import *

st.title("📊 Exploratory Data Analysis ")
st.sidebar.write("# 📊 EDA ")

# Variable Setup
TRAIN_DATA_DIR = "./train_data/" 
TEST_DATA_DIR = "./test_data/"
PROCESSED_DATA_DIR = "./preprocessed/"
MODELS_DIR = "./models/"

# Reading files
train_text = pd.read_csv(TRAIN_DATA_DIR+'training_text',sep='\|\|',engine='python',header=None,names=['ID','Text'],skiprows=1,index_col='ID')
train_variant = pd.read_csv(TRAIN_DATA_DIR+'training_variants',index_col='ID')

#######################################################
##############  Data Import & Details  ################
#######################################################

intro_data_details= """<h6>1. Data Import & Details: </h6>"""

text_fields = """<small><p>"Text" - literature which every expert needs to go through manually prior giving conclusion, which is very time and human resource consuming</p></small>"""

variant_fields = """
<small><p>"ID" - mapper which will be used to merge both dataframes/tables</p></small>
<small><p>"Gene" - Gene Sequance</p></small>
<small><p>"Variation" - Gene's variant </p></small>
<small><p>"Class" - Which class of Cancer is detected by this data</p></small>
"""

# Streamlit components
st.write(intro_data_details,unsafe_allow_html=True)

text_col, variant_col = st.columns(2)
with text_col:
    text_expander = st.expander(label='Raw Training Text')
    with text_expander:
        st.caption("Number of datapoints: {}".format(train_text.shape[0]))
        st.caption("Number of Features:{}".format(train_text.shape[1]))
        st.caption("Features: {}".format(train_text.columns.tolist()))
        st.write(text_fields,unsafe_allow_html=True)
        st.write(train_text.head())

with variant_col:
    variant_expander = st.expander(label='Raw Training Variants')
    with variant_expander:
        st.caption("Number of datapoints: {}".format(train_variant.shape[0]))
        st.caption("Number of Features: {}".format(train_variant.shape[1]))
        st.caption("Features: {}".format(train_variant.columns.tolist()))
        st.write(variant_fields,unsafe_allow_html=True)
        st.write(train_variant.head())

#######################################################
############  Text Data Pre-Processing  ###############
#######################################################

intro_data_details= """
<h6>2. Text Data Pre-Processing: </h6>
<ol>
<li>Replaced all the special charecters.</li>
<li>Removed multiple space with single space.</li>
<li>Converted text to lower case</li>
<li>Removed Stopwords from the text</li>
<li>Added Null text as a combination of "GENE VARIANTION"</li>
</ol>
"""

merged_results = pd.merge(train_text,train_variant,on='ID')


# loading stop words from nltk library
stop_words = set(stopwords.words('english'))

@st.cache(suppress_st_warning=True)
def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+',' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
        
        merged_results[column][index] = string

#text processing stage.
# start_time = time.clock()
for index, row in merged_results.iterrows():
    if type(row['Text']) is str:
        nlp_preprocessing(row['Text'], index, 'Text')
    else:
        print("there is no text description for id:",index)
# print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
merged_results.loc[merged_results['Text'].isnull(),'Text'] = merged_results['Gene'] +' '+merged_results['Variation']


# Streamlit components
st.write(intro_data_details,unsafe_allow_html=True)

combined_expander = st.expander(label='View Dataframe')
with combined_expander:
    st.write(merged_results.head())

#######################################################
###### Test, Train and Cross Validation Split  ########
#######################################################

data_distribution_markup="""
<h6>3. Train, Cross-Validation & Test Split distributions: </h6>
<li>As we have used statify by class labels, they are following same distribution across all three sets</li>
<li>We have splitted the data set Train : CV : Test with 64:16:20 ratio. </li>
"""
 
# Splitting data into train, test and cross validation (64:20:16)
y_true = merged_results['Class'].values
merged_results.Gene      = merged_results.Gene.str.replace('\s+', '_',regex=True)
merged_results.Variation = merged_results.Variation.str.replace('\s+', '_',regex=True)

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
X_train, test_df, y_train, y_test = train_test_split(merged_results, y_true, stratify=y_true, test_size=0.2)
# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

def get_distribution(df,dftype,ax):
    # it returns a dict, keys as class labels and values as the number of data points in that class
    df_class_distribution = df['Class'].value_counts().sort_index()
        
    df_class_distribution.plot(kind='bar',ax=ax, cmap = plt.colormaps['Pastel1'])
    ax.set_xlabel('Classes')
    ax.set_ylabel('Data points per Class')
    ax.set_title('{} data'.format(dftype))
    
sns.set()
sns.color_palette()
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))
fig.suptitle('Distribution of classes')
get_distribution(train_df,"Train",axes[0])
get_distribution(cv_df,"Cross-Validation",axes[1])
get_distribution(test_df,"Test",axes[2])

# Streamlit components
st.write(data_distribution_markup,unsafe_allow_html=True)
st.pyplot(fig)


#######################################################
############  Random Model Performance  ###############
#######################################################

random_model_markup="""
<h6>4. Random Model Performance: </h6>
<li>Selecting/Guessing random class resulting LogLoss values (shown in caption of matrices). Which means any model created going forward should have less than this loss values.</li>
<li>We need to maximize the diagonal cell values for the better model.</li>
<li>Why Confusion matrices, because it will tell us how good out model is performing on each class. We want a model which performs equally good for all classes and not biased toward single class.</li>
<li>Why not, AUC or F1. Because we have multiclass where it won't help much.</li>
"""
test_data_len = test_df.shape[0]
cv_data_len = cv_df.shape[0]

cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
train_logloss = log_loss(y_cv,cv_predicted_y, eps=1e-15)

test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
test_logloss = log_loss(y_test,test_predicted_y, eps=1e-15)

predicted_y =np.argmax(test_predicted_y, axis=1)

st.write(random_model_markup,unsafe_allow_html=True)
st.caption("Log loss (Cross Validation): {}".format(round(train_logloss,3)))
st.caption("Log loss (Test): {}".format(round(test_logloss,3)))
    
base_model_expander = st.expander(label='Base Model\'s confusion matrices')
with base_model_expander:
    plot_confusion_matrix(y_test, predicted_y+1)
    
#######################################################
###############  Univariate Analysis  #################
#######################################################

univariate_analysis_markup="""
<h6>4. Univariate Analysis: </h6>
"""
st.write(univariate_analysis_markup,unsafe_allow_html=True)
gene_feature_expander = st.expander(label='1. Gene feature Analysis')
with gene_feature_expander:
    ###############  4.1.1  #################
    st.write("""<h5>1. How Gene's are Distributed?<h5>""",unsafe_allow_html=True)
    unique_genes = train_df['Gene'].value_counts()
    st.caption("Total number of unique Gene's: {}".format(unique_genes.shape[0]))
    st.write(unique_genes.reset_index().transpose())
    
    s = sum(unique_genes.values)
    h = unique_genes.values/s
    
    # st.write(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write("<h6>PDF</h6>",unsafe_allow_html=True)
        pdf = pd.DataFrame({"Gene Index":range(1,unique_genes.shape[0]+1),"Portion": h})
        st.line_chart(data=pdf,x='Gene Index',y='Portion')
        
    with col2:
        st.write("<h6>CDF</h6>",unsafe_allow_html=True)
        c = np.cumsum(h)
        cdf = pd.DataFrame({"Gene Index":range(1,unique_genes.shape[0]+1),"Portion": c})
        st.line_chart(data=cdf,x='Gene Index',y='Portion')
    st.write("""<ul align='center'>
                      TOP 65 Genes having 80% of portion.
                  </ul>""",unsafe_allow_html=True)

    ###############  4.1.2  #################
    st.write("""<br/><h5>2. How to featurize this feature "Gene"?<h5>
                    <ul>
                        <li>One hot Encoding</li>
                        <li>Response coding</li>
                    </ul>
                    <small>For this problem of multi-class classification with categorical features, one-hot encoding is better for Logistic regression while response coding is better for Random Forests.</small>
             """,unsafe_allow_html=True)
    
    #response-coding of the Gene feature
    # alpha is used for laplace smoothing
    alpha = 1
    # train gene feature
    train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene",train_df))
    # test gene feature
    test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df))
    # cross validation gene feature
    cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", cv_df))
    
    st.caption("Using respone coding method the shape of gene feature: {0}".format(train_gene_feature_responseCoding.shape))
    
    # one-hot encoding of Gene feature.
    gene_vectorizer = CountVectorizer()
    train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
    test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
    cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
    
    st.caption("Using one-hot encoding method the shape of gene feature: {0}".format(train_gene_feature_onehotCoding.shape))
    
    ###############  4.1.3  #################
    st.write("""
            <h5>3. Is feature "Gene" useful in predicting Class values?<h5>
            <small><ul>
            <li>There are many methods, however we can simple build simple model from given feature's data and see if the loss values are affected by it or not. If loss values is less than random model, then it is proof that it is having impact.</li>    
            <li>The following question would be how much impact it is having, that will also given by loss values themselves.</li>
            </ul></small>
            <li>Answer: Yes it is based on below analysis logloss reduces drastically so it is very useful feature for modelling.</li>
            """,unsafe_allow_html=True)
    # hyperparameter of SGD classifier.
    # Refer: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    alpha = [10 ** x for x in range(-5, 1)] 
    cv_log_error_array=[]
    cv_logloss = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=random_state)
        clf.fit(train_gene_feature_onehotCoding, y_train)
        # Platt Scalling
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_gene_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(alpha, cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha")
    plt.ylabel("LogLoss")
    st.pyplot(fig)
    
    logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": cv_log_error_array})
    st.caption("Logloss values at given Alpha")
    st.write(logloss_df.set_index('Alpha').T)

    ###############  4.1.4  #################
    st.write("""
            <h5>4. Is feature "Gene" stable across all sets?<h5>
            <small><ul>
            <li>For consistency reasons, this needs to be checked as well. Stable means similar data present across all dataset</li>
            <li>It is necessary as different dataset won't truly say how good model is performing overall. All data is seen in train will be used for CV and Test.</li>
            <i>Do not confuse this with data leakage.</i>
            </small></ul>
            <li>Answer: Yes, it is as CV & Test loss values are not too far from training loss.</li>
            """,unsafe_allow_html=True)
    
    best_alpha = np.argmin(cv_log_error_array)
    clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log_loss', random_state=random_state)
    clf.fit(train_gene_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_gene_feature_onehotCoding, y_train)
    
    best_logloss_across = {}

    predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)
    best_logloss_across["train"] = {"alpha":alpha[best_alpha], "logloss": log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)}
    
    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
    best_logloss_across["cv"] =   {"alpha":alpha[best_alpha], "logloss":log_loss(y_cv  , predict_y, labels=clf.classes_, eps=1e-15)}
    
    predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)
    best_logloss_across["test"] = {"alpha":alpha[best_alpha], "logloss":log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)}
    st.write(best_logloss_across)
  
    ###############  4.1.5  #################
    st.write("""
            <h5>5. Is feature "Gene" covering most data in test & cross-validation sets?<h5>
            <small><ul>
                <li>Why you ask? - In CV and Test phase, if this feature's values set are not covering the most features values used during training phase, then it is harder</li>
                <i>Maximizing the overall feature's values intersection across all sets is standard practice.</i>
            </ul></small>
            <li>Answer: Yes, it is as covering most value set across CV & Test phases.</li>
            """,unsafe_allow_html=True)
    
    cv_coverage   = cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]
    test_coverage = test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

    st.caption("In CV   data: {0} out of {1} covering: {2}%".format(cv_coverage, cv_df.shape[0], round((cv_coverage/cv_df.shape[0])*100,2)))
    st.caption("In test data: {0} out of {1} covering: {2}%".format(test_coverage, test_df.shape[0], round((test_coverage/test_df.shape[0])*100,2)))

####################################################
##############  Variation  #########################
####################################################
    
variation_feature_expander = st.expander(label='2. Variation feature Analysis')
with variation_feature_expander:
    ###############  4.2.1  #################
    st.write("""<h5>1. How Variation's are Distributed?<h5>""",unsafe_allow_html=True)
    unique_variation = train_df['Variation'].value_counts()
    st.caption("Total number of unique Variation's: {}".format(unique_variation.shape[0]))
    st.caption("As you can see Top 4 variants have the hights count and then counts decreases flat.")
    st.write(unique_variation.reset_index().transpose())
    
    s = sum(unique_variation.values)
    h = unique_variation.values/s
    
    # st.write(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write("<h6>PDF</h6>",unsafe_allow_html=True)
        pdf = pd.DataFrame({"Variation Index":range(1,unique_variation.shape[0]+1),"Portion": h})
        st.line_chart(data=pdf,x='Variation Index',y='Portion')
        
    with col2:
        st.write("<h6>CDF</h6>",unsafe_allow_html=True)
        c = np.cumsum(h)
        cdf = pd.DataFrame({"Variation Index":range(1,unique_variation.shape[0]+1),"Portion": c})
        st.line_chart(data=cdf,x='Variation Index',y='Portion')
    st.caption("""<ul align='center'>
                      As there are drastic drop and then continuous increase the CDF is increasing gradually. However, spliting data with these feature values will have less overlap.
                  </ul>""",unsafe_allow_html=True)

    ###############  4.2.2  #################
    st.write("""<br/><h5>2. How to featurize this feature "Variation"?<h5>
                    <ul>
                        <li>One hot Encoding</li>
                        <li>Response coding</li>
                    </ul>
                    <small>Note: OHE has issue of high dimensionality vector creation and Response Coding has issue of "Response Leakage" where non-trained feature value occur in CV or Test Phase. At that time, we keep probabilty of it as equi-probable.
                    Same as earlier.</small>
             """,unsafe_allow_html=True)
    
    #response-coding of the Variation feature
    # alpha is used for laplace smoothing
    alpha = 1
    # train variation feature
    train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation",train_df))
    # test variation feature
    test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))
    # cross validation variation feature
    cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))
    
    st.caption("Using respone coding method the shape of Variation feature: {0}".format(train_variation_feature_responseCoding.shape))
    
    # one-hot encoding of variation feature.
    variation_vectorizer = CountVectorizer()
    train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
    test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
    cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
    
    st.caption("Using one-hot encoding method the shape of variation feature: {0}".format(train_variation_feature_onehotCoding.shape))
    
    ###############  4.2.3  #################
    st.write("""
            <h5>3. Is feature "Variation" useful in predicting Class values?<h5>
            <li>Answer: From the loss values it seems it will be useful. Not as good as "Gene" feature though.</li>
            """,unsafe_allow_html=True)
    # hyperparameter of SGD classifier.
    # Refer: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    alpha = [10 ** x for x in range(-5, 1)] 
    cv_log_error_array=[]
    cv_logloss = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=random_state)
        clf.fit(train_variation_feature_onehotCoding, y_train)
        # Platt Scalling
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_variation_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(alpha, cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha")
    plt.ylabel("LogLoss")
    st.pyplot(fig)
    
    logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": cv_log_error_array})
    st.caption("Logloss values at given Alpha")
    st.write(logloss_df.set_index('Alpha').T)

    ###############  4.2.4  #################
    st.write("""
            <h5>4. Is feature "Variation" stable across all sets?<h5>
            <small><ul>
            <li>As stated earlier in PDF & CDF splitting such value set will have less overlap accross sets. Less even value spread results</li>
            </small></ul>
            <li>Answer: Not sure, it is as CV & Test loss values are little far from training loss.</li>
            """,unsafe_allow_html=True)
    
    best_alpha = np.argmin(cv_log_error_array)
    clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log_loss', random_state=random_state)
    clf.fit(train_variation_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_variation_feature_onehotCoding, y_train)
    
    best_logloss_across = {}

    predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding)
    best_logloss_across["train"] = {"alpha":alpha[best_alpha], "logloss": log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)}
    
    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
    best_logloss_across["cv"] =   {"alpha":alpha[best_alpha], "logloss":log_loss(y_cv  , predict_y, labels=clf.classes_, eps=1e-15)}
    
    predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)
    best_logloss_across["test"] = {"alpha":alpha[best_alpha], "logloss":log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)}
    st.write(best_logloss_across)
  
    ###############  4.2.5  #################
    st.write("""
            <h5>5. Is feature "Variation" covering most data in test & cross-validation sets?<h5>
            <small><ul>
                <i>Maximizing the overall feature's values intersection across all sets is standard practice.</i>
            </ul></small>
            <li>Answer: No, it is terribly spread across sets due to splitting less value frequencies.</li>
            """,unsafe_allow_html=True)
    
    cv_coverage   = cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]
    test_coverage = test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

    st.caption("In CV   data: {0} out of {1} covering: {2}%".format(cv_coverage, cv_df.shape[0], round((cv_coverage/cv_df.shape[0])*100,2)))
    st.caption("In test data: {0} out of {1} covering: {2}%".format(test_coverage, test_df.shape[0], round((test_coverage/test_df.shape[0])*100,2)))

    
####################################################
###################  Text  #########################
####################################################
    
text_feature_expander = st.expander(label='3. Text feature Analysis')
with text_feature_expander:
    ###############  4.3.1  #################
    st.write("""<h5>1. How are word frequencies distributed?<h5>""",unsafe_allow_html=True)
    
    # building a CountVectorizer with all the words that occured minimum 3 times in train data
    text_vectorizer = CountVectorizer(min_df=3)
    train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['Text'])
    # getting all the feature names (words)
    train_text_features= text_vectorizer.get_feature_names()

    # train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
    train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1

    # zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
    text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))
    st.caption("Total number of corpus words in train data: {0}".format(len(train_text_features)))
    
    # build a word dict based on the words in that class
    dict_list = []
    for i in range(1,10):
        cls_text = train_df[train_df['Class']==i]
        dict_list.append(extract_dictionary_paddle(cls_text))
        
    # dict_list[i] is build on i'th  class text data
    # total_dict is buid on whole training text data
    total_dict = extract_dictionary_paddle(train_df)

    confuse_array = []
    for i in train_text_features:
        ratios = []
        max_val = -1
        for j in range(0,9):
            ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))
        confuse_array.append(ratios)
    confuse_array = np.array(confuse_array)
    
    def get_text_responsecoding(df):
        """
        Usage: get_text_responsecoding(df)
        Returns: text_feature_responseCoding

        Refer: https://stackoverflow.com/a/1602964
        """
        text_feature_responseCoding = np.zeros((df.shape[0],9))
        for i in range(0,9):
            row_index = 0
            for index, row in df.iterrows():
                sum_prob = 0
                for word in row['Text'].split():
                    sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))
                text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['Text'].split()))
                row_index += 1
        return text_feature_responseCoding
    
    #Response coding of text features
    train_text_feature_responseCoding = get_text_responsecoding(train_df)
    test_text_feature_responseCoding  = get_text_responsecoding(test_df)
    cv_text_feature_responseCoding    = get_text_responsecoding(cv_df)

    # Normalizing each rows
    train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T
    test_text_feature_responseCoding  = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T
    cv_text_feature_responseCoding    = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T

    # Normalize every feature
    train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)

    # Trasform & Normalize test and cv vectorizers
    test_text_feature_onehotCoding = text_vectorizer.transform(test_df['Text'])
    test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

    cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['Text'])
    cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)

    sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))
    sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))

    # Number of words for a given frequency.
    corp = Counter(sorted_text_occur)
    
    st.line_chart(data=pd.DataFrame.from_dict(dict(corp),orient='index'))
    
    
    ###############  4.3.2  #################
    st.write("""<h5>2. How to featurize feature "Text"?<h5>
             <small>
                 <ul>
                     <li>Bag Of Words/One Hot Encoding</li>
                     <li>Response Encoding</li>
                     <li>TFIDF</li>
                     <li>Word2Vec</li>
                 </ul>
             </small>
             """,unsafe_allow_html=True)
    
    ###############  4.3.3  #################
    st.write("""<h5>3. Is feature "Text" useful in predicitng Classes?<h5>
                <li>Yes. Definitely..!</li>
             """,unsafe_allow_html=True)
    
    alpha = [10 ** x for x in range(-5, 1)]
    cv_log_error_array=[]
    cv_logloss = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log_loss', random_state=random_state)
        clf.fit(train_text_feature_onehotCoding, y_train)
        # Platt Scalling
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_text_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(alpha, cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]),rotation=20)
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha")
    plt.ylabel("LogLoss")
    st.pyplot(fig)
    
    logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": cv_log_error_array})
    st.caption("Logloss values at given Alpha")
    st.write(logloss_df.set_index('Alpha').T)

    ###############  4.3.4  #################
    st.write("""
            <h5>4. Is feature "Text" stable across all sets?<h5>
            <li>Answer: Yes, it seems, we cannot be sure.</li>
            """,unsafe_allow_html=True)
    
    best_alpha = np.argmin(cv_log_error_array)
    clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log_loss', random_state=random_state)
    clf.fit(train_text_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_text_feature_onehotCoding, y_train)
    
    best_logloss_across = {}

    predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding)
    best_logloss_across["train"] = {"alpha":alpha[best_alpha], "logloss": log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)}
    
    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
    best_logloss_across["cv"] =   {"alpha":alpha[best_alpha], "logloss":log_loss(y_cv  , predict_y, labels=clf.classes_, eps=1e-15)}
    
    predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding)
    best_logloss_across["test"] = {"alpha":alpha[best_alpha], "logloss":log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)}
    st.write(best_logloss_across)
  
    ###############  4.3.5  #################
    st.write("""
            <h5>5. Is feature "Text" covering most data in test & cross-validation sets?<h5>
            <li>Answer: Yes, this is the by far the best overlap/intersection of sets.</li>
            """,unsafe_allow_html=True)
    def get_intersec_text(df):
        df_text_vec = CountVectorizer(min_df=3)
        df_text_fea = df_text_vec.fit_transform(df['Text'])
        df_text_features = df_text_vec.get_feature_names()

        df_text_fea_counts = df_text_fea.sum(axis=0).A1
        df_text_fea_dict = dict(zip(list(df_text_features),df_text_fea_counts))
        len1 = len(set(df_text_features))
        len2 = len(set(train_text_features) & set(df_text_features))
        return len1,len2
    
    len1, len2 = get_intersec_text(test_df)
    st.caption("{0}% of word of test data appeared in train data".format(np.round((len2/len1)*100, 3)))
    len1, len2 = get_intersec_text(cv_df)
    st.caption("{0}% of word of Cross Validation appeared in train data".format(np.round((len2/len1)*100, 3)))

    
modelling_prep_expander = st.expander(label='>>> Data Preparation for Modelling')
with modelling_prep_expander:
    # Merging gene, variance and text features
    # building train, test and cross validation data sets
    # a = [[1, 2], 
    #      [3, 4]]
    # b = [[4, 5], 
    #      [6, 7]]
    # hstack(a, b) = [[1, 2, 4, 5],
    #                [ 3, 4, 6, 7]]
    
    
    # Prepating OHE features
    train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))
    test_gene_var_onehotCoding  = hstack((test_gene_feature_onehotCoding, test_variation_feature_onehotCoding))
    cv_gene_var_onehotCoding    = hstack((cv_gene_feature_onehotCoding,   cv_variation_feature_onehotCoding))

    train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
    train_y = np.array(list(train_df['Class']))

    test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
    test_y = np.array(list(test_df['Class']))

    cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()
    cv_y = np.array(list(cv_df['Class']))
    
    st.caption("Appending/Stacking each feature to create final feature set.")
    st.caption("Final features = [ GENE FEATURE COLUMNS | VARIATION FEATURE COLUMNS | TEXT FEATURE COLUMNS ]")
    
    st.write("One hot encoding features:")
    st.caption("Train data = {0}".format(train_x_onehotCoding.shape))
    st.caption("Cross Validation data = {0}".format(cv_x_onehotCoding.shape))
    st.caption("Test data = {0}".format(test_x_onehotCoding.shape))
    
    
    # Prepating Response Coding features
    train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))
    test_gene_var_responseCoding  = np.hstack((test_gene_feature_responseCoding, test_variation_feature_responseCoding))
    cv_gene_var_responseCoding    = np.hstack((cv_gene_feature_responseCoding,   cv_variation_feature_responseCoding))

    train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))
    test_x_responseCoding  = np.hstack((test_gene_var_responseCoding,  test_text_feature_responseCoding))
    cv_x_responseCoding    = np.hstack((cv_gene_var_responseCoding,    cv_text_feature_responseCoding))
    
    st.write("Response encoding features:")
    st.caption("Train data = {0}".format(train_x_responseCoding.shape))
    st.caption("Cross Validation data = {0}".format(test_x_responseCoding.shape))
    st.caption("Test data = {0}".format(cv_x_responseCoding.shape))
    
    export_preprocessed = {"train_df": [PROCESSED_DATA_DIR+"train_df.pkl",train_df],\
                           "cv_df": [PROCESSED_DATA_DIR+"cv_df.pkl",cv_df],\
                           "test_df": [PROCESSED_DATA_DIR+"test_df.pkl",test_df],\
                           "y_train": [PROCESSED_DATA_DIR+"y_train.pkl",y_train],\
                           "y_cv": [PROCESSED_DATA_DIR+"y_cv.pkl",y_cv],\
                           "y_test": [PROCESSED_DATA_DIR+"y_test.pkl",y_test],\
                           # Below are only for EDA and creation of final one hot encoded & response variables
                           # "train_gene_var_onehotCoding": [PROCESSED_DATA_DIR+"train_gene_var_onehotCoding.pkl",train_gene_var_onehotCoding],\
                           # "test_gene_var_onehotCoding" : [PROCESSED_DATA_DIR+"test_gene_var_onehotCoding.pkl",test_gene_var_onehotCoding],\
                           # "cv_gene_var_onehotCoding"   : [PROCESSED_DATA_DIR+"cv_gene_var_onehotCoding.pkl",cv_gene_var_onehotCoding],\
                           # "train_gene_var_responseCoding": [PROCESSED_DATA_DIR+"train_gene_var_responseCoding.pkl",train_gene_var_responseCoding],\
                           # "test_gene_var_responseCoding" : [PROCESSED_DATA_DIR+"test_gene_var_responseCoding.pkl",test_gene_var_responseCoding],\
                           # "cv_gene_var_responseCoding"   : [PROCESSED_DATA_DIR+"cv_gene_var_responseCoding.pkl",cv_gene_var_responseCoding],\
                           #
                           # Numpy arrays
                           "train_x_onehotCoding"       : [PROCESSED_DATA_DIR+"train_x_onehotCoding.pkl",train_x_onehotCoding],\
                           "test_x_onehotCoding"        : [PROCESSED_DATA_DIR+"test_x_onehotCoding.pkl",test_x_onehotCoding],\
                           "cv_x_onehotCoding"          : [PROCESSED_DATA_DIR+"cv_x_onehotCoding.pkl",cv_x_onehotCoding],\
                           "train_x_responseCoding"     : [PROCESSED_DATA_DIR+"train_x_responseCoding.pkl",train_x_responseCoding],\
                           "test_x_responseCoding"      : [PROCESSED_DATA_DIR+"test_x_responseCoding.pkl",test_x_responseCoding],\
                           "cv_x_responseCoding"        : [PROCESSED_DATA_DIR+"cv_x_responseCoding.pkl",cv_x_responseCoding],\
                           "train_y"                    : [PROCESSED_DATA_DIR+"train_y.pkl",train_y],\
                           "test_y"                     : [PROCESSED_DATA_DIR+"test_y.pkl",test_y],\
                           "cv_y"                       : [PROCESSED_DATA_DIR+"cv_y.pkl",cv_y]}
    
    # Saving Splits - Used decorator so to mitigate repititive saves load
    @st.cache(suppress_st_warning=True)
    def save_processed_data(export_path):
        for df_name in export_path:
            path, df = export_path[df_name]
            st.caption("Name: {0} -> Shape: {1}".format(df_name,df.shape))
            if os.path.exists(path):
                continue
            else:
                joblib.dump(df, path)
    st.write("Saving files..!")
    save_processed_data(export_preprocessed)