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
sys.path.insert(0, './')
from utilities import *

# Setting seed for reproducibility
random_state = 42
np.random.RandomState(random_state)
np.random.seed(random_state)
random.seed(random_state)

st.title("🤖 Baseline Modelling & Hyper Tuning")
st.sidebar.markdown("# 🤖 Modelling")


st.caption("❗For Modelling, please run Jupyter Notebook named : \"2. Modelling & Tuning.ipynb\" in notebooks folder.❗")
st.caption("❗We will use the metadata collected throughout the train-tune-testing.❗")
st.caption("❗It take alot of train time and here we want to show the analysis and model details of fune-tune models.❗")


# Environment Variables & Data imports
TRAIN_DATA_DIR = "./train_data/" 
TEST_DATA_DIR = "./test_data/"
PROCESSED_DATA_DIR = "./preprocessed/"
MODELS_DIR = "./models/"
# Metadata imports
metadata = joblib.load(MODELS_DIR+"models_metadata.pkl")
meta_dict_name_map = { "nb"  : "MultinomialNB",\
                       "knn" : "KNN",\
                       "lr_balanced": "LOGREG_BALANCED",\
                       "lr_unbalanced": "LOGREG_UNBALANCED",\
                       "rf_ohe" : "Random-Forest_with_OHE",\
                       "rf_rc" : "Random-Forest_with_RC", \
                       "svm" : "SVM",\
                       "stacked" : "Stacked-Classifier",\
                       "voting" : "VOTING-Classifier"\
                     }

Random_CV_Logloss   = round(2.450964298929681,3)
Random_Test_Logloss = round(2.5194404639993384,3)

#########################################################################
#######################      NAIVE BAYES     ############################
#########################################################################

nb_details= """
<h5>1. MultinomialNB</h5>
<small>
    <ul>
        <li>For text classification problems, NB is always a go to for baseline model due to discrete features.</li>
        <li>We are using multinomial NB here with smoothing.</li>
    </ul>
</small>
"""
with st.expander("1. Naive Bayes"):
    st.markdown(nb_details,unsafe_allow_html=True)
    meta_dict = metadata[meta_dict_name_map["nb"]]
    st.latex(r'''
                {\widehat\theta}_{yi} = \frac{N_{yi}+\alpha}{N_y+\alpha n}
            ''')
    st.caption("Where N{yi} is the number of times feature appears in a sample of class in the training set & N{y} is the total count of all features for class.")
    
    Hyperparameter, Tuning, best_tab = st.tabs(["Hyperparameters","Hyperparameter Tuning","Best Parameter Analysis"])
    
    with Hyperparameter:
        if "hp_selected" not in st.session_state:
            st.session_state.hp_selected = "Alpha"

        hp_options, hp_details = st.columns([1,3])

        with hp_options:
            st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                     options = ["Alpha"], 
                                                     key="hpvisibility")

        with hp_details:
            if st.session_state.hp_selected == "Alpha":
                st.caption("1. Using higher alpha values will push the likelihood towards a value of 0.5.")
                st.caption("2. It means the probability of a word equal to 0.5 for both the positive and negative reviews.")
                st.caption("3. Alpha values also used for smoothing and not making the probability 0.")  
            
    with Tuning:
        st.markdown("Plotting CV loss across Alpha.")
        # plotting cv
        tune_dict = meta_dict["Tuning"]
        alpha = sorted(tune_dict.keys())
        logloss = [ tune_dict[alp] for alp in tune_dict]
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(np.log10(alpha), logloss, c = 'g', marker = "o")
        for i, txt in enumerate(np.round(logloss,3)):
            ax.annotate(text=(alpha[i],str(txt)), xy=(np.log10(alpha[i]),logloss[i]), xytext=(np.log10(alpha[i])-0.4,logloss[i]+0.005), rotation=20)
        plt.xticks(np.log10(alpha))
        plt.title("Cross Validation (LogLoss) Error for different Alpha", pad=25)
        plt.xlabel("Alpha (10's power)")
        plt.ylabel("LogLoss")
        st.pyplot(fig, dpi=400)
        
        # Not so helping graph
        # tune_df = pd.DataFrame.from_dict(tune_dict, orient='index', columns=["logloss"]).reset_index()
        # st.write(tune_df)
        # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        # plt.grid(visible=True)
        # st.line_chart(data = tune_df, x = 'index', y= 'logloss', use_container_width=True)
    
    with best_tab:
        # For comparison with random model
        loglosses = meta_dict["LogLoss"]
        train_loss = round(loglosses['Train-loss'],3)
        cv_loss = round(loglosses['CV-loss'],3)
        test_loss = round(loglosses['Test-loss'],3)
        
        st.markdown("LogLoss in comparion with Random model")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Train Loss", value = train_loss)
        col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
        col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')
        
        # Writing best parameters
        best_params = meta_dict["best_parameters"]
        st.markdown("This is the best parameter values:")
        st.write(best_params)
        
        # Write Ups
        st.markdown("Observations")
        st.caption("As you can see logloss values are dropped drastically which is great.")
        st.caption("The misclassfication rate is ~45% which is high.")
        
        