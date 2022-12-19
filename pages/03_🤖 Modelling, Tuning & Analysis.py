import streamlit as st
import psutil
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
import plotly.express as px
import warnings
import subprocess
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

st.title("🤖 Modelling, Hyper Tuning & Analysis")
st.sidebar.markdown("# 🤖 Modelling")

# Environment Variables & Data imports
TRAIN_DATA_DIR = "./train_data/" 
TEST_DATA_DIR = "./test_data/"
PROCESSED_DATA_DIR = "./preprocessed/"
MODELS_DIR = "./models/"

model_py_path = "./notebooks/Modelling_Tuning.py"
    
    
@st.cache(suppress_st_warning=True)
def is_running(script_path):
    for q in psutil.process_iter():
        if q.name().startswith('python3'):
            if len(q.cmdline())>1 and script_path in q.cmdline()[1] and q.pid !=os.getpid():
                print("'{}' Process is already running".format(script_path))
                return True
    return False

st.write("""---""")
# If metadata exists then only render page else note for directions
if os.path.exists(MODELS_DIR+"models_metadata.pkl"):
    # Metadata imports
    metadata = joblib.load(MODELS_DIR+"models_metadata.pkl")
    Random_CV_Logloss   = round(2.450964298929681,3)
    Random_Test_Logloss = round(2.5194404639993384,3)

    # Data imports
    train_df = joblib.load(PROCESSED_DATA_DIR+"train_df.pkl")
    cv_df    = joblib.load(PROCESSED_DATA_DIR+"cv_df.pkl")
    test_df  = joblib.load(PROCESSED_DATA_DIR+"test_df.pkl")
    y_train  = joblib.load(PROCESSED_DATA_DIR+"y_train.pkl")
    y_cv     = joblib.load(PROCESSED_DATA_DIR+"y_cv.pkl")
    y_test   = joblib.load(PROCESSED_DATA_DIR+"y_test.pkl")

    train_x_onehotCoding = joblib.load(PROCESSED_DATA_DIR+"train_x_onehotCoding.pkl")
    test_x_onehotCoding  = joblib.load(PROCESSED_DATA_DIR+"test_x_onehotCoding.pkl")
    cv_x_onehotCoding    = joblib.load(PROCESSED_DATA_DIR+"cv_x_onehotCoding.pkl")

    train_x_responseCoding = joblib.load(PROCESSED_DATA_DIR+"train_x_responseCoding.pkl")
    test_x_responseCoding  = joblib.load(PROCESSED_DATA_DIR+"test_x_responseCoding.pkl")
    cv_x_responseCoding    = joblib.load(PROCESSED_DATA_DIR+"cv_x_responseCoding.pkl")

    train_y = joblib.load(PROCESSED_DATA_DIR+"train_y.pkl")
    test_y  = joblib.load(PROCESSED_DATA_DIR+"test_y.pkl")
    cv_y    = joblib.load(PROCESSED_DATA_DIR+"cv_y.pkl")

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

    model_options_list = ('1. Naive Bayes', \
                          '2. K Nearest Neighbors',\
                          '3. Logistic Regression',\
                          '4. Support Vector Machine (SVM)',\
                          '5. Random Forest Classifier',\
                          '6. StackedClassifier (Ensemble)',\
                          '7. MaxVoting Classifier (Ensemble)')

    selected_model = st.selectbox(label   = "Select Model",
                                  options = model_options_list,
                                  index   = 0)
    st.write("""---""")
    #########################################################################
    #######################      NAIVE BAYES     ############################
    #########################################################################
    if selected_model == model_options_list[0]:
        st.write("1. MultinomialNB")
        st.caption("> For text classification problems, NB is always a goto for baseline model due to discrete features.\n> We are using multinomial NB here with smoothing.")
        st.latex(r'''
                    {\widehat\theta}_{yi} = \frac{N_{yi}+\alpha}{N_y+\alpha n}
                ''')
        st.caption("Where $N{yi}$ is the number of times feature appears in a sample of class in the training set & $N_{y}$ is the total count of all features for class.")

        # Fetching model specific details
        meta_dict = metadata[meta_dict_name_map["nb"]]
        correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
        incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
        loglosses = meta_dict["LogLoss"]
        saved_base_model_path = meta_dict["Models"]["base_classifier"]
        saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
        tune_dict = meta_dict["Tuning"]
        best_params = meta_dict["best_parameters"]

        hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

        with hyperparameter:
            if "hp_selected" not in st.session_state:
                st.session_state.hp_selected = "Alpha"

            hp_options, hp_details = st.columns([1,3])

            with hp_options:
                st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                         options = ["Alpha"])

            with hp_details:
                if st.session_state.hp_selected == "Alpha":
                    st.caption("""
                                1. Using higher alpha values will push the likelihood towards a value of 0.5.

                                2. It means the probability of a word equal to 0.5 for both the positive and negative reviews.

                                3. Alpha values also used for smoothing and not making the probability 0.""")  

        with tuning:
            # For comparison with random model
            train_loss = round(loglosses['Train-loss'],3)
            cv_loss = round(loglosses['CV-loss'],3)
            test_loss = round(loglosses['Test-loss'],3)

            st.write("Optimal LogLoss in comparion with Random model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Train Loss", value = train_loss)
            col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
            col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

            # plotting cv
            st.write("Plotting CV loss across Alpha.")
            alpha = sorted(tune_dict.keys())
            logloss = [ tune_dict[alp] for alp in tune_dict]
            logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": logloss})
            st.write(logloss_df.set_index('Alpha').T)

            # Graph Hyperparameter vs LogLoss
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(np.log10(alpha), logloss, c = 'g', marker = "o")
            for i, txt in enumerate(np.round(logloss,3)):
                ax.annotate(text=(alpha[i],str(txt)), xy=(np.log10(alpha[i]),logloss[i]), xytext=(np.log10(alpha[i])-0.4,logloss[i]+0.005), rotation=20)
            plt.xticks(np.log10(alpha))
            plt.title("Cross Validation (LogLoss) Error for different Alpha", pad=25)
            plt.xlabel("Alpha (10's power)")
            plt.ylabel("LogLoss")
            sns.set()
            plt.grid(visible=True)
            st.pyplot(fig, dpi=400)

        with features_details:
            clf = joblib.load(saved_base_model_path)
            sig_clf = joblib.load(saved_calibrated_model_path)

            logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
                                                                                                       train_y, \
                                                                                                       test_x_onehotCoding, \
                                                                                                       test_y, \
                                                                                                       clf)
            st.write("Confusion Matrix for test data points.")
            plot_confusion_matrix(test_y, pred_y)

            st.write("""---""")

            point_select,infer_details = st.columns([1,3])

            with point_select:
                which_type_to_check = st.radio(
                    label = "Which type of Point you want to validate.",
                    options = ('Correct', 'Misclassified'),
                    horizontal=True
                )
                st.session_state.point_to_check = which_type_to_check

            with infer_details:
                if st.session_state.point_to_check == 'Correct':
                    st.write("Checking Correctly classified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(correct_points),\
                                                     value = len(correct_points)//2)
                    test_point_index = correct_points[test_point_lst_index]


                elif st.session_state.point_to_check == "Misclassified" :
                    st.write("Checking Misclassified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(incorrect_points),\
                                                     value = len(incorrect_points)//2)
                    test_point_index = incorrect_points[test_point_lst_index]

                st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
                st.metric(label = "Actual Class", value = test_y[test_point_index])
                st.metric(label = "Predicted Class", value = predicted_cls[0])

                st.write("Classwise Probabilities:")
                pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                  "Prob":pred_class_probabilties[0]})
                st.dataframe(prob_df.set_index('Class').T)

                # inspection on feature presence
                no_feature = st.slider(label = 'Select top features to compare with: ',\
                                       min_value = 0,\
                                       max_value = test_x_onehotCoding.shape[1],\
                                       value = min(100,test_x_onehotCoding.shape[1]),\
                                       step = 25)
                indices=np.argsort(-1*clf.feature_log_prob_)[predicted_cls-1][:,:no_feature]
                get_impfeature_names(indices[0], \
                                     test_df['Text'].iloc[test_point_index],\
                                     test_df['Gene'].iloc[test_point_index],\
                                     test_df['Variation'].iloc[test_point_index], \
                                     no_feature, \
                                     train_df)

        with observations:
            # Writing best parameters
            st.write("These are the best parameter values:")
            st.write(best_params)

            # Write Ups
            st.write("Observations")
            st.caption("""
                        1. Random Model's Logloss: {0} & This model's Logloss {1}

                        2. As you can see logloss values are dropped drastically which is great.

                        3. The misclassfication rate is ~{2}% which is high.

                        4. Computationally it is very efficient & scalable. It works very well with text data.

                        5. The main drawback is it assumes that all the features are independent in nature which is never true.

                        6. MultinomialNB consider that frequency of word (tf) while creating feature vector while Burnoulli uses presence (BoW) to create feature vectors.

                        7. In general, classical text ML alpgorithms misses the semantic meaning of text data.""".\
                       format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

    #########################################################################
    #######################   K NEAREST NEIGHBORS  ##########################
    #########################################################################
    if selected_model == model_options_list[1]: 
        # Fetching model specific details
        meta_dict = metadata[meta_dict_name_map["knn"]]
        correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
        incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
        loglosses = meta_dict["LogLoss"]
        saved_base_model_path = meta_dict["Models"]["base_classifier"]
        saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
        tune_dict = meta_dict["Tuning"]
        best_params = meta_dict["best_parameters"]

        st.write("2. K Nearest Neighbors")
        st.caption("""
                    The Nearest Neighbors type of problem has 2 methods, setting threshold as 1) Radius & 2) number of points, around query point and Majority wins.

                    The KNN doesn't create learning model, but it remembers the data points distance (provided) proximity metric. That is why it is also known as lazy learner.

                    It is basically assigning weights which are near to query point where weights are uniformly distributed.

                    By default distance metric is Euclidean distance. 

                    As high dimensions data do not work well with distance metrics, we will select Response Coding featurized data.""")

        hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

        with hyperparameter:
            if "knn_hp_selected" not in st.session_state:
                st.session_state.knn_hp_selected = "K"

            hp_options, hp_details = st.columns([1,3])

            with hp_options:
                st.session_state.knn_hp_selected = st.radio(label = "Hyperparameters" ,
                                                         options = ["K"])

            with hp_details:
                if st.session_state.knn_hp_selected == "K":
                    st.caption("""
                               1. Larger the value of K, we'll haeve more point considerations.")

                               2. Too small value won't make generalization. Too large (or N itself) won't make it specific to.")

                               3. Default value is 5 neighbors.""")  

        with tuning:
            # For comparison with random model
            train_loss = round(loglosses['Train-loss'],3)
            cv_loss = round(loglosses['CV-loss'],3)
            test_loss = round(loglosses['Test-loss'],3)

            st.write("Optimal LogLoss in comparion with Random model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Train Loss", value = train_loss)
            col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
            col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

            # plotting cv
            st.write("Plotting CV loss across Alpha.")
            alpha = sorted(tune_dict.keys())
            logloss = [ tune_dict[alp] for alp in tune_dict]
            logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": logloss})
            st.write(logloss_df.set_index('Alpha').T)

            # Graph Hyperparameter vs LogLoss
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(alpha, logloss, c = 'g', marker = "o")
            for i, txt in enumerate(np.round(logloss,3)):
                ax.annotate(text=(alpha[i],str(txt)), xy=(alpha[i],logloss[i]), xytext=(alpha[i]+0.7,logloss[i]-0.005), rotation=0)
            # plt.xticks(np.log10(alpha))
            plt.title("Cross Validation (LogLoss) Error for different K", pad=15)
            plt.xlabel("K Neighbors")
            plt.ylabel("LogLoss")
            sns.set()
            plt.grid(visible=True)
            st.pyplot(fig, dpi=400)

        with features_details:
            clf = joblib.load(saved_base_model_path)
            sig_clf = joblib.load(saved_calibrated_model_path)

            logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_responseCoding, \
                                                                                                       train_y, \
                                                                                                       test_x_responseCoding, \
                                                                                                       test_y, \
                                                                                                       clf)
            st.write("Confusion Matrix for test data points.")
            plot_confusion_matrix(test_y, pred_y)

            st.write("""---""")

            point_select,infer_details = st.columns([1,3])

            with point_select:
                which_type_to_check = st.radio(
                    label = "Which type of Point you want to validate.",
                    options = ('Correct', 'Misclassified'),
                    horizontal=True
                )
                st.session_state.point_to_check = which_type_to_check

            with infer_details:
                if st.session_state.point_to_check == 'Correct':
                    st.write(">Checking Correctly classified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(correct_points),\
                                                     value = len(correct_points)//2)
                    test_point_index = correct_points[test_point_lst_index]


                elif st.session_state.point_to_check == "Misclassified" :
                    st.write(">Checking Misclassified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(incorrect_points),\
                                                     value = len(incorrect_points)//2)
                    test_point_index = incorrect_points[test_point_lst_index]

                st.caption("Selected Test-Point: {0}".format(test_point_index))
                choosen_point_features = test_x_responseCoding[test_point_index].reshape(1,-1)

                predicted_cls = sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))
                st.metric(label = "Actual Class", value = test_y[test_point_index])
                st.metric(label = "Predicted Class", value = predicted_cls[0])

                best_neighbors = best_params['n_neighbors']
                neighbors = clf.kneighbors(choosen_point_features, best_neighbors)
                # inspection on feature presence
                st.caption("""
                > Optimal value for nearest neighbors knn is {0} 

                > The nearest neighbours of the test points

                > \tClass: Number of point which belongs to this class

                >{1}""".\
                      format(best_neighbors,Counter(train_y[neighbors[1][0]])))

        with observations:
            # Writing best parameters
            st.write("These are the best parameter values:")
            st.write(best_params)

            # Write Ups
            st.write("Observations")
            st.caption("""
                        1. Random Model's Logloss: {0} & This model's Logloss {1}. As you can see logloss values are dropped drastically which is great.

                        2. The misclassfication rate is ~{2}% which is high.

                        3. Does not work well with high dimensions

                        4. Poor performance on imbalanced data

                        5. Sensitive to noisy data, missing values and outliers

                        6. Optimal value of K — If chosen incorrectly, the model will be under or overfitted to the data

                        7. Need feature scaling, results in additional-step computationally and less interpretable.""".\
                       format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

    #########################################################################
    ######################    LOGISTIC REGRESSION    ########################
    #########################################################################
    if selected_model == model_options_list[2]:
        st.write("3. Logistic Regression")
        st.caption("""
                    It is linear model.

                    It also provides probabilities of classes which is useful in our case.

                    We are using SGD classifier for training and choosing "LogLoss" as loss makes it probabilistic classifier Logistic Regression.""")

        st.latex(r'''
                    {W^* = argmin_w \sum_{n=1}^{i} [(-Y_i * log(P_i))-(1-Y_i)*(log(1-P_i)) + Regularization\_Term]}
                ''')
        st.caption("Where  $P_i = \sigma(W^T*X_i)\ \&\ Y_i \in \{+1,0\}$ ")

        is_balanced = st.radio( label   = "How class is balanced or not while modelling?",
                                options = ('Unbalanced', 'Balanced'),
                                index   = 0,
                                horizontal = True)

        if is_balanced == "Unbalanced":
            # Fetching model specific details
            meta_dict = metadata[meta_dict_name_map["lr_unbalanced"]]
            correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
            incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
            loglosses = meta_dict["LogLoss"]
            saved_base_model_path = meta_dict["Models"]["base_classifier"]
            saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
            tune_dict = meta_dict["Tuning"]
            best_params = meta_dict["best_parameters"]

            hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

            with hyperparameter:

                hp_options, hp_details = st.columns([1,3])

                with hp_options:
                    st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                             options = ["Alpha","Penalty"], 
                                                             index = 0)

                with hp_details:
                    if st.session_state.hp_selected == "Alpha":
                        st.caption("1. Using higher alpha values will push the likelihood towards a value of 0.5.")
                        st.caption("2. It means the probability of a word equal to 0.5 for both the positive and negative reviews.")
                        st.caption("3. Alpha values also used for smoothing and not making the probability 0.")

                    if st.session_state.hp_selected == "Penalty":
                        st.caption("1. The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector.")
                        st.caption("2. Avaiable options are 1) The squared euclidean norm L2 or 2) the absolute norm L1 or 3) a combination of both (Elastic Net).")

            with tuning:
                # For comparison with random model
                train_loss = round(loglosses['Train-loss'],3)
                cv_loss = round(loglosses['CV-loss'],3)
                test_loss = round(loglosses['Test-loss'],3)

                st.write("Optimal LogLoss in comparion with Random model")
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Train Loss", value = train_loss)
                col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
                col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

                # plotting cv
                st.write("Plotting CV loss across Alpha.")
                alpha = sorted(tune_dict.keys())
                logloss = [ tune_dict[alp] for alp in tune_dict]
                logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": logloss})
                st.write(logloss_df.set_index('Alpha').T)

                # Graph Hyperparameter vs LogLoss
                fig, ax = plt.subplots(figsize=(8,6))
                ax.plot(np.log10(alpha), logloss, c = 'g', marker = "o")
                for i, txt in enumerate(np.round(logloss,3)):
                    ax.annotate(text=(alpha[i],str(txt)), xy=(np.log10(alpha[i]),logloss[i]), xytext=(np.log10(alpha[i])-0.4,logloss[i]+0.005), rotation=20)
                plt.xticks(np.log10(alpha))
                plt.title("Cross Validation (LogLoss) Error for different Alpha", pad=25)
                plt.xlabel("Alpha (10's power)")
                plt.ylabel("LogLoss")
                sns.set()
                plt.grid(visible=True)
                st.pyplot(fig, dpi=400)

            with features_details:
                clf = joblib.load(saved_base_model_path)
                sig_clf = joblib.load(saved_calibrated_model_path)

                logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
                                                                                                           train_y, \
                                                                                                           test_x_onehotCoding, \
                                                                                                           test_y, \
                                                                                                           clf)
                st.write("Confusion Matrix for test data points.")
                plot_confusion_matrix(test_y, pred_y)

                st.write("""---""")

                point_select,infer_details = st.columns([1,3])

                with point_select:
                    which_type_to_check = st.radio(
                        label = "Which type of Point you want to validate.",
                        options = ('Correct', 'Misclassified'),
                        horizontal=True
                    )
                    st.session_state.point_to_check = which_type_to_check

                with infer_details:
                    if st.session_state.point_to_check == 'Correct':
                        st.write("Checking Correctly classified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(correct_points),\
                                                         value = len(correct_points)//2)
                        test_point_index = correct_points[test_point_lst_index]


                    elif st.session_state.point_to_check == "Misclassified" :
                        st.write("Checking Misclassified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(incorrect_points),\
                                                         value = len(incorrect_points)//2)
                        test_point_index = incorrect_points[test_point_lst_index]

                    st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                    predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
                    st.metric(label = "Actual Class", value = test_y[test_point_index])
                    st.metric(label = "Predicted Class", value = predicted_cls[0])

                    st.write("Classwise Probabilities:")
                    pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                    prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                      "Prob":pred_class_probabilties[0]})
                    st.dataframe(prob_df.set_index('Class').T)

                    # inspection on feature presence
                    no_feature = st.slider(label = 'Select top features to compare with: ',\
                                           min_value = 0,\
                                           max_value = test_x_onehotCoding.shape[1],\
                                           value = min(100,test_x_onehotCoding.shape[1]),\
                                           step = 25)
                    indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
                    get_impfeature_names(indices[0], \
                                         test_df['Text'].iloc[test_point_index],\
                                         test_df['Gene'].iloc[test_point_index],\
                                         test_df['Variation'].iloc[test_point_index], \
                                         no_feature, \
                                         train_df)

            with observations:
                # Writing best parameters
                st.write("These are the best parameter values:")
                st.write(best_params)

                # Write Ups
                st.write("Observations")
                st.caption("""
                       > Random Model's Logloss: {0} & This model's Logloss {1}

                       > As you can see logloss values are dropped drastically which is great.

                       > The misclassfication rate is ~{2}% which is high.""".\
                           format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))


        if is_balanced == "Balanced":
            # Fetching model specific details
            meta_dict = metadata[meta_dict_name_map["lr_balanced"]]
            correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
            incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
            loglosses = meta_dict["LogLoss"]
            saved_base_model_path = meta_dict["Models"]["base_classifier"]
            saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
            tune_dict = meta_dict["Tuning"]
            best_params = meta_dict["best_parameters"]


            hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

            with hyperparameter:
                hp_options, hp_details = st.columns([1,3])

                with hp_options:
                    st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                             options = ["Alpha","Penalty"], 
                                                             index = 0)

                with hp_details:
                    if st.session_state.hp_selected == "Alpha":
                        st.caption("1. Using higher alpha values will push the likelihood towards a value of 0.5.")
                        st.caption("2. It means the probability of a word equal to 0.5 for both the positive and negative reviews.")
                        st.caption("3. Alpha values also used for smoothing and not making the probability 0.")

                    if st.session_state.hp_selected == "Penalty":
                        st.caption("1. The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector.")
                        st.caption("2. Avaiable options are 1) The squared euclidean norm L2 or 2) the absolute norm L1 or 3) a combination of both (Elastic Net).")

            with tuning:
                # For comparison with random model
                train_loss = round(loglosses['Train-loss'],3)
                cv_loss = round(loglosses['CV-loss'],3)
                test_loss = round(loglosses['Test-loss'],3)

                st.write("Optimal LogLoss in comparion with Random model")
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Train Loss", value = train_loss)
                col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
                col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

                # plotting cv
                st.write("Plotting CV loss across Alpha.")
                alpha = sorted(tune_dict.keys())
                logloss = [ tune_dict[alp] for alp in tune_dict]
                logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": logloss})
                st.write(logloss_df.set_index('Alpha').T)

                # Graph Hyperparameter vs LogLoss
                fig, ax = plt.subplots(figsize=(8,6))
                ax.plot(np.log10(alpha), logloss, c = 'g', marker = "o")
                for i, txt in enumerate(np.round(logloss,3)):
                    ax.annotate(text=(alpha[i],str(txt)), xy=(np.log10(alpha[i]),logloss[i]), xytext=(np.log10(alpha[i])-0.4,logloss[i]+0.005), rotation=20)
                plt.xticks(np.log10(alpha))
                plt.title("Cross Validation (LogLoss) Error for different Alpha", pad=25)
                plt.xlabel("Alpha (10's power)")
                plt.ylabel("LogLoss")
                sns.set()
                plt.grid(visible=True)
                st.pyplot(fig, dpi=400)

            with features_details:
                clf = joblib.load(saved_base_model_path)
                sig_clf = joblib.load(saved_calibrated_model_path)

                logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
                                                                                                           train_y, \
                                                                                                           test_x_onehotCoding, \
                                                                                                           test_y, \
                                                                                                           clf)
                st.write("Confusion Matrix for test data points.")
                plot_confusion_matrix(test_y, pred_y)

                st.write("""---""")

                point_select,infer_details = st.columns([1,3])

                with point_select:
                    which_type_to_check = st.radio(
                        label = "Which type of Point you want to validate.",
                        options = ('Correct', 'Misclassified'),
                        horizontal=True
                    )
                    st.session_state.point_to_check = which_type_to_check

                with infer_details:
                    if st.session_state.point_to_check == 'Correct':
                        st.write("Checking Correctly classified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(correct_points),\
                                                         value = len(correct_points)//2)
                        test_point_index = correct_points[test_point_lst_index]


                    elif st.session_state.point_to_check == "Misclassified" :
                        st.write("Checking Misclassified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(incorrect_points),\
                                                         value = len(incorrect_points)//2)
                        test_point_index = incorrect_points[test_point_lst_index]

                    st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                    predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
                    st.metric(label = "Actual Class", value = test_y[test_point_index])
                    st.metric(label = "Predicted Class", value = predicted_cls[0])

                    st.write("Classwise Probabilities:")
                    pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                    prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                      "Prob":pred_class_probabilties[0]})
                    st.dataframe(prob_df.set_index('Class').T)

                    # inspection on feature presence
                    no_feature = st.slider(label = 'Select top features to compare with: ',\
                                           min_value = 0,\
                                           max_value = test_x_onehotCoding.shape[1],\
                                           value = min(100,test_x_onehotCoding.shape[1]),\
                                           step = 25)
                    indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
                    get_impfeature_names(indices[0], \
                                         test_df['Text'].iloc[test_point_index],\
                                         test_df['Gene'].iloc[test_point_index],\
                                         test_df['Variation'].iloc[test_point_index], \
                                         no_feature, \
                                         train_df)

            with observations:
                # Writing best parameters
                st.write("These are the best parameter values:")
                st.write(best_params)

                # Write Ups
                st.write("Observations")
                st.caption("""
                       > Random Model's Logloss: {0} & This model's Logloss {1}

                       > As you can see logloss values are dropped drastically which is great.

                       > The misclassfication rate is ~{2}% which is high.""".\
                           format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

    #########################################################################
    #################     SUPPORT VECTOR MACHINE     ########################
    #########################################################################
    if selected_model == model_options_list[3]:
        st.write("4. Support Vector Machine (SVM)")
        st.caption("""
                    SVM is vesatile function which can be used for classification, regression and Outlier detection.

                    SVM is chooses subset of training datapoints as points to create class boundries, which is called Support Vectors. Which makes it memory efficient.

                    It doesn't have problem with High dimensional data.

                    The key feature it provide is tobe selective about the Kernel function along with your custom kernel as well. We are using Radial Basis Function (RBF) kernel which generally work best with any problems. You can found more details sklean documentation.

                    We are using SGD with Hingeloss ${max(0, 1-z_i)}$, which is same as SVM.
                    """)
        st.latex(r'''
                    {W^*, b^* = argmin_{(w,b)} [ C * \frac{1}{n} \sum_{i=1}^{i=n} (1-Y_i (W^TX_i+b)  + \frac{||W||_2}{2} )]}
                ''')
        st.caption(" $\zeta_i = 1-Dj\ \  where\  C\ , \zeta_i \ge 0$")

        # Fetching model specific details
        meta_dict = metadata[meta_dict_name_map["svm"]]
        correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
        incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
        loglosses = meta_dict["LogLoss"]
        saved_base_model_path = meta_dict["Models"]["base_classifier"]
        saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
        tune_dict = meta_dict["Tuning"]
        best_params = meta_dict["best_parameters"]

        hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

        with hyperparameter:
            if "hp_selected" not in st.session_state:
                st.session_state.hp_selected = "C"

            hp_options, hp_details = st.columns([1,3])

            with hp_options:
                    st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                             options = ["C","Class Weight","Penalty"], 
                                                             index = 0)

            with hp_details:
                if st.session_state.hp_selected == "C":
                    st.caption("1. When using SVM kernel, C is common hyperparameter across all.")
                    st.caption("2. The parameter C trades off misclassification of training examples against simplicity of the decision surface.")
                    st.caption("3. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly.")

                if st.session_state.hp_selected == "Class Weight":
                    st.caption("1. Weights associated with classes. If not given, all classes are supposed to have equal weights.")
                    st.caption("2. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as $n\_samples / (n\_classes * np.bincount(y))$")

                if st.session_state.hp_selected == "Penalty":
                    st.caption("1. The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector.")
                    st.caption("2. Avaiable options are 1) The squared euclidean norm L2 or 2) the absolute norm L1 or 3) a combination of both (Elastic Net).")

        with tuning:
            # For comparison with random model
            train_loss = round(loglosses['Train-loss'],3)
            cv_loss = round(loglosses['CV-loss'],3)
            test_loss = round(loglosses['Test-loss'],3)

            st.write("Optimal LogLoss in comparion with Random model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Train Loss", value = train_loss)
            col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
            col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

            # plotting cv
            st.write("Plotting CV loss across C.")
            alpha = sorted(tune_dict.keys())
            logloss = [ tune_dict[alp] for alp in tune_dict]
            logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": logloss})
            st.write(logloss_df.set_index('Alpha').T)

            # Graph Hyperparameter vs LogLoss
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(np.log10(alpha), logloss, c = 'g', marker = "o")
            for i, txt in enumerate(np.round(logloss,3)):
                ax.annotate(text=(alpha[i],str(txt)), xy=(np.log10(alpha[i]),logloss[i]), xytext=(np.log10(alpha[i])+0.1,logloss[i]+0.005), rotation=10)
            plt.xticks(np.log10(alpha))
            plt.title("Cross Validation (LogLoss) Error for different C", pad=25)
            plt.xlabel("C (10's power)")
            plt.ylabel("LogLoss")
            sns.set()
            plt.grid(visible=True)
            st.pyplot(fig, dpi=400)

        with features_details:
            clf = joblib.load(saved_base_model_path)
            sig_clf = joblib.load(saved_calibrated_model_path)

            logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
                                                                                                       train_y, \
                                                                                                       test_x_onehotCoding, \
                                                                                                       test_y, \
                                                                                                       clf)
            st.write("Confusion Matrix for test data points.")
            plot_confusion_matrix(test_y, pred_y)

            st.write("""---""")

            point_select,infer_details = st.columns([1,3])

            with point_select:
                which_type_to_check = st.radio(
                    label = "Which type of Point you want to validate.",
                    options = ('Correct', 'Misclassified'),
                    horizontal=True
                )
                st.session_state.point_to_check = which_type_to_check

            with infer_details:
                if st.session_state.point_to_check == 'Correct':
                    st.write("Checking Correctly classified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(correct_points),\
                                                     value = len(correct_points)//2)
                    test_point_index = correct_points[test_point_lst_index]


                elif st.session_state.point_to_check == "Misclassified" :
                    st.write("Checking Misclassified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(incorrect_points),\
                                                     value = len(incorrect_points)//2)
                    test_point_index = incorrect_points[test_point_lst_index]

                st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
                st.metric(label = "Actual Class", value = test_y[test_point_index])
                st.metric(label = "Predicted Class", value = predicted_cls[0])

                st.write("Classwise Probabilities:")
                pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                  "Prob":pred_class_probabilties[0]})
                st.dataframe(prob_df.set_index('Class').T)

                # inspection on feature presence
                no_feature = st.slider(label = 'Select top features to compare with: ',\
                                       min_value = 0,\
                                       max_value = test_x_onehotCoding.shape[1],\
                                       value = min(100,test_x_onehotCoding.shape[1]),\
                                       step = 25)
                indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
                get_impfeature_names(indices[0], \
                                     test_df['Text'].iloc[test_point_index],\
                                     test_df['Gene'].iloc[test_point_index],\
                                     test_df['Variation'].iloc[test_point_index], \
                                     no_feature, \
                                     train_df)

        with observations:
            # Writing best parameters
            st.write("These are the best parameter values:")
            st.write(best_params)

            # Write Ups
            st.write("Observations")
            st.caption("""
                       > Random Model's Logloss: {0} & This model's Logloss {1}

                       > As you can see logloss values are dropped drastically which is great.

                       > The misclassfication rate is ~{2}% which is high.""".\
                       format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

    #########################################################################
    #########################    RANDOM FOREST    ###########################
    #########################################################################
    if selected_model == model_options_list[4]:
        st.write("5. Random Forest Classifier")
        st.caption("""
                    Random Forest is based on the bagging algorithm and uses Ensemble Learning technique. It creates as many trees on the subset of the data and combines the output of all the trees. In this way it reduces overfitting problem in decision trees and also reduces the variance and therefore improves the accuracy.



                    However, the interpretability suffers the most as we are ensemble the several weak classifiers to generate final output

                    It works like a charm as we are learning on sub set of data, it is time efficient, but memory intensive.

                    As a single classifier is working with subset of data with row & column sampled data, high dimensions are also no problem for RF.
                    """)

        selected_feature_set = st.radio( label = "Which feature set do you want use?",
                                options = ('One Hot Encoded', 'Response Coding'),
                                index   = 0,
                                horizontal = True)
        st.caption("RF models takes more time to load due to its size... please be patient.")
        if selected_feature_set == "One Hot Encoded":
            # Fetching model specific details
            meta_dict = metadata[meta_dict_name_map["rf_ohe"]]
            correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
            incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
            loglosses = meta_dict["LogLoss"]
            saved_base_model_path = meta_dict["Models"]["base_classifier"]
            saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
            tune_dict = meta_dict["Tuning"]
            best_params = meta_dict["best_parameters"]

            hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

            with hyperparameter:

                hp_options, hp_details = st.columns([1,3])

                with hp_options:
                    st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                             options = ["estimators","max_depth","Others"], 
                                                             index = 0)

                with hp_details:
                    if st.session_state.hp_selected == "estimators":
                        st.caption("1. The number of trees in the forest")
                        st.caption("2. Default value is 100, which means it will create 100 classifier with subset data feeded to them.")
                        st.caption("3. At the end, max vote will decide the outcome of perticular point.")
                        st.caption("4. As this are randomly generated samples, the variance reduces.")


                    if st.session_state.hp_selected == "max_depth":
                        st.caption("1. The maximum depth of each weak classifier tree.")
                        st.caption("2. This is necessary parameter as if not used it will overfit and not able to generalize the data.")

                    if st.session_state.hp_selected == "Others":
                        st.caption("We are using Gini impurity measure, which is used to split at every the nodes.")
                        st.caption("By setting n_jobs = -1, we are optimizing by creating parellel executions of multiple trees, based on cores available.")

            with tuning:
                # For comparison with random model
                train_loss = round(loglosses['Train-loss'],3)
                cv_loss = round(loglosses['CV-loss'],3)
                test_loss = round(loglosses['Test-loss'],3)

                st.write("Optimal LogLoss in comparion with Random model")
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Train Loss", value = train_loss)
                col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
                col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

                # plotting cv
                x = [val[0] for val in tune_dict]
                y = [val[1] for val in tune_dict]
                z = [val[2] for val in tune_dict]

                z_data = pd.DataFrame({"n_estimators":x,"max_depth":y,"LogLoss":z})
                fig = px.line_3d(z_data, x="n_estimators", 
                                 y="max_depth", 
                                 z="LogLoss",
                                 markers=True,
                                 title="LogLoss with RF Hyperparameter n_estimator & max_depth",
                                 height= 500,
                                 width = 500
                                )
                st.plotly_chart(fig, sharing="streamlit", theme="streamlit", dpi=400)

            with features_details:
                clf = joblib.load(saved_base_model_path)
                sig_clf = joblib.load(saved_calibrated_model_path)

                logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
                                                                                                           train_y, \
                                                                                                           test_x_onehotCoding, \
                                                                                                           test_y, \
                                                                                                           clf)
                st.write("Confusion Matrix for test data points.")
                plot_confusion_matrix(test_y, pred_y)

                st.write("""---""")

                point_select,infer_details = st.columns([1,3])

                with point_select:
                    which_type_to_check = st.radio(
                        label = "Which type of Point you want to validate.",
                        options = ('Correct', 'Misclassified'),
                        horizontal=True
                    )
                    st.session_state.point_to_check = which_type_to_check

                with infer_details:
                    if st.session_state.point_to_check == 'Correct':
                        st.write("Checking Correctly classified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(correct_points),\
                                                         value = len(correct_points)//2)
                        test_point_index = correct_points[test_point_lst_index]


                    elif st.session_state.point_to_check == "Misclassified" :
                        st.write("Checking Misclassified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(incorrect_points),\
                                                         value = len(incorrect_points)//2)
                        test_point_index = incorrect_points[test_point_lst_index]

                    st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                    predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
                    st.metric(label = "Actual Class", value = test_y[test_point_index])
                    st.metric(label = "Predicted Class", value = predicted_cls[0])

                    st.write("Classwise Probabilities:")
                    pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                    prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                      "Prob":pred_class_probabilties[0]})
                    st.dataframe(prob_df.set_index('Class').T)

                    # inspection on feature presence
                    no_feature = st.slider(label = 'Select top features to compare with: ',\
                                           min_value = 0,\
                                           max_value = test_x_onehotCoding.shape[1],\
                                           value = min(100,test_x_onehotCoding.shape[1]),\
                                           step = 25)
                    indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]
                    get_impfeature_names(indices[0], \
                                         test_df['Text'].iloc[test_point_index],\
                                         test_df['Gene'].iloc[test_point_index],\
                                         test_df['Variation'].iloc[test_point_index], \
                                         no_feature, \
                                         train_df)

            with observations:
                # Writing best parameters
                st.write("These are the best parameter values:")
                st.write(best_params)

                # Write Ups
                st.write("Observations")
                st.caption("""
                    Random Model's Logloss: {0} & This model's Logloss {1}. As you can see logloss values are dropped drastically which is great.

                    The misclassfication rate is ~{2}% which is high.

                    > Advantages of Random Forest

                    1. Random Forest can automatically handle missing values & usually robust to outliers and can handle them automatically..

                    2. No feature scaling required: No feature scaling (standardization and normalization) required in case of Random Forest as it uses rule based approach instead of distance calculation.

                    3. Handles non-linear parameters efficiently: Non linear parameters don't affect the performance of a Random Forest unlike curve based algorithms. So, if there is high non-linearity between the independent variables, Random Forest may outperform as compared to other curve based algorithms.

                    4. Random Forest algorithm is very stable. Even if a new data point is introduced in the dataset, the overall algorithm is not affected much since the new data may impact one tree, but it is very hard for it to impact all the trees.

                    > Disadvantages of Random Forest

                    1. Complexity: Random Forest creates a lot of trees (unlike only one tree in case of decision tree) and combines their outputs. By default, it creates 100(default value) trees in Python sklearn library. To do so, this algorithm requires much more computational power and resources. On the other hand decision tree is simple and does not require so much computational resources.

                    2. Longer Training Period: Random Forest require much more time to train as compared to decision trees as it generates a lot of trees (instead of one tree in case of decision tree) and makes decision on the majority of votes.
                       """.format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

        if selected_feature_set == "Response Coding":
            # Fetching model specific details
            meta_dict = metadata[meta_dict_name_map["rf_rc"]]
            correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
            incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
            loglosses = meta_dict["LogLoss"]
            saved_base_model_path = meta_dict["Models"]["base_classifier"]
            saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
            tune_dict = meta_dict["Tuning"]
            best_params = meta_dict["best_parameters"]

            hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

            with hyperparameter:

                hp_options, hp_details = st.columns([1,3])

                with hp_options:
                    st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                             options = ["estimators","max_depth","Others"], 
                                                             index = 0)

                with hp_details:
                    if st.session_state.hp_selected == "estimators":
                        st.caption("1. The number of trees in the forest")
                        st.caption("2. Default value is 100, which means it will create 100 classifier with subset data feeded to them.")
                        st.caption("3. At the end, max vote will decide the outcome of perticular point.")
                        st.caption("4. As this are randomly generated samples, the variance reduces.")


                    if st.session_state.hp_selected == "max_depth":
                        st.caption("1. The maximum depth of each weak classifier tree.")
                        st.caption("2. This is necessary parameter as if not used it will overfit and not able to generalize the data.")

                    if st.session_state.hp_selected == "Others":
                        st.caption("We are using Gini impurity measure, which is used to split at every the nodes.")
                        st.caption("By setting n_jobs = -1, we are optimizing by creating parellel executions of multiple trees, based on cores available.")

            with tuning:
                # For comparison with random model
                train_loss = round(loglosses['Train-loss'],3)
                cv_loss = round(loglosses['CV-loss'],3)
                test_loss = round(loglosses['Test-loss'],3)

                st.write("Optimal LogLoss in comparion with Random model")
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Train Loss", value = train_loss)
                col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
                col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

                # plotting cv
                x = [val[0] for val in tune_dict]
                y = [val[1] for val in tune_dict]
                z = [val[2] for val in tune_dict]

                z_data = pd.DataFrame({"n_estimators":x,"max_depth":y,"LogLoss":z})
                fig = px.line_3d(z_data, x="n_estimators", 
                                 y="max_depth", 
                                 z="LogLoss",
                                 markers=True,
                                 title="LogLoss with RF Hyperparameter n_estimator & max_depth",
                                 height= 500,
                                 width = 500
                                )
                st.plotly_chart(fig, sharing="streamlit", theme="streamlit", dpi=400)

            with features_details:
                clf = joblib.load(saved_base_model_path)
                sig_clf = joblib.load(saved_calibrated_model_path)

                logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
                                                                                                           train_y, \
                                                                                                           test_x_onehotCoding, \
                                                                                                           test_y, \
                                                                                                           clf)
                st.write("Confusion Matrix for test data points.")
                plot_confusion_matrix(test_y, pred_y)

                st.write("""---""")

                point_select,infer_details = st.columns([1,3])

                with point_select:
                    which_type_to_check = st.radio(
                        label = "Which type of Point you want to validate.",
                        options = ('Correct', 'Misclassified'),
                        horizontal=True
                    )
                    st.session_state.point_to_check = which_type_to_check

                with infer_details:
                    if st.session_state.point_to_check == 'Correct':
                        st.write("Checking Correctly classified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(correct_points),\
                                                         value = len(correct_points)//2)
                        test_point_index = correct_points[test_point_lst_index]


                    elif st.session_state.point_to_check == "Misclassified" :
                        st.write("Checking Misclassified point.")
                        test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                         min_value = 0,\
                                                         max_value = len(incorrect_points),\
                                                         value = len(incorrect_points)//2)
                        test_point_index = incorrect_points[test_point_lst_index]

                    st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                    predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])
                    st.metric(label = "Actual Class", value = test_y[test_point_index])
                    st.metric(label = "Predicted Class", value = predicted_cls[0])

                    st.write("Classwise Probabilities:")
                    pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                    prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                      "Prob":pred_class_probabilties[0]})
                    st.dataframe(prob_df.set_index('Class').T)

                    # inspection on feature presence on 27 features 
                    # no_feature = st.slider(label = 'Select top features to compare with: ',\
                    #                       min_value = 0,\
                    #                       max_value = test_x_onehotCoding.shape[1],\
                    #                       value = min(100,test_x_onehotCoding.shape[1]),\
                    #                       step = 25)
                    # indices=np.argsort(-1*clf.coef_)[predicted_cls-1][:,:no_feature]

                    indices = np.argsort(-clf.feature_importances_)
                    st.write("""---""")
                    for i in indices:
                        if i<9:
                            st.caption("Gene is important feature".format(i))
                        elif i<18:
                            st.caption("Variation is important feature".format(i))
                        else:
                            st.caption("Text is important feature".format(i))


            with observations:
                # Writing best parameters
                st.write("These are the best parameter values:")
                st.write(best_params)

                # Write Ups
                st.write("Observations")
                st.caption("""
                       > Random Model's Logloss: {0} & This model's Logloss {1}

                       > As you can see logloss values are dropped drastically which is great.

                       > As you can see the difference between trainloss with CV & Test loss is higher than previous models, we can see that it is overfitting aggrasively.

                       > Also note that less number of features 

                       > The misclassfication rate is ~{2}% which is high.
                       """.\
                           format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

    #########################################################################
    ################     STACKED CLASSIFIERS     ############################
    #########################################################################
    if selected_model == model_options_list[5]:
        st.write("6. StackedClassifier (Ensemble)")
        st.caption("Stacked Classifier is meta classifier of multiple DIFFERENT(unlike RF) models.")
        st.caption("Final decider/meta classifier model, we have selected Logistic regression which will have hyperparameter of C.")

        st.write("<img src=\"https://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier_files/stacking_cv_classification_overview.png\">",unsafe_allow_html=True)

        # Fetching model specific details
        meta_dict = metadata[meta_dict_name_map["stacked"]]
        correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
        incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
        loglosses = meta_dict["LogLoss"]
        saved_base_model_path = meta_dict["Models"]["base_classifier"]
        # saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
        tune_dict = meta_dict["Tuning"]
        best_params = meta_dict["best_parameters"]
        base_models = meta_dict["base_classifiers"]

        hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

        with hyperparameter:
            hp_options, hp_details = st.columns([1,3])

            with hp_options:
                st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                        options = ["C","Others"],
                                                        index=0)

            with hp_details:
                if st.session_state.hp_selected == "C":
                    st.caption("1. When using SVM kernel, C is common hyperparameter across all.")
                    st.caption("2. The parameter C trades off misclassification of training examples against simplicity of the decision surface.")
                    st.caption("3. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly.")
                if st.session_state.hp_selected == "Others":
                    st.caption("Base classifier Hyperparameter is alpha for classifiers: 1) Logistic Regression, 2) SVM, 3) MultinomialNB")
                    

        with tuning:
            # For comparison with random model
            train_loss = round(loglosses['Train-loss'],3)
            cv_loss = round(loglosses['CV-loss'],3)
            test_loss = round(loglosses['Test-loss'],3)

            st.write("Optimal LogLoss in comparion with Random model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Train Loss", value = train_loss)
            col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
            col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')

            # plotting cv
            st.write("Plotting CV loss across Alpha.")
            alpha = [tup[0] for tup in tune_dict ]
            logloss = [tup[2] for tup in tune_dict ]
            logloss_df = pd.DataFrame({"Alpha":alpha, "CV LogLoss": logloss})
            st.write(logloss_df.set_index('Alpha').T)

            # Graph Hyperparameter vs LogLoss
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(np.log10(alpha), logloss, c = 'g', marker = "o")
            for i, txt in enumerate(np.round(logloss,3)):
                ax.annotate(text=(alpha[i],str(txt)), xy=(np.log10(alpha[i]),logloss[i]), xytext=(np.log10(alpha[i])-0.4,logloss[i]+0.005), rotation=20)
            plt.xticks(np.log10(alpha))
            plt.title("Cross Validation (LogLoss) Error for different C", pad=25)
            plt.xlabel("C (10's power)")
            plt.ylabel("LogLoss")
            sns.set()
            plt.grid(visible=True)
            st.pyplot(fig, dpi=400)

        with features_details:
            clf = joblib.load(saved_base_model_path)
            # sig_clf = joblib.load(saved_calibrated_model_path)
            # logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
            #                                                                                            train_y, \
            #                                                                                            test_x_onehotCoding, \
            #                                                                                            test_y, \

            sig_clf_probs = clf.predict_proba(test_x_onehotCoding)
            pred_y = clf.predict(test_x_onehotCoding)

            st.write("Confusion Matrix for test data points.")
            plot_confusion_matrix(test_y, pred_y)

            st.write("""---""")

            point_select,infer_details = st.columns([1,3])

            with point_select:
                which_type_to_check = st.radio(
                    label = "Which type of Point you want to validate.",
                    options = ('Correct', 'Misclassified'),
                    horizontal=True
                )
                st.session_state.point_to_check = which_type_to_check

            with infer_details:
                if st.session_state.point_to_check == 'Correct':
                    st.write("Checking Correctly classified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(correct_points),\
                                                     value = len(correct_points)//2)
                    test_point_index = correct_points[test_point_lst_index]


                elif st.session_state.point_to_check == "Misclassified" :
                    st.write("Checking Misclassified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(incorrect_points),\
                                                     value = len(incorrect_points)//2)
                    test_point_index = incorrect_points[test_point_lst_index]

                st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                predicted_cls = clf.predict(test_x_onehotCoding[test_point_index])
                st.metric(label = "Actual Class", value = test_y[test_point_index])
                st.metric(label = "Predicted Class", value = predicted_cls[0])

                st.write("Classwise Probabilities:")
                pred_class_probabilties = np.round(clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                  "Prob":pred_class_probabilties[0]})
                st.dataframe(prob_df.set_index('Class').T)

        with observations:
            # Base classifiers details
            st.write("These are the base classifier details:")
            st.write(base_models)

            # Writing best parameters
            st.write("These are the best meta LR classifier values:")
            st.write(best_params)

            # Write Ups
            st.write("Observations")
            st.caption("""
                        1. Random Model's Logloss: {0} & This model's Logloss {1}

                        2. As you can see logloss values are dropped drastically which is great.

                        3. The misclassfication rate is ~{2}% which is very high.""".\
                       format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

    #########################################################################
    ################     VOTING CLASSIFIER     ############################
    #########################################################################
    if selected_model == model_options_list[6]:
        st.write("7. MaxVoting Classifier (Ensemble)")
        st.caption("This is simplest of ensemble of different base classifier where it use a majority vote or the average predicted probabilities (soft vote) to predict the class labels.")
        st.caption("Such a classifier can be useful for a set of equally well performing models in order to balance out their individual weaknesses.")

        # Fetching model specific details
        meta_dict = metadata[meta_dict_name_map["voting"]]
        correct_points = meta_dict['Classified-Test-Datapoints']['Correct']
        incorrect_points = meta_dict['Classified-Test-Datapoints']['Misclassified']
        loglosses = meta_dict["LogLoss"]
        saved_base_model_path = meta_dict["Models"]["base_classifier"]
        best_params = meta_dict["best_parameters"]
        # below keys are not present for this classifier - expected
        # saved_calibrated_model_path = meta_dict["Models"]["calibrated_classifier"]
        # tune_dict = meta_dict["Tuning"]
        # base_models = meta_dict["base_classifiers"]

        hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & Deepdive", "Observations"])

        with hyperparameter:
            hp_options, hp_details = st.columns([1,3])

            with hp_options:
                st.session_state.hp_selected = st.radio(label = "Hyperparameters" ,
                                                        options = ["Voting","Others"],
                                                        index=0)

            with hp_details:
                if st.session_state.hp_selected == "Voting":
                    st.caption("It takes one of the two values {‘hard’, ‘soft’}.")
                    st.caption("1. If ‘hard’, uses predicted class labels for majority rule voting. ")
                    st.caption("2. If ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.")

                if st.session_state.hp_selected == "Others":
                    st.caption("""
                               We are using base estimators so tuning can be done accordingly: 
                               
                               1: 'Logistic Regression' 
                               
                               2: 'Support Vector Classifier' & 
                               
                               3: 'Random Forest'""")

        with tuning:
            st.caption("There is no tuning of meta classifier as base classifier's probabilities are used to vote.")
            # For comparison with random model
            train_loss = round(loglosses['Train-loss'],3)
            cv_loss = round(loglosses['CV-loss'],3)
            test_loss = round(loglosses['Test-loss'],3)

            st.write("Optimal LogLoss in comparision with Random model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Train Loss", value = train_loss)
            col2.metric(label="Cross-Validation Loss",  value = cv_loss, delta = round(cv_loss-Random_CV_Logloss,3) , delta_color='inverse')
            col3.metric(label="Test Loss",  value = test_loss, delta =  round(test_loss - Random_Test_Logloss,3),delta_color='inverse')


        with features_details:
            clf = joblib.load(saved_base_model_path)
            # sig_clf = joblib.load(saved_calibrated_model_path)
            # logloss, misclassfied_points, pred_y = report_log_loss_and_misclassified_points_and_pred_y(train_x_onehotCoding, \
            #                                                                                            train_y, \
            #                                                                                            test_x_onehotCoding, \
            #                                                                                            test_y, clf)

            sig_clf_probs = clf.predict_proba(test_x_onehotCoding)
            pred_y = clf.predict(test_x_onehotCoding)

            st.write("Confusion Matrix for test data points.")
            plot_confusion_matrix(test_y, pred_y)

            st.write("""---""")

            point_select,infer_details = st.columns([1,3])

            with point_select:
                which_type_to_check = st.radio(
                    label = "Which type of Point you want to validate.",
                    options = ('Correct', 'Misclassified'),
                    horizontal=True
                )
                st.session_state.point_to_check = which_type_to_check

            with infer_details:
                if st.session_state.point_to_check == 'Correct':
                    st.write("Checking Correctly classified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(correct_points),\
                                                     value = len(correct_points)//2)
                    test_point_index = correct_points[test_point_lst_index]


                elif st.session_state.point_to_check == "Misclassified" :
                    st.write("Checking Misclassified point.")
                    test_point_lst_index = st.slider(label = 'Pick a Point?',\
                                                     min_value = 0,\
                                                     max_value = len(incorrect_points),\
                                                     value = len(incorrect_points)//2)
                    test_point_index = incorrect_points[test_point_lst_index]

                st.caption("Selected Test-Point: {0}".format(test_point_lst_index))

                predicted_cls = clf.predict(test_x_onehotCoding[test_point_index])
                st.metric(label = "Actual Class", value = test_y[test_point_index])
                st.metric(label = "Predicted Class", value = predicted_cls[0])

                st.write("Classwise Probabilities:")
                pred_class_probabilties = np.round(clf.predict_proba(test_x_onehotCoding[test_point_index]),4)
                prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                                  "Prob":pred_class_probabilties[0]})
                st.dataframe(prob_df.set_index('Class').T)

        with observations:
            # Base classifiers details
            # st.write("These are the base classifier details:")
            # st.write(base_models)

            # Writing best parameters
            st.write("These are the best meta LR classifier values:")
            st.write(best_params)

            # Write Ups
            st.write("Observations")
            st.caption("""
                        1. Random Model's Logloss: {0} & This model's Logloss {1}. As you can see logloss values are dropped drastically which is great.

                        2. The misclassfication rate is ~{2}% which is very high.

                        3. Class imbalance is problematic for such classifier""".\
                       format(Random_Test_Logloss, round(best_params["LogLoss"],3), round(best_params["Misclassified-Percent"],3)*100))

else:
    st.caption("❗For Modelling, please run Jupyter Notebook named : \"2. Modelling & Tuning.ipynb\" in notebooks folder.❗")
    st.caption("❗We will use the metadata collected throughout the train-tune-testing.❗")
    st.caption("❗It take alot of train time and here we want to show the analysis and model details of fune-tune models.❗")
    
    st.caption("Running Notebook... It will take about 45 mins... to generate it first time only.")
    if not is_running(model_py_path) and os.path.exists(model_py_path):
        subprocess.Popen(model_py_path, shell=True)
        
st.write("""---""")