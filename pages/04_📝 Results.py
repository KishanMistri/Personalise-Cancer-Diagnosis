import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
# Importing utility functions from parent directory. Can't keep in pages as it creates menu in sidebar here.
import sys 
sys.path.insert(0, '../')
from utilities import *

st.title("📝 Results")
st.sidebar.markdown("# 📝 Results ")

# Variable Setup
MODELS_DIR = "./models/"

st.write("""---""")
# If metadata exists then only render page else note for directions
if os.path.exists(MODELS_DIR+"models_metadata.pkl"):
    # Metadata imports
    metadata = joblib.load(MODELS_DIR+"models_metadata.pkl")
    Random_CV_Logloss   = round(2.450964298929681,3)
    Random_Test_Logloss = round(2.5194404639993384,3)
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

    st.subheader("Summary")
    # Printing df
    models_names = ["Random Model"]
    train_loss = [Random_CV_Logloss]
    cv_loss = [Random_CV_Logloss]
    test_loss = [Random_Test_Logloss]
    mis_class_rate = [50]

    for model in meta_dict_name_map:
        m_dict = metadata[meta_dict_name_map[model]]
        models_names.append(meta_dict_name_map[model])
        train_loss.append(round(m_dict['LogLoss']['Train-loss'],3))
        cv_loss.append(round(m_dict['LogLoss']['CV-loss'],3))
        test_loss.append(round(m_dict['LogLoss']['Test-loss'],3))
        mis_class_rate.append(round(m_dict['best_parameters']['Misclassified-Percent']*100,2))

    #|| Model Name || Train-Loss || CV-Loss || Test-Loss || MisClassification-Rate ||
    table = pd.DataFrame({"Model Name": models_names,
                          "Train Loss": train_loss,
                          "Cross-Validation Loss": cv_loss,
                          "Test Loss": test_loss,
                          "Mis-Classification Rate (%)":mis_class_rate,
                         }).set_index("Model Name").style.background_gradient(cmap='RdYlGn_r') # coolwarm
    # table.style.format('{:.3f}')
    st.dataframe(table,use_container_width=True)

    # For comparison with random model
    idx = np.argmin(np.array(test_loss))
    st.write("Optimal LogLoss in comparion with Random model")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Train Loss", value = train_loss[idx])
    col2.metric(label="Cross-Validation Loss",  value = cv_loss[idx], delta = round(cv_loss[idx]-Random_CV_Logloss,3) , delta_color='inverse')
    col3.metric(label="Test Loss",  value = test_loss[idx], delta =  round(test_loss[idx] - Random_Test_Logloss,3),delta_color='inverse')
    
    st.write("""---""")

    st.subheader("Final Thoughts")
    st.caption("""
                > From above table, we can see that Random Forest Model is the clear winner by metric evaluation. However, we need to take many production level constrain into account prior to deploy the just best metric response model. However, Interpretibility suffers due to it being ensemble.

                > Random Forest require much more time to train as compared to decision trees as it generates a lot of trees (instead of one tree in case of decision tree) and makes decision on the majority of votes. This is acceptable as we don't have constrain of latency for our usecase. 

                > Random Forest creates a lot of trees (unlike only one tree in case of decision tree) and combines their outputs. Which requires much more computational power and resources. On the other hand decision tree is simple and does not require so much computational resources. 
                
                > In addition to this, if we are using multiple platform where the model size might be a concern then RF models are tend to be larger model. We need to use advance methods to make it suitable for devices. 
               """)

else:
    st.caption("❗For Modelling & Results, please run Jupyter Notebook named : \"2. Modelling & Tuning.ipynb\" in notebooks folder, which is used to generate the metadata to render this page first.❗")
    
st.write("""---""")
