
#########################################################################
#######################      NAIVE BAYES     ############################
#########################################################################
with selected_model == "Naive Bayes":
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
    
    
    Hyperparameter, Tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & deepdive", "Observations"])
    
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
        st.caption("As you can see logloss values are dropped drastically which is great.")
        st.caption("The misclassfication rate is ~44.67% which is high.")

#########################################################################
#######################   K Nearest Neighbors  ##########################
#########################################################################
with selected_model == "K Nearest Neighbors":
    
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
    st.caption("> Nearest Neighbors type of problem has methods, setting threshold as 1) Radius & 2) number of points, around query point and Majority wins.")
    st.caption("> KNN doesn't create learning model, but it remembers the data points distance (provided) proximity metric.")
    st.caption("> Basically assigning weights which are near to query point where weights are uniformly distributed.")
    st.caption("> By default distance metric is Euclidean distance.")
    st.caption("> As high dimensions data do not work well with distance metrics, we will select Response Coding featurized data.")
    
    hyperparameter, tuning, features_details, observations = st.tabs(["Hyperparameters","Hyperparameter Tuning", "Feature Importance & deepdive", "Observations"])
    
    with hyperparameter:
        if "knn_hp_selected" not in st.session_state:
            st.session_state.knn_hp_selected = "K"

        hp_options, hp_details = st.columns([1,3])

        with hp_options:
            st.session_state.knn_hp_selected = st.radio(label = "Hyperparameters" ,
                                                     options = ["K"])

        with hp_details:
            if st.session_state.knn_hp_selected == "K":
                st.caption("1. Larger the value of K, we'll haeve more point considerations.")
                st.caption("2. Too small value won't make generalization. Too large (or N itself) won't make it specific to.")
                st.caption("3. Default value is 5 neighbors.")  
            
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
        plt.title("Cross Validation (LogLoss) Error for different K", pad=25)
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
            
            predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index])
            st.metric(label = "Actual Class", value = test_y[test_point_index])
            st.metric(label = "Predicted Class", value = predicted_cls[0])
            
            st.write("Classwise Probabilities:")
            pred_class_probabilties = np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index]),4)
            prob_df = pd.DataFrame.from_dict({"Class" : [int(i) for i in range(1,len(pred_class_probabilties[0])+1)],\
                                              "Prob":pred_class_probabilties[0]})
            st.dataframe(prob_df.set_index('Class').T)

            # inspection on feature presence
            # neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), best_neighbors)
            # print("Optimal value for nearest neighbors knn is {0} \nThe nearest neighbours of the test points belongs to classes \n>>> {1}".\
            #       format(best_neighbors,Counter(train_y[neighbors[1][0]])))
            # no_feature = st.slider(label = 'Select top features to compare with: ',\
            #                        min_value = 0,\
            #                        max_value = test_x_responseCoding.shape[1],\
            #                        value = min(100,test_x_responseCoding.shape[1]),\
            #                        step = 25)
            # indices=np.argsort(-1*clf.feature_log_prob_)[predicted_cls-1][:,:no_feature]
            # get_impfeature_names(indices[0], \
            #                      test_df['Text'].iloc[test_point_index],\
            #                      test_df['Gene'].iloc[test_point_index],\
            #                      test_df['Variation'].iloc[test_point_index], \
            #                      no_feature, \
            #                      train_df)
        
    with observations:
        # Writing best parameters
        st.write("These are the best parameter values:")
        st.write(best_params)
        
        # Write Ups
        st.write("Observations")
        st.caption("As you can see logloss values are dropped drastically which is great.")
        st.caption("The misclassfication rate is ~27% which is moderate.")
        
        st.caption("Computational complexity with large dataset as it has to go through whole dataset.")
        st.caption("Does not work well with high dimensions")
        st.caption("Poor performance on imbalanced data")
        st.caption("Sensitive to noisy data, missing values and outliers")
        st.caption("Optimal value of K — If chosen incorrectly, the model will be under or overfitted to the data")
        st.caption("Need feature scaling, results in additional-step computationally and less interpretable.")
 