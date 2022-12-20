# Personalise-Cancer-Diagnosis

[![Website Badge](https://img.shields.io/badge/Website-3b5998?style=flat-square&logo=google-chrome&logoColor=white)](https://kmistri-personalise-cancer-diagnosis.streamlit.app/)

<details>
  <summary>Click here for awesomness.</summary>
  
  ### Screens:
  
  <ol>
  <li>Home:</li>
  <img src="https://user-images.githubusercontent.com/20341930/208598084-5d031500-6ba8-453e-ba54-647fc26156f6.png" width="600" height="400" />

  <li>Introduction:</li>
  <img src="https://user-images.githubusercontent.com/20341930/208598906-561aa576-121f-464e-8076-2ae6e90a50b6.png" width="600" height="400" />

  <li>EDA:</li>
  <p>This is little bit large <a href = "https://drive.google.com/file/d/1mD2-jLS-XOPy16NH19shgW72L8FQjFNc/view?usp=sharing">file</a></p>

  <li>Modelling:</li>
  <ul>
    <li>Model List</li>
    <img src="https://user-images.githubusercontent.com/20341930/208599494-a7d5c090-f870-49f7-82c4-4fd2a740c312.png" width="700" height="600" />
    <li>Hyper Parameter Tuning for each models:</li>
    <img src="https://user-images.githubusercontent.com/20341930/208599561-b2b5b58f-373b-4d5c-ba78-258c0314b410.png" width="700" height="1000" />
    <li>Feature Importance & Deepdive:</li>
    <img src="https://user-images.githubusercontent.com/20341930/208599576-1f157979-42e3-4a87-ba58-677e00cbd859.png" width="700" height="1800" />
    <li>Model Observations:</li>
    <img src="https://user-images.githubusercontent.com/20341930/208599802-8f28d872-ba48-4349-8749-db79fcbc628b.png" width="700" height="600" />
    <li>3D Interactable Plots for Multiple parameters:</li>
    <img src="https://user-images.githubusercontent.com/20341930/208599517-96beda8d-9024-4f4e-97c5-44480ce0e692.png" width="700" height="600" />
  </ul>
  <li>Results:</li>
  <img src="https://user-images.githubusercontent.com/20341930/208599151-9890df7a-371f-408f-8434-62d22da0c341.png" width="700" height="600" />
</ol>
</details>


From the expert, the time to diagnosis of cancer takes a lot of time as it includes new studies/papers, which makes this a time-consuming and exhaustive process. With machine learning, we can fast-track the majority of scenarios and help the expert get updated details.

I have used below classical machine learning algorithms for the problem.

1: Naive Bayes

2: K Nearest Neighbors

3: Logistic Regression

4: Support Vector Machine (SVM)

5: Random Forest Classifier

6: StackedClassifier (Ensemble)

7: MaxVoting Classifier (Ensemble)

As you might know, these algorithms have their limitation and advantages, I have tried to incorporate the best use of them by remediating the problems. Like

- The curse of dimensionality has been addressed by Response Coding.

- Class imbalance can be tuned with stratified splits and using the Class weight parameter whenever exploitable.

- Compute intensive Hyper-Tuning with parallelism when needed.

- At last, the beautiful interface & ton of integration of streamlit used.


### Deployment

1: Download package.

![PCD - Directory Structure with all files](https://user-images.githubusercontent.com/20341930/208602029-b261f026-5dc2-490a-9990-3963d9a14033.png)

2: Give executable rights to setup.sh and run it.

Note: 

    - Change values in setup.sh if you find it necessary.
    
    - For model generation, please run the Notebook to export data models. I have added here for (lightweight) streamlit runtime.

Once complete run, check host's 8080 (default) port.
