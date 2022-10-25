# Mental Health Status Classification Project

## Background
---

<img src="https://user-images.githubusercontent.com/34255556/196916971-21406264-79e5-4f70-a418-08c94e4c1b91.png" width="600">

Ever since the start of the pandemic back in 2020, the government expects more lockdowns through the winter to slow down the spread of the virus. One of the concerns raised by the government is the wellbeing of young primary school children since they are removed from their friends and play environments.

For this project, the main goal is to deploy initial classification models that help to monitor a child’s wellbeing status during the pandemic period. By identifying factors that influence the status of a child’s wellbeing through a customized survey, the government will be able to make more informed decisions/new policies that reduce the risk of a child having behavioral or emotional difficulties or both.

Dataset is provided in .json format by client under Training_Batch_Files folder for model training. (not included in this repository due to data confidentiality reasons)

For model prediction, a web API is used (created using StreamLit) for user input. Note that results generated from model prediction along with user inputs can be stored in various formats (i.e. in CSV file format or another database).

## Contents
- [Code and Resources Used](#code-and-resources-used)
- [Model Training Setting](#model-training-setting)
- [Project Findings](#project-findings)
  - [EDA](#1-eda-exploratory-data-analysis)
  - [Best classification model and pipeline configuration](#2-best-classification-model-and-pipeline-configuration)
  - [Summary of model evaluation metrics from best classification model](#3-summary-of-model-evaluation-metrics-from-best-classification-model)
  - [Hyperparameter importances from Optuna (Final model)](#4-hyperparameter-importances-from-optuna-final-model)
  - [Hyperparameter tuning optimization history from Optuna](#5-hyperparameter-tuning-optimization-history-from-optuna)
  - [Overall confusion matrix and classification report from final model trained](#6-overall-confusion-matrix-and-classification-report-from-final-model-trained)
  - [Precision Recall Curve from best classification model](#7-precision-recall-curve-from-best-classification-model)
  - [Learning Curve Analysis](#8-learning-curve-analysis)
  - [Feature Importance based on Shap Values for every class](#9-feature-importance-based-on-shap-values-for-every-class)
- [CRISP-DM Methodology](#crisp-dm-methodology)
- [Project Architecture Summary](#project-architecture-summary)
- [Project Folder Structure](#project-folder-structure)
- [Project Instructions (Local Environment)](#project-instructions-local-environment)
- [Project Instructions (Docker)](#project-instructions-docker)
- [Project Instructions (Heroku with Docker)](#project-instructions-heroku-with-docker)
- [Initial Data Cleaning and Feature Engineering](#initial-data-cleaning-and-feature-engineering)
- [Machine Learning Pipelines Configuration](#machine-learning-pipelines-configuration)
  - [Handling imbalanced data](#i-handling-imbalanced-data)
  - [Feature Engineering](#ii-feature-engineering)
  - [Encoding "interval" features](#iii-encoding-interval-features)
  - [Encoding "binary" features](#iv-encoding-binary-features)
  - [Encoding "ordinal" features with different magnitudes "for certain"](#v-encoding-ordinal-features-with-different-magnitudes-for-certain)
  - [Encoding features with rare categories](#vi-encoding-features-with-rare-categories)
  - [Encoding "nominal" and "ordinal" features with uncertainty in magnitude difference](#vii-encoding-nominal-and-ordinal-features-with-uncertainty-in-magnitude-difference)
  - [Encoding "time-related" features](#viii-encoding-time-related-features)  
  - [Feature Scaling](#x-feature-scaling)
  - [Feature Selection](#xi-feature-selection)
  - [Cluster Feature representation](#xii-cluster-feature-representation)
- [Legality](#legality)

## Code and Resources Used
---
- **Python Version** : 3.10.0
- **Packages** : borutashap, feature-engine, featurewiz, imbalanced-learn, joblib, catboost, lightgbm, matplotlib, pymongo, numpy, optuna, pandas, plotly, scikit-learn, scipy, seaborn, shap, streamlit, tqdm, xgboost, yellowbrick
- **Dataset source** : From 360DIGITMG (For confidentiality reasons, dataset is not included here)
- **Database**: MongoDB Atlas
- **MongoDB documentation**: https://www.mongodb.com/docs/
- **Optuna documentation** : https://optuna.readthedocs.io/en/stable/
- **Feature Engine documentation** : https://feature-engine.readthedocs.io/en/latest/
- **Imbalanced Learn documentation** : https://imbalanced-learn.org/stable/index.html
- **Scikit Learn documentation** : https://scikit-learn.org/stable/modules/classes.html
- **Shap documentation**: https://shap.readthedocs.io/en/latest/index.html
- **XGBoost documentation**: https://xgboost.readthedocs.io/en/stable/
- **LightGBM documentation**: https://lightgbm.readthedocs.io/en/latest/index.html
- **CatBoost documentation**: https://catboost.ai/en/docs/
- **Numpy documentation**: https://numpy.org/doc/stable/
- **Pandas documentation**: https://pandas.pydata.org/docs/
- **Plotly documentation**: https://plotly.com/python/
- **Matplotlib documentation**: https://matplotlib.org/stable/index.html
- **Seaborn documentation**: https://seaborn.pydata.org/
- **Yellowbrick documentation**: https://www.scikit-yb.org/en/latest/
- **Scipy documentation**: https://docs.scipy.org/doc/scipy/
- **Streamlit documentation**: https://docs.streamlit.io/
- **Blueprint for encoding categorical variables**: https://miro.medium.com/max/1400/1*gXKCiYdrIESmcbte2AtNeg.jpeg

## Model Training Setting
---
For this project, nested cross validation with stratification is used for identifying the best model class to use for model deployment. The inner loop of nested cross validation consists of 3 fold cross validation using Optuna (TPE Multivariate Sampler with 40 trials on optimizing average F1 macro score) for hyperparameter tuning on different training and validation sets, while the outer loop of nested cross validation consists of 5 fold cross validation for model evaluation on different test sets.

The diagram below shows how nested cross validation works:
<img src="https://mlr.mlr-org.com/articles/pdf/img/nested_resampling.png" width="600" height="350">

Given the dataset for this project is small (less than 1000 samples), nested cross validation is the most suitable cross validation method to use for model algorithm selection to provide a more realistic generalization error of machine learning models.

The following list of classification models are tested in this project:
- Logistic Regression
- Linear SVC
- K Neighbors Classifier
- Gaussian Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Ada Boost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

For model evaluation on multiclass classification, the following metrics are used in this project:
- Balanced accuracy
- Precision (macro)
- Recall (macro)
- F1 score (macro) - (Main metric for Optuna hyperparameter tuning)
- Matthew's correlation coefficient

## Project Findings
---

#### 1. EDA (Exploratory Data Analysis)

All plots generated from this section can be found in Intermediate_Train_Results/EDA folder.

#### i. Basic metadata of dataset
On initial inspection, the current dataset used in this project has a total of 71 features. Both "_id" and "ID.1 features represent unique identifier of a given record and the remaining features have  mix of "float", "int" and "object" data types. Upon closer inspection on data dictionary, there are several date-time related features where further information can be extracted and remaining features are considered as categorical variables.

Given that there is no target variable, this project requires creating target variable manually (Wellbeing_Category_WMS - mainly based on variables related to Me and My Feelings Questionnaire. More details can be found in the coding file labeled "train preprocessing.py")

![Target_Class_Distribution](https://user-images.githubusercontent.com/34255556/196934993-fc9bbe23-81c3-459c-8263-2ff26a51b31f.png)

From the diagram above, there is a very clear indication of target imbalance between all 4 classes for multiclass classification. This indicates that target imbalancing needs to be addressed during model training.

![Proportion of null values](https://user-images.githubusercontent.com/34255556/196935092-1ba7c4e8-740f-49e7-bfc3-2247ee977b32.png)

From the diagram above, most features with missing values identified have missing proportions approximately less than 1%, except for "Method_of_keepintouch" feature with approximately 3% containing missing values.

Furthermore, the following sets of plots are created for every feature of the dataset that contains less than 100 unique values:
1. Count plot (Number of unique values per category)
2. Count plot (Number of unique values per category by target class)
3. Bar plot (Number of missing values by target class) - For features with missing values

For features with more than 100 unique values, a CSV file is generated which represents the distribution of categories.
In addition, it was observed that features like "Breakfast_ytd", 'Method_of_keepintouch' and 'Type_of_play_places' can contain multiple values (seperated by ";" symbol). Those features are split into its individual categories first before generating a CSV file for representing distribution of categories.

The set of figures below shows an example of the following plots mentioned above for Method_of_keepintouch feature:

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196936871-869fbabc-b1a1-49e2-93f4-987097581926.png">
<img src="https://user-images.githubusercontent.com/34255556/196936977-e1f2e831-a255-4650-8c5d-4b27cfc560ef.png">
<img src="https://user-images.githubusercontent.com/34255556/196937003-dc7e5cd2-7464-40a7-bc77-2dc4d8bd3ca2.png" width="500">
</p>

---
#### 2. Best classification model and pipeline configuration

The following information below summarizes the configuration of the best model identified in this project:

  - <b>Best model class identified</b>: Linear Support Vector Classifier

  - <b>Method of handling imbalanced data</b>: None
  
  - <b>Contrast encoding method for ordinal data</b>: Sum Encoder

  - <b>Method of feature scaling</b>: MinMaxScaler

  - <b>Feature selection method</b>: FeatureWiz

  - <b>Number of features selected</b>: 68

  - <b>List of features selected</b>: ['Number_people_household_2', 'Number_people_household_3', 'Enoughtime_toplay_1', 'Enoughtime_toplay_3', 'Outdoorplay_freq_3',  'Method_of_keepintouch_I live near them so I can see them (at a social distance);By phone (texting, calling or video calling);On social media;On games consoles', 'Method_of_keepintouch_I live near them so I can see them (at a social distance);By phone (texting, calling or video calling)', 'Method_of_keepintouch_By phone (texting, calling or video calling);On social media', 'Method_of_keepintouch_By phone (texting, calling or video calling);On social media;On games consoles', 'Method_of_keepintouch_By phone (texting, calling or video calling)', 'Sugarsnack_in_week_1', 'Sugarsnack_in_week_2', 'Sugarsnack_in_week_3', 'Tired_in_week_0', 'Tired_in_week_1', 'Tired_in_week_2', 'Tired_in_week_3', 'Garden', 'Play_near_water', 'Play_in_grass_area', 'Play_in_house','Hours_slept', 'Contact_by_visit', 'Contact_by_phone', 'Snacks_Brk', 'Yogurt_Brk', 'Healthy_Cereal_Brk', 'Easywalk_topark', 'Read_Info_Sheet', 'School_Health_Records', 'Gender_Girl', 'Life_scale', 'School_scale', 'Play_inall_places_3', 'Doingwell_schoolwork_0', 'Sports_in_week_0', 'Internet_in_week_2', 'Takeawayfood_in_week_1', 'Concentrate_in_week_0', 'Going_school_No, I am at home', 'WIMD_2019_Decile', 'Friends_scale', 'Safety_toplay_scale', 'Type_of_play_places_In my house;In my garden', 'Breakfast_ytd_Sugary cereal e.g. cocopops, frosties, sugar puffs, chocolate cereals', 'Homespace_relax_Sometimes but not all the time', 'Study_Year', 'Lots_of_choices_important_1', 'Lots_of_choices_important_4', 'Lots_of_things_good_at_1', 'Lots_of_things_good_at_2', 'Lots_of_things_good_at_4', 'Softdrink_in_week_0', 'Softdrink_in_week_2', 'Softdrink_in_week_3', 'Sleeptime_ytd_sin', 'Sleeptime_ytd_hour_cos', 'Birth_Date_day_of_year_sin', 'Birth_Date_day_of_year_cos', 'Birth_Date_quarter_sin', 'Birth_Date_quarter_cos', 'Awaketime_today_hour_cos', 'Awaketime_today_hour_sin', 'Timestamp_day_of_week_sin', 'Timestamp_month_cos', 'Timestamp_month_sin']
  
  - <b>Clustering as additional feature</b>: Yes

  - <b>Best model hyperparameters</b>: 
  {'C': 0.13529521130402028, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l1', 'random_state': 120, 'tol': 0.0001,'verbose': 0}
  
Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation.

For every type of classification model tested in this project, a folder is created for every model class within Intermediate_Train_Results folder with the following artifacts:

- Confusion Matrix from 5 fold cross validation (.png format)
- Classification Report from 5 fold cross validation (.png format)
- HP_Importances for every fold (.png format - 5 in total)
- Hyperparameter tuning results for every fold (.csv format - 5 in total)
- Optimization history plot for every fold (.png format - 5 in total)
- Optuna study object for every fold (.pkl format - 5 in total)
- Precision-Recall curve (.png format)

In addition, the following artifacts are also created for the best model class identified after final hyperparameter tuning on the entire dataset:

- Confusion matrix (.png format)
- Classification report (.png format)
- HP_Importances (.png format)
- Hyperparameter tuning results (.csv format)
- Optimization history plot (.png format)
- Optuna study object (.pkl format)
- Learning curve plot (.png format)
- Shap plots for feature importance from every class (.png format - 2 in total)
- Precision recall curve (.png format)

<b>Warning: The following artifacts mentioned above for the best model class identified will not be generated for certain model classes under the following scenarios:
- Shap plots for KNeighborsClassifier and GaussianNB: For generating shap values for these model classes, Kernel explainer from Shap module can be used but with large computational time.
- Shap plots for XGBClassifier with dart booster: Tree explainer from Shap module currently doesn't support XGBClassifier with dart booster.</b>

---
#### 3. Summary of model evaluation metrics from best classification model

The following information below summarizes the evaluation metrics *(average (standard deviation)) from the best model identified in this project along with the confusion matrix from nested cross validation (5 outer fold with 3 inner fold): 

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196926115-2c43b974-4a55-4624-9e17-8db399b9510c.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/196926153-0b2b1d2e-7e09-40f0-9db6-360c87085d1a.png" width="400">
</p>

  - <b>Balanced accuracy (Training set - 3 fold)</b>: 0.4975 (0.0381)
  - <b>Balanced accuracy (Validation set - 3 fold)</b>: 0.3367 (0.0267)
  - <b>Balanced accuracy (Test set - 5 fold)</b>: 0.3484 (0.0284)

  - <b>Precision (Training set - 3 fold)</b>: 0.5574 (0.1139)
  - <b>Precision (Validation set - 3 fold)</b>: 0.3854 (0.0504)
  - <b>Precision (Test set - 5 fold)</b>: 0.3346 (0.0384)

  - <b>Recall (Training set - 3 fold)</b>: 0.4975 (0.0381)
  - <b>Recall (Validation set - 3 fold)</b>: 0.3367 (0.0267)
  - <b>Recall (Test set - 5 fold)</b>: 0.3484 (0.0284)

  - <b>F1 score (Training set - 3 fold)</b>: 0.4952 (0.0562)
  - <b>F1 score (Validation set - 3 fold)</b>: 0.3274 (0.0142)
  - <b>F1 score (Test set - 5 fold)</b>: 0.3293 (0.0265)

  - <b>Matthews Correlation Coefficient (Training set - 3 fold)</b>: 0.3590 (0.0605)
  - <b>Matthews Correlation Coefficient (Validation set - 3 fold)</b>: 0.1631 (0.0316)
  - <b>Matthews Correlation Coefficient (Test set - 5 fold)</b>: 0.1750 (0.0487)

Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation

---
#### 4. Hyperparameter importances from Optuna (Final model)

![HP_Importances_LinearSVC_Fold_overall](https://user-images.githubusercontent.com/34255556/196925529-e25eac89-ea69-4374-9d54-9951e331c90c.png)

From the image above, determining the contrast method for encoding ordinal data and method for handling imbalanced data as part of preprocessing pipeline for Linear SVC model provides the highest influence (0.22), followed by selecting hyperparameter value of "C", "class_weight" and feature selection method. Setting hyperparameter value of penalty and use of clustering as additional feature for Linear SVC model provides little to zero influence on results of hyperparameter tuning. This may suggest that both penalty hyperparameters of Linear SVC model and use of clustering as additional feature can be excluded from hyperparameter tuning in the future during model retraining to reduce complexity of hyperparameter tuning process.

---
#### 5. Hyperparameter tuning optimization history from Optuna

![Optimization_History_LinearSVC_Fold_overall](https://user-images.githubusercontent.com/34255556/196925946-56216317-c37c-4cb5-ad0b-632145be6386.png)

From the image above, the best objective value (average of F1 macro scores from 3 fold cross validation) is identified after 20 trials.

---
#### 6. Overall confusion matrix and classification report from final model trained

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196926313-f89b556b-2cd6-4b95-9095-7c3f2733c7e7.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/196926276-512f430d-5aaa-4916-96af-c71154cdcc2a.png" width="400">
</p>

From the image above, the classification model performs better for cases where a child's wellbeing is either normal or emotional and behaviour significant with more samples being classified correctly. Given that the model evaluation criteria emphasize the costly impact of having both false positives and false negatives equally for all classes, the current classification model is optimized to improve F1 macro score.

---
#### 7. Precision Recall Curve from best classification model

![PrecisionRecall_Curve_LinearSVC_CV](https://user-images.githubusercontent.com/34255556/196927600-68a0119c-c961-4ad1-9cd0-3efa9d7c1258.png)

From the diagram above, precision-recall curve from best model class identified shows that the model performs best on identify wellbeing status of children that are normal (0.89), followed by emotional_significant (0.10), behaviour_significant (0.09) and emotional_and_behaviour_significant (0.06). 

---
#### 8. Learning Curve Analysis

![LearningCurve_LinearSVC](https://user-images.githubusercontent.com/34255556/196927116-575713c0-699f-4f23-bfd8-248341a22c48.png)

From the diagram above, the gap between train and test F1 macro scores (from 5-fold cross validation) gradually decreases as number of training sample size increases.
However, the gap between both scores remain large, which indicates that adding more training data may help to improve generalization of model.

---
#### 9. Feature Importance based on Shap Values for every class

<b> Emotional and behaviour significant class</b>
<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196928977-169bdf6d-27b8-4553-82b0-3cd22c727088.png" width="800">
<img src="https://user-images.githubusercontent.com/34255556/196929024-2083fdcb-1990-4acc-b832-6ecb7e3f5820.png" width="800">
</p>

From both diagrams above, gender of child is the most influential variable from the top 68 variables identified from feature selection using FeatureWiz for predicting whether a child's wellbeing is both emotional and behaviour significant. Shap's summary plot provides indication of how values of different features may impact the result of model prediction. For example, gender of child not being identified as female have higher probability of being emotional and behaviour significant, while a child who is very unhappy with school or life (lower value of scale close to 0) has higher probabiliy of being identified as emotional and behaviour significant.

The following plots below represents feature importance based on shap values for other classes for reference:

<b> Emotional significant class</b>
<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196930962-850b0c9a-b95e-4bc5-a2a8-394c89e1cc85.png" width="800">
<img src="https://user-images.githubusercontent.com/34255556/196931136-4d8b1fe5-16de-4d69-9f70-9bd34ed19a73.png" width="800">
</p>

<b> Behaviour significant class</b>
<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196930911-5de4d998-39f5-4fcf-ae82-80291b84927d.png" width="800">
<img src="https://user-images.githubusercontent.com/34255556/196931084-f93f8c56-796e-47e1-b532-34080bae5ceb.png" width="800">
</p>

<b> Normal class</b>
<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196931003-40fbf74d-df59-4879-bfdb-5e01b65580bf.png" width="800">
<img src="https://user-images.githubusercontent.com/34255556/196931177-effe0b51-0207-4f09-8621-da4fc694204b.png" width="800">
</p>


## CRISP-DM Methodology
---
For any given Machine Learning projects, CRISP-DM (Cross Industry Standard Practice for Data Mining) methodology is the most commonly adapted methodology used.
The following diagram below represents a simple summary of the CRISP-DM methodology for this project:

<img src="https://www.datascience-pm.com/wp-content/uploads/2018/09/crisp-dm-wikicommons.jpg" width="450" height="400">

Note that an alternative version of this methodology, known as CRISP-ML(Q) (Cross Industry Standard Practice for Machine Learning and Quality Assurance) can also be used in this project. However, the model monitoring aspect is not used in this project, which can be considered for future use.

## Project Architecture Summary
---
The following diagram below summarizes the structure for this project:

![image](https://user-images.githubusercontent.com/34255556/197318105-4d4cd686-f6e5-43ed-8ad4-1cff1bbc2adf.png)

Note that all steps mentioned above have been logged accordingly for future reference and easy maintenance, which are stored in <b>Training_Logs</b> folder.

## Project Folder Structure
---
The following points below summarizes the use of every file/folder available for this project:
1. Application_Logger: Helper module for logging model training and prediction process
2. Intermediate_Train_Results: Stores results from EDA, data preprocessing and model training process
3. Model_Training_Modules: Helper modules for model training
4. Saved_Models: Stores best models identified from model training process for model prediction
5. Training_Batch_Files: Stores csv batch files to be used for model training
6. Training_Data_FromDB: Stores compiled data from SQL database for model training
7. Training_Logs: Stores logging information from model training for future debugging and maintenance
8. Dockerfile: Additional file for Docker project deployment
9. README.md: Details summary of project for presentation
10. requirements.txt: List of Python packages to install for project deployment
11. setup.py : Script for installing relevant python packages for project deployment
12. Docker_env: Folder that contains files that are required for model deployment without logging files or results.
13. BorutaShap.py: Modified python script with some changes to coding for performing feature selection based on shap values on test set
14. _tree.py: Modified python script to include AdaBoost Classifier as part of the set of models that support Shap library.
15. pipeline_api.py: Main python file for running training pipeline process and performing model prediction.

## MongoDB Atlas Setup
---

![image](https://user-images.githubusercontent.com/34255556/197315546-b60b36b7-10e2-4b50-9eff-ae62ed44b17d.png)

For this project, data provided by the client in JSON format will be stored in MongoDB Atlas, which is a cloud database platform specially for MongoDB.

The following steps below shows the setup of MongoDB Atlas:

1. Register for a new MongoDB Atlas account for free using the following link: https://www.mongodb.com/cloud/atlas/register
2. After login, create a new database cluster (Shared option) and select the cloud provider and region of your choice:

<img src = "https://user-images.githubusercontent.com/34255556/197315198-8a65d44a-9e75-4d65-9de4-f3c10748b066.png" width="600">

3. Go to Database Access tab under Security section and add a new database user as follows:

<img src = "https://user-images.githubusercontent.com/34255556/197315308-c6c25139-528f-40f4-a3f1-55a6a269df68.png" width="600">

- Keep a record of username and password created for future use.

4. Go to Database tab under Deployment section and click on Connect button:

![image](https://user-images.githubusercontent.com/34255556/197315396-710bae00-c75d-4f69-b267-0ee9e217c819.png)

5. Select "Connect your application" option:

<img src = "https://user-images.githubusercontent.com/34255556/197315427-79eaf258-0ac7-4762-b3d5-4856d8474759.png" width="600">

6. <b>Important: Make a note of the connection string and replace username and password by its values from step 3.</b>

![image](https://user-images.githubusercontent.com/34255556/197315449-c077c899-97ed-4ce7-9d52-25c79bfaa217.png)

- Note that this connection string is required for connecting our API with MongoDB atlas.

The following sections below explains the three main approaches that can be used for deployment in this project after setting up MongoDB Atlas:
1. <b>Docker</b>
2. <b>Cloud Platform (Heroku with Docker)</b>
3. <b>Local environment</b>

## Project Instructions (Docker)
---
<img src="https://user-images.githubusercontent.com/34255556/195037066-21347c07-217e-4ecd-9fef-4e7f8cf3e098.png" width="600">

Deploying this project on Docker allows for portability between different environments and running instances without relying on host operating system.
  
<b>Note that docker image is created under Windows Operating system for this project, therefore these instructions will only work on other windows instances.</b>

<b> For deploying this project onto Docker, the following additional files are essential</b>:
- DockerFile
- requirements.txt
- setup.py

Docker Desktop needs to be installed into your local system (https://www.docker.com/products/docker-desktop/), before proceeding with the following steps:

1. Download and extract the zip file from this github repository into your local machine system.

<img src="https://user-images.githubusercontent.com/34255556/197315695-5f19b123-22a3-4751-82cd-d6bbca13a3d9.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. On line 8 inside Dockerfile, set the environment variable MONGO_DB_URL as the connection string defined in the last step of MongoDB Atlas Setup section.

![image](https://user-images.githubusercontent.com/34255556/197315793-d676cd57-b2e3-4702-9c83-1fcd84efe6d8.png)

4. Build a new docker image on the project directory with the following command:
```
docker build -t api-name .
```

5. Run the docker image on the project directory with the following command: 
```
docker run -e PORT=8501 -p 8501:8501 api-name
```

6. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/197315976-fa90cc7a-a0b3-4c82-9c38-62072db71399.png" width="600">

Browser for the application can be opened from Docker Desktop by clicking on the specific button shown below:

![image](https://user-images.githubusercontent.com/34255556/197315936-2ea47b7a-9919-4010-b806-52f864966ea3.png)

7. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197316040-748289f6-f509-4e29-aac0-1765de6d3167.png" width="600">

8. After running all steps of the training pipeline, run the following command to extract files from a specific directory within the docker container to host machine for viewing:
```
docker cp <container-id>:<source-dir> <destination-dir>
```

9. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197316098-ec71b7df-6819-4c46-944b-27596c6b262b.png" width="600">

10. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197316193-d1cf6fb7-91be-4283-91d6-cced35c70e41.png" width="600">

## Project Instructions (Heroku with Docker)
---
<img src = "https://user-images.githubusercontent.com/34255556/195489080-3673ab77-833d-47f6-8151-0fed308b9eec.png" width="600">

A suitable alternative for deploying this project is to use docker images with cloud platforms like Heroku. 

<b> For deploying models onto Heroku platform, the following additional files are essential</b>:
- DockerFile
- requirements.txt
- setup.py

<b>Note that deploying this project onto other cloud platforms like GCP, AWS or Azure may have different additionnal files required.</b>

For replicating the steps required for running this project on your own Heroku account, the following steps are required:
1. Clone this github repository into your local machine system or your own Github account if available.

<img src="https://user-images.githubusercontent.com/34255556/197315695-5f19b123-22a3-4751-82cd-d6bbca13a3d9.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. On line 8 inside Dockerfile, set the environment variable MONGO_DB_URL as the connection string defined in the last step of MongoDB Atlas Setup section.

![image](https://user-images.githubusercontent.com/34255556/197315793-d676cd57-b2e3-4702-9c83-1fcd84efe6d8.png)

5. From a new command prompt window, login to Heroku account and Container Registry by running the following commands:
```
heroku login
heroku container:login
```
Note that Docker needs to be installed on your local system before login to heroku's container registry.

6. Using the Dockerfile, push the docker image onto Heroku's container registry using the following command:
```
heroku container:push web -a app-name
```

7. Release the newly pushed docker images to deploy app using the following command:
```
heroku container:release web -a app-name
```

8. After successfully deploying docker image onto Heroku, open the app from the Heroku platform and you will see the following interface designed using Streamlit:
<img src = "https://user-images.githubusercontent.com/34255556/197315976-fa90cc7a-a0b3-4c82-9c38-62072db71399.png" width="600">

9. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197316040-748289f6-f509-4e29-aac0-1765de6d3167.png" width="600">

10. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197316098-ec71b7df-6819-4c46-944b-27596c6b262b.png" width="600">

11. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197316193-d1cf6fb7-91be-4283-91d6-cced35c70e41.png" width="600">

<b>Important Note</b>: 
- Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.

- Unlike stand-alone Docker containers, Heroku uses an ephemeral hard drive, meaning that files stored locally from running apps on Heroku will not persist when apps are restarted (once every 24 hours). Any files stored on disk will not be visible from one-off dynos such as a heroku run bash instance or a scheduler task because these commands use new dynos. Best practice for having persistent object storage is to leverage a cloud file storage service such as Amazon’s S3 (not part of project scope but can be considered)

## Project Instructions (Local Environment)
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/197315695-5f19b123-22a3-4751-82cd-d6bbca13a3d9.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. Add environment variable "MONGO_DB_URL" with connection string defined from last step of MongoDB Atlas setup section as value on your local system. The following link provides an excellent guide for setting up environment variables on your local system: https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/

4. Open anaconda prompt and create a new environment with the following syntax: 
```
conda create -n myenv python=3.10
```
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

5. After creating a new anaconda environment, activate the environment using the following command: 
```
conda activate myenv
```

6. Go to the local directory in Command Prompt where Docker_env folder is located and run the following command to install all the python libraries : 
```
pip install -r requirements.txt
```

7. Overwrite both BorutaShap.py and _tree.py scripts in relevant directories (<b>env/env-name/lib/site-packages and env/env-name/lib/site-packages/shap/explainers</b>) where the original files are located.

8. After installing all the required Python libraries, run the following command on your project directory: 
```
streamlit run pipeline_api.py
```

9. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/197315976-fa90cc7a-a0b3-4c82-9c38-62072db71399.png" width="600">

10. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197316040-748289f6-f509-4e29-aac0-1765de6d3167.png" width="600">

11. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197316098-ec71b7df-6819-4c46-944b-27596c6b262b.png" width="600">

12. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197316193-d1cf6fb7-91be-4283-91d6-cced35c70e41.png" width="600">

## Initial Data Cleaning and Feature Engineering
---
After performing Exploratory Data Analysis, the following steps are performed initially on the entire dataset before performing further data preprocessing and model training:

i) Filter data where respondent provides permission to use questionnaire (Use_questionnaire feature)

ii) Derive target variable based on features related to Me and My Feelings questionnaire.

iii) Reformat time related features (i.e Timestamp, Birthdate, Sleeptime_ytd and Awaketime_today) to appropriate form

iv) Removing list of irrelevant colummns identified from dataset (i.e. unique identifier features, features related to target variable to prevent target leakage and LSOA related features that have direct one to one relationship with WIMD related features)

v) Checking for duplicated rows and remove if exist

vi) Split dataset into features and target labels.

vii) Perform missing imputation on categorical variables based on highest frequency for every category.

viii) Save reduced set of features and target values into 2 different CSV files (X.csv and y.csv) for further data preprocessing with pipelines to reduce data leakage.

For more details of which features have been initially removed from the dataset, refer to the following CSV file: <b>Columns_Drop_from_Original.csv</b>

In addition, the following pickle files (with self-explanatory names) have been created inside Intermediate_Train_Results folder during this stage which may be used later on during data preprocessing on test data:
- <b>CategoryImputer.pkl</b>

## Machine Learning Pipelines Configuration
---
While data preprocessing steps can be done on the entire dataset before model training, it is highly recommended to perform all data preprocessing steps within cross validation using pipelines to reduce the risk of data leakage, where information from training data is leaked to validation/test data.

The sections below summarizes the details of Machine Learning pipelines with various variations in steps:

#### i. Handling imbalanced data
While most machine learning models have hyperparameters that allow adjustment of <b>class weights</b> for classification, an alternative solution to handle imbalanced data is to use oversampling method.

For this project, the following methods of handling imbalanced data are tested:

- SMOTEN: Synthetic Minority Over-sampling Technique for Nominal data.
- No oversampling or undersampling required

Note that this dataset do not contain continuous variables, thus the only suitable methods available for handling imbalanced data is either using SMOTEN for oversampling or using class weights hyperparameter.

#### ii. Feature Engineering
The following features are derived after handling imbalanced data if relevant:
- Age (Difference between Timestamp and Birth_Date)
- Hours slept (Difference between Awaketime_today and Sleeptime_ytd)
- Datetime features (i.e. year, month, quarter, week, day_of_week, day_of_month, day_of_year, hour and minute) from Timestamp, Birth_Date, Awaketime_today and Sleeptime_ytd (using DatetimeFeatures function from feature-engine library)
- Number of methods of keep in touch (based on Method_of_keepintouch feature)
- Number of types of play places (based on Type_of_play_places feature)
- Number of breakfast food yesterday (based on Breakfast_ytd feature)

#### iii. Encoding "interval" features
Features that are identifed as interval data types have equal magnitudes between different values. These features can be encoded directly using custom <b>label encoding (from 0)</b>:
- Study_Year
- Safety_toplay_scale
- Health_scale
- School_scale
- Family_scale
- Friends_scale
- Looks_scale
- Life_scale
- WIMD_2019_Rank
- WIMD_2019_Decile
- WIMD_2019_Quintile
- WIMD_2019_Quartile

#### iv. Encoding "binary" features
Features that are identified as binary data types only require simple encoding (1 vs 0):
- Read_Info_Sheet
- School_Health_Records
- Other_children_inhouse
- Easywalk_topark
- Easywalk_somewhere
- Garden
- Keep_in_touch_family_outside_household
- Keep_in_touch_friends
- Sleeptime_ytd_minute
- Awaketime_today_minute

In addition, these following features contain multiple values which can also be split into individual values in binary form:
- Method_of_keepintouch (Contact_by_phone, Contact_by_visit, Contact_by_social_media, Contact_by_game)
- Type_of_play_places (Play_in_house, Play_in_garden, Play_in_grass_area, Play_in_bushes, Play_in_woods, Play_in_field, Play_in_street, Play_in_playground, Play_in_bike_or_park, Play_near_water) * Only top 10 categories are used
- Breakfast_ytd (Bread_Brk, Sugary_Cereal_Brk, Healthy_Cereal_Brk, Fruits_Brk, Yogurt_Brk, Nothing_Brk, Cooked_Breakfast_Brk, Snacks_Brk) * Only top 8 categories are used

#### v. Encoding "ordinal" features with different magnitudes "for certain"
For features that are identified as ordinal data types with high certainty of categories having different magnitudes can be encoded using one of the following contrast methods:
- Backward Difference Encoder
- Polynomial Encoder
- Sum Encoder
- Helmert Encoder

These following features will be encoded using contrast methods, since these features clearly show different magnitudes:
- Fruitveg_ytd
- Number_people_household
- Sports_in_week 
- Internet_in_week
- Tired_in_week
- Concentrate_in_week
- Softdrink_in_week
- Sugarsnack_in_week
- Takeawayfood_in_week

#### vi. Encoding features with rare categories
For features that contain many unique categories, these features may have categories that are considered to be "rare" due to low frequency. To reduce the cardinality of these features, <b>RareLabelEncoder</b> from feature-engine library is used.

#### vii. Encoding "nominal" and "ordinal" features with uncertainty in magnitude difference
The following list of features are ordinal that may or may not have different magnitudes between values:
- Doingwell_schoolwork
- Lots_of_choices_important
- Lots_of_things_good_at
- Feel_partof_community
- Outdoorplay_freq
- Enoughtime_toplay
- Play_inall_places

<b>Note that the features listed above are encoded using label encoding (from 0 in order of importance) as intermediate step before further data encoding.</b>

The following list of features are identified as nominal:
- Gender
- Going_school
- Homespace_relax
- Method_of_keepintouch
- Breakfast_ytd
- Type_of_play_places

<b>For all the features mentioned in this section, features are encoded using either One Hot encoding (for non-tree based models) or CatBoost encoding (for tree-based models).</b>

#### viii. Encoding "time-related" features
All time-related features that are derived from Timestamp, Birth_Date, Sleeptime_ytd and Awaketime_today are encoded using either CyclicalFeatures (for non-tree based models) function from feature-engine library or CatBoost encoding (for tree-based models).

Note that while One Hot encoding is the most popular approach for categorical data encoding, time-related features are usually cyclical in nature such that performing one hot encoding on time related features does not capture the cyclical component.

#### x. Feature Scaling
Feature scaling is only essential in some machine learning models like Logistic Regression, Linear SVC and KNN for faster convergence and to prevent misinterpretation of one feature significantly more important than other features. For this project, MinMax scaler is used since this dataset only contains categorical variables.

#### xi. Feature Selection
Given the current dataset has very large number of features, performing feature selection is essential for simplifying the machine learning model, reducing model training time and to reduce risk of model overfitting.

For this project, the following methods of feature selection are tested:
- Mutual Information
- ANOVA
- Feature Importance using Extra Trees Classifier
- Logistic Regression with Lasso Penalty (l1)
- BorutaShap (Default base learner: Random Forest Classifier)
- FeatureWiz (SULOV (Searching for Uncorrelated List of Variables) + Recursive Feature Elimination with XGBoost Classifier)

#### xii. Cluster Feature representation
After selecting the best features from feature selection, an additional step that can be tested involves representing distance between various points and identified cluster point as a feature (cluster_distance) for model training. From the following research paper (https://link.springer.com/content/pdf/10.1007/s10115-021-01572-6.pdf) written by Maciej Piernik and Tadeusz Morzy in 2021, both authors concluded the following points that will be applied to this project:

-  Adding cluster-generated features may improve quality of classification models (linear classifiers like Logistic Regression and Linear SVC), with extra caution required for non-linear classifiers like K Neighbors Classifier and random forest approaches.

- Encoding clusters as features based on distances between points and cluster representatives with feature scaling is significantly better than solely relying on cluster membership with One Hot encoding. 

- Adding generated cluster features to existing ones is safer option than replacing them altogether, which may yield model improvements without degrading model quality

- No single clustering approach (K-means vs Hierarchical vs DBScan vs Affinity Propagation) provide significantly better results in model performance. Thus, affinity propagation method is used for this project, which automatically determines the number of clusters to use. However, "damping" parameter requires hyperparameter tuning for using Affinity Propagation method.

## Legality
---
This is an internship project made with 360DIGITMG for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
