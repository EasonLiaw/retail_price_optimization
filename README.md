# Retail Price Optimization Project

## Background
---

<img src="https://user-images.githubusercontent.com/34255556/197797026-5ffd63db-835c-40df-9f0f-867d82e5c3d9.png" width="600">

With the COVID-19 pandemic, there has been a signficant boom in the e-commerce industry with more sellers shifting their businesses towards e-commerce platforms. While traditional rule-based methods have been used by retailers in the past to manage price optimization, these methods require manual analysis of customer and market data to ensure prices are aligned with current market conditions. Inevitably, the overall expansion of digitization of retail industry in recent years due to the global pandemic has resulted in a massive increase of sales-related data, such that traditional rule-based methods result in difficulty of continuous monitoring.

For this project, the main goal is to deploy initial regression models that help to predict retail price for various products. With the help of Machine Learning, retailers will be able to utilize the full potential of their data by effectively setting prices that maximizes their profits without discouraging customers from purchasing their products.

Dataset is provided in .csv format by client under Training_Batch_Files folder for model training. (not included in this repository due to data confidentiality reasons) In addition, schema of datasets for model training is provided in .json format by the client for storing csv files into a single PostgreSQL database.

For model prediction, a web API is used (created using StreamLit) for user input and the results generated from model prediction along with user inputs can be stored in various formats (i.e. in CSV file format or another PostgreSQL database).

## Contents
- [Code and Resources Used](#code-and-resources-used)
- [Model Training Setting](#model-training-setting)
- [Project Findings](#project-findings)
  - [EDA](#1-eda-exploratory-data-analysis)
  - [Best regression model and pipeline configuration](#2-best-regression-model-and-pipeline-configuration)
  - [Summary of model evaluation metrics from best regression model](#3-summary-of-model-evaluation-metrics-from-best-regression-model)
  - [Hyperparameter importances from Optuna (Final model)](#4-hyperparameter-importances-from-optuna-final-model)
  - [Hyperparameter tuning optimization history from Optuna](#5-hyperparameter-tuning-optimization-history-from-optuna)
  - [Overall residual plot from final model trained](#6-overall-residual-plot-from-final-model-trained)
  - [Leverage plot](#7-leverage-plot)
  - [Learning Curve Analysis](#8-learning-curve-analysis)
  - [Feature Importance based on Shap Values](#9-feature-importance-based-on-shap-values)
- [CRISP-DM Methodology](#crisp-dm-methodology)
- [Project Architecture Summary](#project-architecture-summary)
- [Project Folder Structure](#project-folder-structure)
- [Project Instructions (Local Environment)](#project-instructions-local-environment)
- [Project Instructions (Docker)](#project-instructions-docker)
- [Project Instructions (Heroku with Docker)](#project-instructions-heroku-with-docker)
- [Initial Data Cleaning and Feature Engineering](#initial-data-cleaning-and-feature-engineering)
- [Machine Learning Pipelines Configuration](#machine-learning-pipelines-configuration)
  - [Handling outliers by capping at extreme values](#i-handling-outliers-by-capping-at-extreme-values)
  - [Gaussian transformation on non-gaussian variables](#ii-gaussian-transformation-on-non-gaussian-variables)
  - [Encoding features with rare categories](#iii-encoding-features-with-rare-categories)
  - [Feature Engineering](#iv-feature-engineering)
  - [Encoding "interval" features](#v-encoding-interval-features)
  - [Encoding "nominal" features](#vi-encoding-nominal-features)
  - [Encoding "time-related" features](#vii-encoding-time-related-features)  
  - [Feature Scaling](#viii-feature-scaling)
  - [Feature Selection](#x-feature-selection)
- [Legality](#legality)

## Code and Resources Used
---
- **Python Version** : 3.10.0
- **Packages** : category_encoders, dython, feature-engine, featurewiz, joblib, catboost, lightgbm, matplotlib, psycopg2-binary, numpy, optuna, pandas, plotly, scikit-learn, scipy, seaborn, shap, streamlit, tqdm, xgboost, yellowbrick
- **Dataset source** : From 360DIGITMG (For confidentiality reasons, dataset is not included here)
- **Database**: PostgreSQL
- **PostgreSQL documentation**: https://www.mongodb.com/docs/
- **Optuna documentation** : https://optuna.readthedocs.io/en/stable/
- **Category_encoders documentation**: https://contrib.scikit-learn.org/category_encoders/index.html
- **Feature Engine documentation** : https://feature-engine.readthedocs.io/en/latest/
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
For this project, nested cross validation is used for identifying the best model class to use for model deployment. The inner loop of nested cross validation consists of 3 fold cross validation using Optuna (TPE Multivariate Sampler with 20 trials on optimizing root mean square error (RMSE)) for hyperparameter tuning on different training and validation sets, while the outer loop of nested cross validation consists of 5 fold cross validation for model evaluation on different test sets.

The diagram below shows how nested cross validation works:
<img src="https://mlr.mlr-org.com/articles/pdf/img/nested_resampling.png" width="600" height="350">

Given the dataset for this project is moderately large (approximately 30000 samples), nested cross validation is still the most suitable cross validation method to use for model algorithm selection to provide a more realistic generalization error of machine learning models.

The following list of classification models are tested in this project:
- Huber Regression
- Ridge
- Lasso
- ElasticNet
- Linear SVR
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Ada Boost Regressor
- Histogram Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

For model evaluation on regression, the following metrics are used in this project:
- Root mean squared error - (Main metric for Optuna hyperparameter tuning)
- Mean absolute error
- Median absolute error

<b>Note that Mean absolute percentage error (MAPE) is not used for this project because there are target values very close to zero value.</b>

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
#### 2. Best regression model and pipeline configuration

The following information below summarizes the configuration of the best model identified in this project:

  - <b>Best model class identified</b>: Histogram Gradient Boosting Regressor

  - <b>Method of handling outliers</b>: Retain outliers

  - <b>Method of feature scaling</b>: None
  
  - <b>Removing highly correlated features (>0.8)</b>: No
  
  - <b>Feature selection method</b>: Feature Importance from ExtraTreesRegressor

  - <b>Number of features selected</b>: 14

  - <b>List of features selected</b>: ['Brand', 'NSU', 'Sales at Cost', 'Gross Sales', 'MRP', 'Fdate_year', 'Fdate_month', 'Fdate_quarter', 'Sales at Cost_median_Brand', 'Gross Sales_mean_Brand', 'Gross Sales_median_Brand', 'MRP_mean_Brand', 'MRP_std_Brand', 'MRP_median_Brand']
  
  - <b>Best model hyperparameters</b>: {'categorical_features': None, 'early_stopping': 'auto', 'l2_regularization': 0.7053706737458695, 'learning_rate': 0.09253085147706015, 'loss': 'squared_error', 'max_bins': 255, 'max_depth': 7, 'max_iter': 100, 'max_leaf_nodes': 20, 'min_samples_leaf': 29, 'monotonic_cst': None, 'n_iter_no_change': 10, 'quantile': None, 'random_state': 120, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
  
Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation.

For every type of regression model tested in this project, a folder is created for every model class within Intermediate_Train_Results folder with the following artifacts:

- HP_Importances for every fold (.png format - 5 in total)
- Hyperparameter tuning results for every fold (.csv format - 5 in total)
- Optimization history plot for every fold (.png format - 5 in total)
- Optuna study object for every fold (.pkl format - 5 in total)
- Residual plot from nested cross validation (.png format)

In addition, the following artifacts are also created for the best model class identified after final hyperparameter tuning on the entire dataset:

- HP_Importances (.png format)
- Hyperparameter tuning results (.csv format)
- Optimization history plot (.png format)
- Optuna study object (.pkl format)
- Residual plot (.png format)
- Leverage plot (.png format)
- Learning curve plot (.png format)
- Shap plots for feature importance (.png format)

<b>Warning: The following artifacts mentioned above for the best model class identified will not be generated for certain model classes under the following scenarios:
- Shap plots for XGBRegressor with dart booster: Tree explainer from Shap module currently doesn't support XGBRegressor with dart booster.</b>

---
#### 3. Summary of model evaluation metrics from best regression model

The following information below summarizes the evaluation metrics *(average (standard deviation)) from the best model identified in this project along with residual plot from nested cross validation (5 outer fold with 3 inner fold): 

![Residual_Plot_HistGradientBoostingRegressor_from_CV](https://user-images.githubusercontent.com/34255556/197925603-cc8ac94a-31fe-4855-9e28-a866f83a0f67.png)

  - <b>Root Mean Squared Error (Training set - 3 fold)</b>: 38.8801 (1.7517)
  - <b>Root Mean Squared Error (Validation set - 3 fold)</b>: 52.6107 (1.7447)
  - <b>Root Mean Squared Error (Test set - 5 fold)</b>: 49.1677 (6.8408)

  - <b>Mean Absolute Error (Training set - 3 fold)</b>: 12.9003 (0.5338)
  - <b>Mean Absolute Error (Validation set - 3 fold)</b>: 14.6027 (0.4006)
  - <b>Mean Absolute Error (Test set - 5 fold)</b>: 14.0344 (0.8000)

  - <b>Median Absolute Error (Training set - 3 fold)</b>: 5.0394 (0.4322)
  - <b>Median Absolute Error (Validation set - 3 fold)</b>: 5.1313 (0.4206)
  - <b>Median Absolute Error (Test set - 5 fold)</b>: 5.0042 (0.4056)

Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation

---
#### 4. Hyperparameter importances from Optuna (Final model)

![HP_Importances_HistGradientBoostingRegressor_Fold_overall](https://user-images.githubusercontent.com/34255556/197925076-3b0b7f66-3cd8-4558-b166-ba3ba12def69.png)

From the image above, setting the learning rate for Histogram Gradient Boosting Regression model provides the highest influence (0.67), followed by selecting hyperparameter value of "l2 regularization", "min_samples_leaf", "max_leaf_nodes" and feature selection method. Setting hyperparameter value of max depth for  Histogram Gradient Boosting Regression model provides little to zero influence on results of hyperparameter tuning. This may suggest that max depth hyperparameter of Histogram Gradient Boosting Regression model can be excluded from hyperparameter tuning in the future during model retraining to reduce complexity of hyperparameter tuning process.

---
#### 5. Hyperparameter tuning optimization history from Optuna

![Optimization_History_HistGradientBoostingRegressor_Fold_overall](https://user-images.githubusercontent.com/34255556/197925433-e0de62dc-9b81-4d0d-bf1f-8ac437339f4e.png)

From the image above, the best objective value (average of RMSE scores from 3 fold cross validation) is identified after 13 trials.

---
#### 6. Overall residual plot from final model trained

![Residual_Plot_HistGradientBoostingRegressor_final_model](https://user-images.githubusercontent.com/34255556/197925793-79e4b552-a20c-48dd-94ba-c29889f68102.png)

From the image above, the regression model performs better for lower price values predicted with smaller variance, as compared to larger price values predicted with larger variance. This clearly indicates that linear models are less suitable for predicting retail prices due to homoscedasticity (constant variance) assumption being violated.

---
#### 7. Leverage plot

![Leverage_Plot](https://user-images.githubusercontent.com/34255556/197926315-56da2865-f219-4611-a5c2-50ac53003436.png)

From the diagram above, there are many records that are identified as having high leverage or large residuals. Using Cook’s distance, this confirms that there are approximately 1027 records (2.76% of total observations) as being highly influential, which requires further investigation on the data for better understanding whether these data anomalies should be treated.

---
#### 8. Learning Curve Analysis

![LearningCurve_HistGradientBoostingRegressor](https://user-images.githubusercontent.com/34255556/197926663-47037f1d-c9e4-4175-bf86-3b5701cbab0d.png)

From the diagram above, the gap between train and test RMSE scores (from 5-fold cross validation) gradually decreases as number of training sample size increases.
However, the gap between both scores remain large, which indicates that adding more training data may help to improve generalization of model.

---
#### 9. Feature Importance based on Shap Values

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/197926822-e023a34b-adc2-4512-9233-ec4615e991e2.png" width="500">
<img src="https://user-images.githubusercontent.com/34255556/197926840-d7f2ca90-2a8a-43b2-b0f9-f70bdf3395e6.png" width="500">
</p>

From both diagrams above, MRP (Maximum Retail Price) is the most influential variable from the top 14 variables identified from feature selection using Extra Trees Regressor for predicting retail price of various products. Shap's summary plot provides indication of how values of different features may impact the result of model prediction. For example, higher values of MRP and Sales at Cost indicate higher predicted values of selling price (SP).

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
13. _tree.py: Modified python script to include AdaBoost Classifier as part of the set of models that support Shap library.
14. pipeline_api.py: Main python file for running training pipeline process and performing model prediction.

The following sections below explains the three main approaches that can be used for deployment in this project:
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

<img src="https://user-images.githubusercontent.com/34255556/197928020-10f6bf6e-57aa-439f-b79f-8111d6e36083.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Create the following volume (postgresql) and network in Docker for connecting between database container and application container using the following syntax:
```
docker volume create postgresql
docker network create postgresqlnet
```
Note that the naming conventions for both volumes and network can be changed.

4. Run the following docker volumes and network for creating a new PostgreSQL container in Docker:
```
docker run --name postgresql -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=mypassword -p 5432:5432 -v /data:/var/lib/postgresql/data --network postgresqlnet -d postgres
```
Note that postgresql refers to the name of the container, which will also be host name of database.

5. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure:
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <default_database_name>} 
```
- For security reasons, this file needs to be stored in private. (Default host is container name defined in step 4 and user is postgres for PostgreSQL)

6. Build a new docker image on the project directory with the following command:
```
docker build -t api-name .
```

7. Run the docker image on the project directory with the following command: 
```
docker run --network postgresqlnet -e PORT=8501 -p 8501:8501 api-name
```
Note that the command above creates a new docker app container with the given image "api-name". Adding network onto the docker app container will allow connection between two separate docker containers.

8. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/197928966-d98530da-cb51-4db8-b052-47f51069c909.png" width="600">

Browser for the application can be opened from Docker Desktop by clicking on the specific button shown below:

![Image3](https://user-images.githubusercontent.com/34255556/197929022-0f029983-6c51-40e9-ba1b-3a44c4787d84.png)

9. From the image above, click on Training Data Validation first for initializing data ingestion into PostgreSQL, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197929141-3df4a3f0-6ac3-4b71-b519-479aebdec935.png" width="600">

10. After running all steps of the training pipeline, run the following command to extract files from a specific directory within the docker container to host machine for viewing:
```
docker cp <container-id>:<source-dir> <destination-dir>
```

11. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197929228-29644a4a-fb9b-48de-ba08-2b4fcb5fe2ae.png" width="600">

12. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197929274-e42dc33c-1f81-40b9-9a8f-5ae73345a872.png" width="600">

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

<img src="https://user-images.githubusercontent.com/34255556/197928020-10f6bf6e-57aa-439f-b79f-8111d6e36083.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. Go to "Resources" tab and search for Heroku Postgres in the add-ons search bar and select it with the relevant pricing plan. (Note that I select Hobby Dev plan, which is currently free. However, recommended plan to select is Hobby- basic, which currently cost about $9 per month to increase storage capacity for this project.)
<img src="https://user-images.githubusercontent.com/34255556/197930492-f59865f6-c5ae-42b5-bd88-9b9f9e404e98.png">

5. Under Model_Training_Modules/validation_train_data.py file, comment out the following lines:
![image](https://user-images.githubusercontent.com/34255556/197930177-1646465f-df80-4651-9ea8-8ad6482e1ed1.png)

Note that the current python script will create new database if not exist in local environment/docker, but these commands will fail when connecting to Heroku PostgreSQL database since a database has already been assigned when adding Heroku PostgreSQL resource to the Heroku app.

6. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure:
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <default_database_name>} 
```
- For security reasons, this file needs to be stored in private. Note that configuration details for connecting application with Heroku PostgreSQL can be found by finding database credentials under settings tab as shown in the image below:

![image](https://user-images.githubusercontent.com/34255556/197931105-1a0b0c14-afc6-44c5-95ff-7bc04f35e22f.png)

7. From a new command prompt window, login to Heroku account and Container Registry by running the following commands:
```
heroku login
heroku container:login
```
Note that Docker needs to be installed on your local system before login to heroku's container registry.

8. Using the Dockerfile, push the docker image onto Heroku's container registry using the following command:
```
heroku container:push web -a app-name
```

9. Release the newly pushed docker images to deploy app using the following command:
```
heroku container:release web -a app-name
```

10. After successfully deploying docker image onto Heroku, open the app from the Heroku platform and you will see the following interface designed using Streamlit:
<img src = "https://user-images.githubusercontent.com/34255556/197928966-d98530da-cb51-4db8-b052-47f51069c909.png" width="600">

11. From the image above, click on Training Data Validation first for initializing data ingestion into Heroku PostgreSQL, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197929141-3df4a3f0-6ac3-4b71-b519-479aebdec935.png" width="600">

12. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197929228-29644a4a-fb9b-48de-ba08-2b4fcb5fe2ae.png" width="600">

13. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197929274-e42dc33c-1f81-40b9-9a8f-5ae73345a872.png" width="600">

<b>Important Note</b>: 
- Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.

- Unlike stand-alone Docker containers, Heroku uses an ephemeral hard drive, meaning that files stored locally from running apps on Heroku will not persist when apps are restarted (once every 24 hours). Any files stored on disk will not be visible from one-off dynos such as a heroku run bash instance or a scheduler task because these commands use new dynos. Best practice for having persistent object storage is to leverage a cloud file storage service such as Amazon’s S3 (not part of project scope but can be considered)

## Project Instructions (Local Environment)
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/197928020-10f6bf6e-57aa-439f-b79f-8111d6e36083.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure:
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <default_database_name>} 
```
- For security reasons, this file needs to be stored in private. (Default host is postgres and user is postgres for PostgreSQL)
- Note that you will need to install PostgreSQL if not available in your local system: https://www.postgresql.org/download/windows/
- Ensure that PostgreSQL services is running on local system as shown in image below:
![image](https://user-images.githubusercontent.com/34255556/197931808-220d182b-4e4e-4d9c-97ba-3a20c102d501.png)

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

7. Overwrite _tree.py scripts in relevant directory (<b>env/env-name/lib/site-packages/shap/explainers</b>) where the original file is located.

8. After installing all the required Python libraries, run the following command on your project directory: 
```
streamlit run pipeline_api.py
```

9. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/197928966-d98530da-cb51-4db8-b052-47f51069c909.png" width="600">

10. From the image above, click on Training Data Validation first for initializing data ingestion into PostgreSQL, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197929141-3df4a3f0-6ac3-4b71-b519-479aebdec935.png" width="600">

11. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197929228-29644a4a-fb9b-48de-ba08-2b4fcb5fe2ae.png" width="600">

12. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197929274-e42dc33c-1f81-40b9-9a8f-5ae73345a872.png" width="600">

## Initial Data Cleaning and Feature Engineering
---
After performing Exploratory Data Analysis, the following steps are performed initially on the entire dataset before performing further data preprocessing and model training:

i) Filter data where Sales at Cost is at least greater than or equal to zero value.  Further investigation will be required to validate those records before including those as part of the analysis.

ii) Ensure NSU, MRP and SP values are strictly non-negative. Gross Sales variable which is multiplication of MRP and NSU will need to be recalculated for affected values.

iii) Filter data where SP (Selling Price) is less than or equal to MRP (Maximum retail price), since it is illegal for retailers to set retail prices beyond the maximum possible retail price.

iv) Derive datetime related features from Fdate column (year, month and quarter)

v) Removing list of irrelevant colummns identified from dataset (i.e. unique identifier features and features that exist only after target variable is derived)

vi) Checking for duplicated rows and remove if exist

vii) Split dataset into features and target labels.

viii) Save reduced set of features and target values into 2 different CSV files (X.csv and y.csv) for further data preprocessing with pipelines to reduce data leakage.

For more details of which features have been initially removed from the dataset, refer to the following CSV file: <b>Columns_Drop_from_Original.csv</b>

## Machine Learning Pipelines Configuration
---
While data preprocessing steps can be done on the entire dataset before model training, it is highly recommended to perform all data preprocessing steps within cross validation using pipelines to reduce the risk of data leakage, where information from training data is leaked to validation/test data.

The sections below summarizes the details of Machine Learning pipelines with various variations in steps:

#### i. Handling outliers by capping at extreme values
Machine learning models like Ridge, Lasso, ElasticNet and Linear SVR are highly sensitive to outliers, which may impact model performance. For those 4 types of models, the presence of this step of the pipeline by capping outliers at extreme ends of gaussian/non-gaussian distribution will be tested accordingly using Winsorizer function from feature-engine library.

Note that Anderson test is used to identify gaussian vs non-gaussian distribution in this pipeline step.

#### ii. Gaussian transformation on non-gaussian variables
In Machine Learning, several machine learning models like Huber Regression, Ridge, Lasso and ElasticNet tends to perform best when data follows the assumption of normal distribution. The following types of gaussian transformation are tested on non-gaussian features and the gaussian transformation that works best on given feature (the lowest test statistic that is smaller than 5% critical value) will be used for gaussian transformation:

- Logarithmic
- Reciprocal
- Square Root
- Yeo Johnson
- Square
- Quantile (Normal distribution)

Note that Anderson test is used to identify whether a given gaussian transformation technique successfully converts a non-gaussian feature to a gaussian feature.

#### iii. Encoding features with rare categories
For features that contain many unique categories (i.e. NAME, Brand and MC), these features may have categories that are considered to be "rare" due to low frequency. To reduce the cardinality of these features, <b>RareLabelEncoder</b> from feature-engine library is used.

#### iv. Feature Engineering
The following additional features are derived:
- Mean of 'NSU','Sales at Cost', 'Gross Sales' and 'MRP' per ZONE
- Standard deviation of 'NSU','Sales at Cost', 'Gross Sales' and 'MRP' per ZONE
- Median of 'NSU','Sales at Cost', 'Gross Sales' and 'MRP' per ZONE
- Mean of 'NSU','Sales at Cost', 'Gross Sales' and 'MRP' per Brand
- Standard deviation of 'NSU','Sales at Cost', 'Gross Sales' and 'MRP' per Brand
- Median of 'NSU','Sales at Cost', 'Gross Sales' and 'MRP' per Brand
- Number of unique materical category (MC) per brand

#### v. Encoding "interval" features
Features that are identifed as interval data types have equal magnitudes between different values. These features can be encoded directly using custom <b>label encoding (from 0)</b>:
- Fdate_Year

#### vi. Encoding "nominal" features
The following list of features are identified as nominal:
- NAME
- ZONE
- Brand
- MC

<b>For all the features mentioned in this section, features are encoded using either One Hot encoding (for non-tree based models) or CatBoost encoding (for tree-based models).</b>

#### vii. Encoding "time-related" features
All time-related features that are derived from Fdate (month and quarter) are encoded using either CyclicalFeatures (for non-tree based models) function from feature-engine library or CatBoost encoding (for tree-based models).

Note that while One Hot encoding is the most popular approach for categorical data encoding, time-related features are usually cyclical in nature such that performing one hot encoding on time related features does not capture the cyclical component.

#### viii. Feature Scaling
Feature scaling is only essential in some machine learning models like Huber Regression, Ridge, Lasso, ElasticNet and Linear SVR for faster convergence and to prevent misinterpretation of one feature significantly more important than other features. 

For this project, the following methods of feature scaling are tested:
- Standard Scaler
- MinMax Scaler
- Robust Scaler
- Standard Scaler for gaussian features + MinMax Scaler for non-gaussian features

#### x. Feature Selection
Given the current dataset has very moderately large number of features, performing feature selection is essential for simplifying the machine learning model, reducing model training time and to reduce risk of model overfitting.

For this project, the following methods of feature selection are tested:
- Mutual Information
- ANOVA
- Feature Importance using Extra Trees Classifier
- Logistic Regression with Lasso Penalty (l1)
- FeatureWiz (SULOV (Searching for Uncorrelated List of Variables) + Recursive Feature Elimination with XGBoost Classifier)

## Legality
---
This is an internship project made with 360DIGITMG for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
