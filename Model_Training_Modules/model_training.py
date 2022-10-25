'''
Author: Liaw Yi Xian
Last Modified: 25th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import optuna
import joblib
import time
import shap
from category_encoders import CatBoostEncoder
from featurewiz import FeatureWiz
from yellowbrick.regressor import CooksDistance
import feature_engine.selection as fes
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine.creation import CyclicalFeatures
import feature_engine.outliers as feo
import feature_engine.transformation as fet
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import cross_validate, KFold, learning_curve
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, make_scorer
from Application_Logger.logger import App_Logger


random_state=120


class model_trainer:


    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of model_trainer class
            Output: None

            Parameters:
            - file_object: String path of logging text file
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.optuna_selectors = {
            'HuberRegressor': {'obj': model_trainer.huber_objective,'reg': HuberRegressor()},
            'Ridge': {'obj': model_trainer.ridge_objective,'reg': Ridge(random_state=random_state)},
            'Lasso': {'obj': model_trainer.lasso_objective,'reg': Lasso(random_state=random_state)},
            'ElasticNet': {'obj': model_trainer.elasticnet_objective,'reg': ElasticNet(random_state=random_state)},
            'LinearSVR': {'obj': model_trainer.svr_objective, 'reg': LinearSVR(random_state=random_state)},
            'DecisionTreeRegressor': {'obj': model_trainer.dt_objective, 'reg': DecisionTreeRegressor(random_state=random_state)},
            'RandomForestRegressor': {'obj': model_trainer.rf_objective, 'reg': RandomForestRegressor(random_state=random_state)},
            'ExtraTreesRegressor': {'obj': model_trainer.et_objective, 'reg': ExtraTreesRegressor(random_state=random_state)},
            'AdaBoostRegressor': {'obj': model_trainer.adaboost_objective, 'reg': AdaBoostRegressor(random_state=random_state)},
            'HistGradientBoostingRegressor': {'obj': model_trainer.gradientboost_objective, 'reg': HistGradientBoostingRegressor(random_state=random_state)},
            'XGBRegressor': {'obj': model_trainer.xgboost_objective, 'reg': XGBRegressor(random_state=random_state)},
            'LGBMRegressor': {'obj': model_trainer.lightgbm_objective, 'reg': LGBMRegressor(random_state=random_state)},
            'CatBoostRegressor': {'obj': model_trainer.catboost_objective,'reg': CatBoostRegressor(random_state=random_state)}
        }


    def setting_attributes(trial, cv_results):
        '''
            Method Name: setting_attributes
            Description: This method sets attributes of metric results for training and validation set from a given Optuna trial
            Output: None

            Parameters:
            - trial: Optuna trial object
            - cv_results: Dictionary object related to results from cross validate function
        '''
        trial.set_user_attr("train_rmse", np.nanmean(cv_results['train_rmse']))
        trial.set_user_attr("val_rmse", cv_results['test_rmse'].mean())
        trial.set_user_attr(
            "train_mean_ae", np.nanmean(cv_results['train_mean_ae']))
        trial.set_user_attr("val_mean_ae", cv_results['test_mean_ae'].mean())
        trial.set_user_attr(
            "train_median_ae", np.nanmean(cv_results['train_median_ae']))
        trial.set_user_attr(
            "val_median_ae", cv_results['test_median_ae'].mean())


    def pipeline_feature_selection_step(
            pipeline, trial, fs_method, drop_correlated, reg, scaling_indicator='no'):
        '''
            Method Name: pipeline_feature_selection_step
            Description: This method adds custom transformer with FeatureSelectionTransformer class into pipeline for performing feature selection.
            Output: None
    
            Parameters:
            - pipeline: imblearn pipeline object
            - trial: Optuna trial object
            - fs_method: String name indicating method of feature selection
            - drop_correlated: String indicator of dropping highly correlated features (yes or no)
            - reg: Model object
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust', 'Combine' and 'no'). Default value is 'no'
        '''
        if fs_method not in ['FeatureWiz']:
            number_to_select = trial.suggest_int('number_features',1,30)
        else:
            number_to_select = None
        trial.set_user_attr("number_features", number_to_select)
        pipeline.steps.append(
            ('featureselection',FeatureSelectionTransformer(fs_method, reg, drop_correlated = drop_correlated, scaling_indicator = scaling_indicator, number = number_to_select)))


    def pipeline_setup(pipeline, trial, reg):
        '''
            Method Name: pipeline_setup
            Description: This method configures pipeline for model training, which varies depending on model class and preprocessing related parameters selected by Optuna.
            Output: None
    
            Parameters:
            - pipeline: sklearn pipeline object
            - trial: Optuna trial object
            - reg: Model object
        '''
        continuous_columns = ['NSU','Sales at Cost', 'Gross Sales', 'MRP']
        outlier_indicator = trial.suggest_categorical('outlier_indicator',['capped','retained']) if type(reg).__name__ in ['Ridge','Lasso','ElasticNet', 'LinearSVR'] else 'retained'
        if outlier_indicator == 'capped':
            pipeline.steps.append(
                ['outlier_capping',OutlierCapTransformer(continuous_columns)])
        if type(reg).__name__ in ['HuberRegressor', 'Ridge','Lasso','ElasticNet']:
            pipeline.steps.append(
                ['gaussian_transform',GaussianTransformer(continuous_columns)])
        nominal_columns = ['NAME','ZONE','Brand','MC']
        pipeline.steps.append(
            ['rare_data',RareLabelEncoder(variables=nominal_columns)])
        pipeline.steps.append(['feature_engine',FeatureEngineTransformer()])
        pipeline.steps.append(['interval_encoding', IntervalDataTransformer()])
        cyclic_columns = ['Fdate_month','Fdate_quarter']
        if type(reg).__name__ in ['HuberRegressor', 'Ridge','Lasso','ElasticNet', 'LinearSVR']:
            pipeline.steps.append(
                ['nominal_encoding',OneHotEncoder(variables = nominal_columns)])
            pipeline.steps.append(
                ['Cyclic_encoding', CyclicalFeatures(variables = cyclic_columns, drop_original=True)])
            scaling_indicator = trial.suggest_categorical(
                'scaling',['Standard','MinMax','Robust','Combine'])
            pipeline.steps.append(
                ('scaling',ScalingTransformer(scaling_indicator)))
        else:
            pipeline.steps.append(
                ['nominal_encoding',CatBoostEncodingTransformer( nominal_columns)])
            pipeline.steps.append(
                ['Cyclic_encoding',CatBoostEncodingTransformer( cyclic_columns)])
            scaling_indicator = 'no'
        fs_method = trial.suggest_categorical(
            'feature_selection',['Lasso','FeatureImportance_ET','MutualInformation','ANOVA','FeatureWiz'])
        if fs_method != 'FeatureWiz':
            drop_correlated = trial.suggest_categorical(
                'drop_correlated',['yes','no'])
        else:
            drop_correlated = 'no'
        model_trainer.pipeline_feature_selection_step(
            pipeline, trial, fs_method, drop_correlated, reg,scaling_indicator=scaling_indicator)
        trial.set_user_attr("outlier_indicator", outlier_indicator)
        trial.set_user_attr("scaling_indicator", scaling_indicator)
        trial.set_user_attr("drop_correlated", drop_correlated)
        trial.set_user_attr("feature_selection", fs_method)
        trial.set_user_attr("Pipeline", pipeline) 


    def huber_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: huber_objective
            Description: This method sets the objective function for huber regression model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        epsilon = trial.suggest_float('epsilon',1.01,2,log=True)
        alpha = trial.suggest_float('alpha',0.000001,1,log=True)
        max_iter = trial.suggest_categorical('max_iter',[10000])
        reg = HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha = alpha)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def ridge_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: ridge_objective
            Description: This method sets the objective function for Ridge regression model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        alpha = trial.suggest_float('alpha',0.000001,2,log=True)
        reg = Ridge(alpha = alpha, random_state=random_state)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def lasso_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: lasso_objective
            Description: This method sets the objective function for Lasso regression model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        alpha = trial.suggest_float('alpha',0.000001,2,log=True)
        max_iter = trial.suggest_categorical('max_iter',[10000])
        reg = Lasso(alpha = alpha, random_state=random_state, max_iter=max_iter)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def elasticnet_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: elasticnet_objective
            Description: This method sets the objective function for ElasticNet regression model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        alpha = trial.suggest_float('alpha',0.000001,2,log=True)
        l1_ratio = trial.suggest_float('l1_ratio',0.000001,0.999999,log=True)
        max_iter = trial.suggest_categorical('max_iter',[10000])
        reg = ElasticNet(
            alpha = alpha, random_state=random_state, max_iter=max_iter, l1_ratio=l1_ratio)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def svr_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: svr_objective
            Description: This method sets the objective function for linear support vector regressor model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        C = trial.suggest_float('C',0.0001,1,log=True)
        loss = trial.suggest_categorical('loss',['squared_epsilon_insensitive'])
        dual = trial.suggest_categorical('dual',[False])
        max_iter = trial.suggest_categorical('max_iter',[10000])
        reg = LinearSVR(
            C=C, random_state=random_state, dual=dual, loss=loss, max_iter=max_iter)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def dt_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: dt_objective
            Description: This method sets the objective function for Decision Tree regressor model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation
    
            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',50,75)
        criterion = trial.suggest_categorical(
            'criterion',['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        reg = DecisionTreeRegressor(
            random_state=random_state, ccp_alpha=ccp_alpha, criterion=criterion)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def rf_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: rf_objective
            Description: This method sets the objective function for Random Forest regressor model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',50,75)
        n_jobs = trial.suggest_categorical('n_jobs',[1])
        n_estimators = trial.suggest_categorical('n_estimators',[50])
        max_depth = trial.suggest_categorical('max_depth',[10])
        reg = RandomForestRegressor(
            random_state=random_state, ccp_alpha=ccp_alpha, n_jobs=n_jobs,n_estimators=n_estimators, max_depth=max_depth)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def et_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: et_objective
            Description: This method sets the objective function for Extra Trees regressor model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',50,75)
        n_jobs = trial.suggest_categorical('n_jobs',[1])
        n_estimators = trial.suggest_categorical('n_estimators',[50])
        max_depth = trial.suggest_categorical('max_depth',[10])
        reg = ExtraTreesRegressor(
            random_state=random_state, ccp_alpha=ccp_alpha, n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def adaboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: adaboost_objective
            Description: This method sets the objective function for AdaBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        learning_rate = trial.suggest_float('learning_rate',0.01,1,log=True)
        n_estimators = trial.suggest_categorical('n_estimators',[50])
        loss = trial.suggest_categorical(
            'loss',['linear', 'square', 'exponential'])
        reg = AdaBoostRegressor(
            learning_rate=learning_rate, random_state=random_state, n_estimators=n_estimators, loss=loss)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def gradientboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: gradientboost_objective
            Description: This method sets the objective function for Hist Gradient Boosting regressor model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        learning_rate = trial.suggest_float('learning_rate',0.01,1,log=True)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes',2,50)
        max_depth = trial.suggest_int('max_depth',2,10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',20,100)
        l2_regularization = trial.suggest_float('l2_regularization',0,1)
        reg = HistGradientBoostingRegressor(
            random_state=random_state, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def xgboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: xgboost_objective
            Description: This method sets the objective function for XGBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        booster = trial.suggest_categorical('booster',['gbtree','dart'])
        rate_drop = trial.suggest_float('rate_drop',0.0001,1,log=True) if booster == 'dart' else None
        eta = trial.suggest_float('eta',0.1,0.5,log=True)
        gamma = trial.suggest_float('gamma',0.1,20,log=True)
        min_child_weight = trial.suggest_float(
            'min_child_weight',0.1,1000,log=True)
        max_depth = trial.suggest_int('max_depth',1,10)
        lambdas = trial.suggest_float('lambda',0.1,1000,log=True)
        alpha = trial.suggest_float('alpha',0.1,100,log=True)
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        colsample_bytree = trial.suggest_float(
            'colsample_bytree',0.5,1,log=True)
        num_round = trial.suggest_categorical('num_round',[50])
        objective = trial.suggest_categorical('objective',['reg:squarederror'])
        eval_metric = trial.suggest_categorical('eval_metric',['rmse'])
        verbosity = trial.suggest_categorical('verbosity',[0])
        tree_method = trial.suggest_categorical('tree_method',['gpu_hist'])
        single_precision_histogram = trial.suggest_categorical(
            'single_precision_histogram',[True])
        reg = XGBRegressor(
            objective=objective, eval_metric=eval_metric, verbosity=verbosity,tree_method = tree_method, booster=booster, eta=eta, gamma=gamma,single_precision_histogram=single_precision_histogram,  min_child_weight=min_child_weight, max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree, lambdas=lambdas, alpha=alpha, random_state=random_state, num_round=num_round, rate_drop=rate_drop)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def lightgbm_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: lightgbm_objective
            Description: This method sets the objective function for LightGBM model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        max_depth = trial.suggest_int('max_depth',3,12)
        num_leaves = trial.suggest_int('num_leaves',8,4096)
        min_child_samples = trial.suggest_int('min_child_samples',5,100)
        boosting_type = trial.suggest_categorical(
            'boosting_type',['gbdt','dart'])
        drop_rate = trial.suggest_float('drop_rate',0.0001,1,log=True) if boosting_type == 'dart' else None
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        subsample_freq = trial.suggest_int('subsample_freq',1,10)
        reg_alpha = trial.suggest_float('reg_alpha',0.1,100,log=True)
        reg_lambda = trial.suggest_float('reg_lambda',0.1,100,log=True)
        min_split_gain = trial.suggest_float('min_split_gain',0.1,15,log=True)
        max_bin = trial.suggest_categorical("max_bin", [63])
        device_type = trial.suggest_categorical('device_type',['gpu'])
        gpu_use_dp = trial.suggest_categorical('gpu_use_dp',[False])
        reg = LGBMRegressor(
            num_leaves=num_leaves, learning_rate=learning_rate, boosting_type=boosting_type, max_depth=max_depth, min_child_samples = min_child_samples, max_bin=max_bin, reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample = subsample, subsample_freq = subsample_freq, min_split_gain=min_split_gain, random_state=random_state, device_type=device_type,gpu_use_dp=gpu_use_dp, drop_rate=drop_rate, drop_seed = random_state)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def catboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: catboost_objective
            Description: This method sets the objective function for CatBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents RMSE score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        max_depth = trial.suggest_int('max_depth',4,10)
        l2_leaf_reg = trial.suggest_int('l2_leaf_reg',2,10)
        random_strength = trial.suggest_float('random_strength',0.1,10,log=True)
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        boosting_type = trial.suggest_categorical('boosting_type',['Plain'])
        loss_function = trial.suggest_categorical('loss_function',['RMSE'])
        nan_mode = trial.suggest_categorical('nan_mode',['Min'])
        task_type = trial.suggest_categorical('task_type',['GPU'])
        iterations = trial.suggest_categorical('iterations',[50])
        verbose = trial.suggest_categorical('verbose',[False])
        reg = CatBoostRegressor(
            max_depth = max_depth, l2_leaf_reg = l2_leaf_reg, learning_rate=learning_rate, random_strength=random_strength, boosting_type = boosting_type, loss_function=loss_function,nan_mode=nan_mode,random_state=random_state,task_type=task_type, iterations=iterations, verbose=verbose)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, reg)
        cv_results = model_trainer.classification_metrics(
            reg,pipeline,X_train_data,y_train_data,cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_rmse'])


    def classification_metrics(
            reg,pipeline,X_train_data,y_train_data, cv_jobs, fit_params=None):
        '''
            Method Name: classification_metrics
            Description: This method performs 3-fold cross validation on the training set and performs model evaluation on the validation set.
            Output: Dictionary of metric scores from 3-fold cross validation.

            Parameters:
            - reg: Model object
            - pipeline: imblearn pipeline object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - cv_jobs: Number of cross validation jobs to run in parallel
            - fit_params: Additional parameters passed to fit method of cross_validate function in the form of dictionary
        '''
        pipeline_copy = clone(pipeline)
        pipeline_copy.steps.append(('reg',reg))
        cv_results = cross_validate(
            pipeline_copy, X_train_data, y_train_data, cv=3,
            scoring={"rmse": make_scorer(mean_squared_error, squared=False),
            "mean_ae": make_scorer(mean_absolute_error),
            "median_ae": make_scorer(median_absolute_error)},
            n_jobs=cv_jobs,return_train_score=True,error_score='raise',fit_params=fit_params)
        return cv_results


    def optuna_optimizer(self, obj, n_trials, fold):
        '''
            Method Name: optuna_optimizer
            Description: This method creates a new Optuna study object if the given Optuna study object doesn't exist or otherwise using existing Optuna study object and optimizes the given objective function. In addition, the following plots and results are also created and saved:
            1. Hyperparameter Importance Plot
            2. Optimization History Plot
            3. Optuna study object
            4. Optimization Results (csv format)
            
            Output: Single best trial object
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold: Fold number from nested cross-validation in outer loop
        '''
        try:
            if f"OptStudy_{obj.__name__}_Fold_{fold}.pkl" in os.listdir(self.folderpath+obj.__name__):
                study = joblib.load(
                    self.folderpath+obj.__name__+f"/OptStudy_{obj.__name__}_Fold_{fold}.pkl")
            else:
                sampler = optuna.samplers.TPESampler(
                    multivariate=True, seed=random_state)
                study = optuna.create_study(
                    direction='minimize',sampler=sampler)
            study.optimize(
                obj, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
            trial = study.best_trial
            if trial.number !=0:
                param_imp_fig = optuna.visualization.plot_param_importances(study)
                opt_fig = optuna.visualization.plot_optimization_history(study)
                param_imp_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/HP_Importances_{obj.__name__}_Fold_{fold}.png')
                opt_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/Optimization_History_{obj.__name__}_Fold_{fold}.png')
            joblib.dump(
                study, self.folderpath + obj.__name__ + f'/OptStudy_{obj.__name__}_Fold_{fold}.pkl')
            study.trials_dataframe().to_csv(
                self.folderpath + obj.__name__ + f"/Hyperparameter_Tuning_Results_{obj.__name__}_Fold_{fold}.csv",index=False)
            del study
        except Exception as e:
            self.log_writer.log(
                self.file_object, f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
            raise Exception(
                f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
        return trial


    def residual_plot(self, reg, figtitle, plotname, actual_value, pred_value):
        '''
            Method Name: residual_plot
            Description: This method plots residuals from the model and saves plot within the given model class folder.
            Output: None

            Parameters:
            - reg: Model object
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_value: Actual target values from dataset
            - pred_value: Predicted target values
        '''
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x = pred_value, y= np.subtract(pred_value,actual_value))
        plt.axhline(y=0, color='black')
        plt.title(f'Residual plot for {type(reg).__name__} {figtitle}')
        plt.ylabel('Residuals')
        plt.xlabel('Predicted Value')
        plt.savefig(
            self.folderpath+type(reg).__name__+f'/Residual_Plot_{type(reg).__name__}_{plotname}.png',bbox_inches='tight')
        plt.clf()


    def leverage_plot(self, reg, input_data, output_data):
        '''
            Method Name: leverage_plot
            Description: This method plots leverage plot of a given model and saves plot within the given model class folder.
            Output: None

            Parameters:
            - reg: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        visualizer = CooksDistance()
        visualizer.fit(input_data, output_data)
        visualizer.show(
            outpath=self.folderpath+type(reg).__name__+'/Leverage_Plot.png', clear_figure=True)


    def learning_curve_plot(self, reg, input_data, output_data):
        '''
            Method Name: learning_curve_plot
            Description: This method plots learning curve of 5 fold cross validation and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator = reg, X = input_data, y = output_data, cv= KFold(n_splits=5, shuffle=True, random_state=random_state), scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.3, 1.0, 10))
        train_scores = np.abs(train_scores)
        validation_scores = np.abs(validation_scores)
        plt.style.use('seaborn-whitegrid')
        plt.grid(True)
        plt.fill_between(train_sizes, train_scores.mean(axis = 1) - train_scores.std(axis = 1), train_scores.mean(axis = 1) + train_scores.std(axis = 1), alpha=0.25, color='blue')
        plt.plot(train_sizes, train_scores.mean(axis = 1), label = 'Training Score', marker='.',markersize=14)
        plt.fill_between(train_sizes, validation_scores.mean(axis = 1) - validation_scores.std(axis = 1), validation_scores.mean(axis = 1) + validation_scores.std(axis = 1), alpha=0.25, color='green')
        plt.plot(train_sizes, validation_scores.mean(axis = 1), label = 'Cross Validation Score', marker='.',markersize=14)
        plt.ylabel('Score')
        plt.xlabel('Training instances')
        plt.title(f'Learning Curve for {type(reg).__name__}')
        plt.legend(frameon=True, loc='best')
        plt.savefig(
            self.folderpath+type(reg).__name__+f'/LearningCurve_{type(reg).__name__}.png',bbox_inches='tight')
        plt.clf()


    def shap_plot(self, reg, input_data):
        '''
            Method Name: shap_plot
            Description: This method plots feature importance and its summary using shap values and saves plot within the given model class folder. Note that this function will not work specifically for XGBoost models that use 'dart' booster.
            Output: None

            Parameters:
            - reg: Model object
            - input_data: Features from dataset
        '''
        if type(reg).__name__ in ['HuberRegressor','LinearSVR', 'Ridge', 'Lasso', 'ElasticNet']:
            explainer = shap.LinearExplainer(reg, input_data)
            explainer_obj = explainer(input_data)
            shap_values = explainer.shap_values(input_data)
        else:
            if ('dart' in reg.get_params().values()) and (type(reg).__name__ == 'XGBRegressor'):
                return
            explainer = shap.TreeExplainer(reg)
            explainer_obj = explainer(input_data)
            shap_values = explainer.shap_values(input_data)
        plt.figure()
        shap.summary_plot(
            shap_values, input_data, plot_type="bar", show=False, max_display=40)
        plt.title(f'Shap Feature Importances for {type(reg).__name__}')
        plt.savefig(
            self.folderpath+type(reg).__name__+f'/Shap_Feature_Importances_{type(reg).__name__}.png',bbox_inches='tight')
        plt.clf()
        plt.figure()
        shap.plots.beeswarm(explainer_obj, show=False, max_display=40)
        plt.title(f'Shap Summary Plot for {type(reg).__name__}')
        plt.savefig(
            self.folderpath+type(reg).__name__+f'/Shap_Summary_Plot_{type(reg).__name__}.png',bbox_inches='tight')
        plt.clf()


    def model_training(
            self, reg, obj, input_data, output_data, n_trials, fold_num):
        '''
            Method Name: model_training
            Description: This method performs Optuna hyperparameter tuning using 3 fold cross validation on given dataset. The best hyperparameters with the best pipeline identified is used for model training.
            
            Output: 
            - model_copy: Trained model object
            - best_trial: Optuna's best trial object from hyperparameter tuning
            - input_data_transformed: Transformed features from dataset
            - best_pipeline: sklearn pipeline object

            On Failure: Logging error and raise exception

            Parameters:
            - reg: Model object
            - obj: Optuna objective function
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold_num: Indication of fold number for model training (can be integer or string "overall")
        '''
        func = lambda trial: obj(trial, input_data, output_data)
        func.__name__ = type(reg).__name__
        self.log_writer.log(
            self.file_object, f"Start hyperparameter tuning for {type(reg).__name__} for fold {fold_num}")
        best_trial = self.optuna_optimizer(func, n_trials, fold_num)
        self.log_writer.log(
            self.file_object, f"Hyperparameter tuning for {type(reg).__name__} completed for fold {fold_num}")
        self.log_writer.log(
            self.file_object, f"Start using best pipeline for {type(reg).__name__} for transforming training and validation data for fold {fold_num}")
        best_pipeline = best_trial.user_attrs['Pipeline']
        input_data_transformed = best_pipeline.fit_transform(
            input_data, output_data)
        self.log_writer.log(
            self.file_object, f"Finish using best pipeline for {type(reg).__name__} for transforming training and validation data for fold {fold_num}")
        for parameter in ['scaling','feature_selection','number_features','drop_correlated','outlier_indicator']:
            if parameter in best_trial.params.keys():
                best_trial.params.pop(parameter)
        self.log_writer.log(
            self.file_object, f"Finish hyperparameter tuning for {type(reg).__name__} for fold {fold_num}")
        model_copy = clone(reg)
        model_copy = model_copy.set_params(**best_trial.params)
        model_copy.fit(input_data_transformed, output_data)
        return model_copy, best_trial, input_data_transformed, best_pipeline


    def hyperparameter_tuning(
            self, obj, reg, n_trials, input_data, output_data):
        '''
            Method Name: hyperparameter_tuning
            Description: This method performs Nested 3 Fold Cross Validation on the entire dataset, where the inner loop (3-fold) performs Optuna hyperparameter tuning and the outer loop (5-fold) performs model evaluation to obtain overall generalization error of model. The best hyperparameters with the best pipeline identified from inner loop is used for model training on the entire training set and model evaluation on the test set for the outer loop.
            In addition, the following intermediate results are saved for a given model class:
            1. Model_Performance_Results_by_Fold (csv file)
            2. Overall_Model_Performance_Results (csv file)
            3. Residual Plot image
            
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - reg: Model object
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        try:
            num_folds = 5
            skfold = KFold(
                n_splits=num_folds, shuffle=True, random_state=random_state)
            rmse_train_cv, mean_ae_train_cv, median_ae_train_cv= [], [], []
            rmse_val_cv, mean_ae_val_cv, median_ae_val_cv= [], [], []
            rmse_test_cv, mean_ae_test_cv, median_ae_test_cv= [], [], []
            actual_values, pred_values = [], []
            for fold, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(input_data, output_data)):
                input_sub_train_data = input_data.iloc[outer_train_idx,:].reset_index(drop=True)
                output_sub_train_data = output_data.iloc[outer_train_idx].reset_index(drop=True)
                model_copy, best_trial, input_train_data_transformed, best_pipeline = self.model_training(reg, obj, input_sub_train_data, output_sub_train_data, n_trials, fold+1)
                input_val_data = input_data.iloc[outer_valid_idx,:].reset_index(drop=True)
                input_val_data_transformed = best_pipeline.transform(input_val_data)
                val_pred = model_copy.predict(input_val_data_transformed)
                actual_values.extend(output_data.iloc[outer_valid_idx].tolist())
                pred_values.extend(val_pred)
                rmse_outer_val_value = mean_squared_error(
                    np.array(output_data.iloc[outer_valid_idx]), val_pred, squared=False)
                mean_ae_outer_val_value = mean_absolute_error(
                    np.array(output_data.iloc[outer_valid_idx]), val_pred)
                median_ae_outer_val_value = median_absolute_error(
                    np.array(output_data.iloc[outer_valid_idx]), val_pred)
                cv_lists = [rmse_train_cv, mean_ae_train_cv, median_ae_train_cv, rmse_val_cv, mean_ae_val_cv, median_ae_val_cv, rmse_test_cv, mean_ae_test_cv, median_ae_test_cv]
                metric_values = [best_trial.user_attrs['train_rmse'], best_trial.user_attrs['train_mean_ae'], best_trial.user_attrs['train_median_ae'], best_trial.user_attrs['val_rmse'], best_trial.user_attrs['val_mean_ae'], best_trial.user_attrs['val_median_ae'], rmse_outer_val_value, mean_ae_outer_val_value, median_ae_outer_val_value]
                for cv_list, metric in zip(cv_lists, metric_values):
                    cv_list.append(metric)
                self.log_writer.log(
                    self.file_object, f"Evaluating model performance for {type(reg).__name__} on validation set completed for fold {fold+1}")
                optimized_results = pd.DataFrame({
                    'Feature_selector':best_trial.user_attrs['feature_selection'], 'Drop_correlated_features': best_trial.user_attrs['drop_correlated'], 'Models': type(model_copy).__name__, 'Best_params': str(model_copy.get_params()), 'Number_features': [len(input_train_data_transformed.columns.tolist())], 'Features': [input_train_data_transformed.columns.tolist()], 'Outlier_handling_method': best_trial.user_attrs['outlier_indicator'], 'Feature_scaling_handled': best_trial.user_attrs['scaling_indicator'], 'Outer_fold': fold+1,'rmse_inner_train_cv': best_trial.user_attrs['train_rmse'], 'rmse_inner_val_cv': best_trial.user_attrs['val_rmse'], 'rmse_outer_val_cv': [rmse_outer_val_value],'mean_ae_inner_train_cv': best_trial.user_attrs['train_mean_ae'], 'mean_ae_inner_val_cv': best_trial.user_attrs['val_mean_ae'], 'mean_ae_outer_val_cv': [mean_ae_outer_val_value],'median_ae_inner_train_cv': best_trial.user_attrs['train_median_ae'], 'median_ae_inner_val_cv': best_trial.user_attrs['val_median_ae'], 'median_ae_outer_val_cv': [median_ae_outer_val_value]})
                optimized_results.to_csv(
                    self.folderpath+'Model_Performance_Results_by_Fold.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Model_Performance_Results_by_Fold.csv'))
                self.log_writer.log(
                    self.file_object, f"Optimized results for {type(reg).__name__} model saved for fold {fold+1}")
                time.sleep(10)
            average_results = pd.DataFrame({
                'Models': type(model_copy).__name__, 'rmse_train_cv_avg': np.mean(rmse_train_cv), 'rmse_train_cv_std': np.std(rmse_train_cv), 'rmse_val_cv_avg': np.mean(rmse_val_cv), 'rmse_val_cv_std': np.std(rmse_val_cv), 'rmse_test_cv_avg': np.mean(rmse_test_cv), 'rmse_test_cv_std': np.std(rmse_test_cv), 'mean_ae_train_cv_avg': np.mean(mean_ae_train_cv), 'mean_ae_train_cv_std': np.std(mean_ae_train_cv), 'mean_ae_val_cv_avg': np.mean(mean_ae_val_cv), 'mean_ae_val_cv_std': np.std(mean_ae_val_cv), 'mean_ae_test_cv_avg': np.mean(mean_ae_test_cv), 'mean_ae_test_cv_std': np.std(mean_ae_test_cv), 'median_ae_train_cv_avg': np.mean(median_ae_train_cv), 'median_ae_train_cv_std': np.std(median_ae_train_cv), 'median_ae_val_cv_avg': np.mean(median_ae_val_cv),'median_ae_val_cv_std': np.std(median_ae_val_cv),'median_ae_test_cv_avg': np.mean(median_ae_test_cv),'median_ae_test_cv_std': np.std(median_ae_test_cv)}, index=[0])
            self.residual_plot(
                reg, '(from CV)', 'from_CV', actual_values, pred_values)
            average_results.to_csv(
                self.folderpath+'Overall_Model_Performance_Results.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Overall_Model_Performance_Results.csv'))
            self.log_writer.log(
                self.file_object, f"Average optimized results for {type(reg).__name__} model saved")                
        except Exception as e:
            self.log_writer.log(
                self.file_object, f'Hyperparameter tuning on {type(reg).__name__} model failed with the following error: {e}')
            raise Exception(
                f'Hyperparameter tuning on {type(reg).__name__} model failed with the following error: {e}')


    def final_overall_model(self, obj, reg, input_data, output_data, n_trials):
        '''
            Method Name: final_overall_model
            Description: This method performs hyperparameter tuning on best model algorithm identified using 3 fold cross validation on entire dataset. The best hyperparameters identified are then used to train the entire dataset before saving model for deployment.
            In addition, the following intermediate results are saved for a given model class:
            1. Residual Plot image
            2. Learning Curve image
            3. Leverage Plot image
            4. Shap Feature Importances (barplot image)
            5. Shap Summary Plot (beeswarm plot image)
            
            Output: None

            Parameters:
            - obj: Optuna objective function
            - reg: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - n_trials: Number of trials for Optuna hyperparameter tuning
        '''
        self.log_writer.log(
            self.file_object, f"Start final model training on all data for {type(reg).__name__}")
        overall_model, best_trial, input_data_transformed, best_pipeline = self.model_training(reg, obj, input_data, output_data, n_trials, 'overall')
        joblib.dump(best_pipeline,'Saved_Models/Preprocessing_Pipeline.pkl')
        joblib.dump(overall_model,'Saved_Models/FinalModel.pkl')
        actual_values = output_data
        pred_values = overall_model.predict(input_data_transformed)
        self.residual_plot(
            reg, '(Final Model)', 'final_model', actual_values, pred_values)
        self.leverage_plot(overall_model, input_data_transformed, output_data)
        self.learning_curve_plot(
            overall_model, input_data_transformed, output_data)
        self.shap_plot(overall_model, input_data_transformed)
        self.log_writer.log(
            self.file_object, f"Finish final model training on all data for {type(reg).__name__}")
        

    def model_selection(
            self, input, output, num_trials, folderpath, model_names):
        '''
            Method Name: model_selection
            Description: This method performs model algorithm selection using Nested Cross Validation (5-fold cv outer loop for model evaluation and 3-fold cv inner loop for hyperparameter tuning)
            Output: None

            Parameters:
            - input: Features from dataset
            - output: Target column from dataset
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
            - model_names: List of model names provided as input by user for model selection
        '''
        self.log_writer.log(
            self.file_object, 'Start process of model selection')
        self.input = input
        self.output = output
        self.num_trials = num_trials
        self.folderpath = folderpath
        self.model_names = model_names
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        input_data = self.input.copy()
        output_data = self.output['SP'].copy()
        for selector in self.model_names:
            obj = self.optuna_selectors[selector]['obj']
            reg = self.optuna_selectors[selector]['reg']
            path = os.path.join(self.folderpath, type(reg).__name__)
            if not os.path.exists(path):
                os.mkdir(path)
            self.hyperparameter_tuning(
                obj = obj, reg = reg, n_trials = self.num_trials, input_data = input_data, output_data = output_data)
            time.sleep(10)
        overall_results = pd.read_csv(
            self.folderpath + 'Overall_Model_Performance_Results.csv')
        self.log_writer.log(
            self.file_object, f"Best model identified based on RMSE is {overall_results.iloc[overall_results['rmse_test_cv_avg'].idxmin()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['rmse_test_cv_avg'].idxmin()]['rmse_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['rmse_test_cv_avg'].idxmin()]['rmse_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on mean absolute error score is {overall_results.iloc[overall_results['mean_ae_test_cv_avg'].idxmin()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['mean_ae_test_cv_avg'].idxmin()]['mean_ae_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['mean_ae_test_cv_avg'].idxmin()]['mean_ae_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on median absolute error score is {overall_results.iloc[overall_results['median_ae_test_cv_avg'].idxmin()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['median_ae_test_cv_avg'].idxmin()]['median_ae_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['median_ae_test_cv_avg'].idxmin()]['median_ae_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, 'Finish process of model selection')


    def final_model_tuning(
           self, input_data, output_data, num_trials, folderpath, model_name):
        '''
            Method Name: final_model_tuning
            Description: This method performs final model training from best model algorithm identified on entire dataset using 3-fold cross validation.
            Output: None

            Parameters:
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
            - model_name: Name of model selected for final model training.
        '''
        self.input_data = input_data
        self.output_data = output_data
        self.num_trials = num_trials
        self.folderpath = folderpath
        self.model_name = model_name
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        self.log_writer.log(
            self.file_object, f"Start performing hyperparameter tuning on best model identified overall: {self.model_name}")
        obj = self.optuna_selectors[self.model_name]['obj']
        reg = self.optuna_selectors[self.model_name]['reg']
        input_data = self.input_data.copy()
        output_data = self.output_data['SP'].copy()
        self.final_overall_model(
            obj = obj, reg = reg, input_data = input_data, output_data = output_data, n_trials = self.num_trials)
        self.log_writer.log(
            self.file_object, f"Finish performing hyperparameter tuning on best model identified overall: {self.model_name}")


class CheckGaussian():
    

    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of CheckGaussian class
            Output: None
        '''
        pass


    def check_gaussian(self, X):
        '''
            Method Name: check_gaussian
            Description: This method classifies features from dataset into gaussian vs non-gaussian columns.
            Output: self
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        self.gaussian_columns = []
        self.non_gaussian_columns = []
        for column in X_.columns:
            result = st.anderson(X_[column])
            if result[0] > result[1][2]:
                self.non_gaussian_columns.append(column)
            else:
                self.gaussian_columns.append(column)
        return


class OutlierCapTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of OutlierCapTransformer class
            Output: None
            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(OutlierCapTransformer, self).__init__()
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method classifies way of handling outliers based on whether features are gaussian or non-gaussian, while fitting respective class methods from feature-engine.outliers library
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        self.check_gaussian(X_[self.continuous])
        if self.non_gaussian_columns!=[]:
            self.non_gaussian_winsorizer = feo.Winsorizer(
                capping_method='iqr', tail='both', fold=1.5, add_indicators=False,variables=self.non_gaussian_columns)
            self.non_gaussian_winsorizer.fit(X_)
        if self.gaussian_columns!=[]:
            self.gaussian_winsorizer = feo.Winsorizer(
                capping_method='gaussian', tail='both', fold=3, add_indicators=False,variables=self.gaussian_columns)
            self.gaussian_winsorizer.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective class methods from feature-engine.outliers library
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        if self.non_gaussian_columns != []:
            X_ = self.non_gaussian_winsorizer.transform(X_)
        if self.gaussian_columns != []:
            X_ = self.gaussian_winsorizer.transform(X_)
        return X_


class GaussianTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of GaussianTransformer class
            Output: None

            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(GaussianTransformer, self).__init__()
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method tests for various gaussian transformation techniques on non-gaussian variables. Non-gaussian variables that best successfully transformed to gaussian variables based on Anderson test will be used for fitting on respective gaussian transformers.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        self.check_gaussian(X_[self.continuous])
        transformer_list = [
            fet.LogTransformer(), fet.ReciprocalTransformer(), fet.PowerTransformer(exp=0.5), fet.YeoJohnsonTransformer(), fet.PowerTransformer(exp=2), QuantileTransformer(output_distribution='normal')
        ]
        transformer_names = [
            'logarithmic','reciprocal','square-root','yeo-johnson','square','quantile'
        ]
        result_names, result_test_stats, result_columns, result_critical_value=[], [], [], []
        for transformer, name in zip(transformer_list, transformer_names):
            for column in self.non_gaussian_columns:
                try:
                    X_transformed = pd.DataFrame(
                        transformer.fit_transform(X_[[column]]), columns = [column])
                    result_columns.append(column)
                    result_names.append(name)
                    result_test_stats.append(
                        st.anderson(X_transformed[column])[0])
                    result_critical_value.append(
                        st.anderson(X_transformed[column])[1][2])
                except:
                    continue
        results = pd.DataFrame(
            [pd.Series(result_columns, name='Variable'), 
            pd.Series(result_names,name='Transformation_Type'),
            pd.Series(result_test_stats, name='Test-stats'), 
            pd.Series(result_critical_value, name='Critical value')]).T
        best_results = results[results['Test-stats']<results['Critical value']].groupby(by='Variable')[['Transformation_Type','Test-stats']].min()
        transformer_types = best_results['Transformation_Type'].unique()
        for type in transformer_types:
            variable_list = best_results[best_results['Transformation_Type'] == type].index.tolist()
            if type == 'logarithmic':
                self.logtransformer = fet.LogTransformer(variables=variable_list)
                self.logtransformer.fit(X_)
            elif type == 'reciprocal':
                self.reciprocaltransformer = fet.ReciprocalTransformer(variables=variable_list)
                self.reciprocaltransformer.fit(X_)
            elif type == 'square-root':
                self.sqrttransformer = fet.PowerTransformer(exp=0.5, variables=variable_list)
                self.sqrttransformer.fit(X_)
            elif type == 'yeo-johnson':
                self.yeojohnsontransformer = fet.YeoJohnsonTransformer(variables=variable_list)
                self.yeojohnsontransformer.fit(X_)
            elif type == 'square':
                self.squaretransformer = fet.PowerTransformer(exp=2, variables=variable_list)
                self.squaretransformer.fit(X_)
            elif type == 'quantile':
                self.quantiletransformer = QuantileTransformer(output_distribution='normal',random_state=random_state)
                self.quantiletransformer.fit(X_[variable_list])
                self.quantilevariables = variable_list
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs gaussian transformation on features using respective gaussian transformers.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        if hasattr(self, 'logtransformer'):
            try:
                X_ = self.logtransformer.transform(X_)
            except:
                old_variable_list = self.logtransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]<=0).sum()>0:
                        self.logtransformer.variables_.remove(var)
                X_ = self.logtransformer.transform(X_)
        if hasattr(self, 'reciprocaltransformer'):
            try:
                X_ = self.reciprocaltransformer.transform(X_)
            except:
                old_variable_list = self.reciprocaltransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]==0).sum()>0:
                        self.reciprocaltransformer.variables_.remove(var)
                X_ = self.reciprocaltransformer.transform(X_)
        if hasattr(self, 'sqrttransformer'):
            try:
                X_ = self.sqrttransformer.transform(X_)
            except:
                old_variable_list = self.sqrttransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]==0).sum()>0:
                        self.sqrttransformer.variables_.remove(var)
                X_ = self.sqrttransformer.transform(X_)
        if hasattr(self, 'yeojohnsontransformer'):
            X_ = self.yeojohnsontransformer.transform(X_)
        if hasattr(self, 'squaretransformer'):
            X_ = self.squaretransformer.transform(X_)
        if hasattr(self, 'quantiletransformer'):
            X_[self.quantilevariables] = pd.DataFrame(
                self.quantiletransformer.transform(X_[self.quantilevariables]), columns = self.quantilevariables)
        return X_


class FeatureEngineTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of FeatureEngineTransformer class
            Output: None
        '''
        pass


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method performs grouping of continuous features by certain categorical features.
            Output: self
        '''
        X_ = X.reset_index(drop=True).copy()
        self.Zone_grouped = X_.groupby(['ZONE'])[['NSU','Sales at Cost', 'Gross Sales', 'MRP']].agg(['mean','std','median'])
        self.Zone_grouped.columns = ['_'.join(col) + '_Zone' for col in self.Zone_grouped.columns.values]
        self.Brand_grouped = X_.groupby(['Brand'])[['NSU','Sales at Cost', 'Gross Sales', 'MRP']].agg(['mean','std','median'])
        self.Brand_grouped.columns = ['_'.join(col) + '_Brand' for col in self.Brand_grouped.columns.values]
        self.BrandMC_grouped = X_.groupby(['Brand'])['MC'].agg(['nunique'])
        self.BrandMC_grouped.columns = ['Brand_MC_nunique']
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method merges grouping of continuous features by certain categorical features with existing dataset to create additional features.
            Output: Additional features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        X_ = pd.merge(X_, self.Zone_grouped.reset_index(), how='left')
        X_ = pd.merge(X_, self.Brand_grouped.reset_index(), how='left')
        X_ = pd.merge(X_, self.BrandMC_grouped.reset_index(), how='left')
        return X_


class IntervalDataTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of IntervalDataTransformer class
            Output: None
        '''
        pass


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method simply passes the fit method of transformer without execution.
            Output: self
        '''
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs ordinal encoding on interval features.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        X_['Fdate_year'] = X_['Fdate_year'] - 2017
        return X_
    

class ScalingTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, scaler):
        '''
            Method Name: __init__
            Description: This method initializes instance of ScalingTransformer class
            Output: None

            Parameters:
            - scaler: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust' and 'Combine')
        '''
        super(ScalingTransformer, self).__init__()
        self.scaler = scaler


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method fits dataset onto respective scalers selected.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        if self.scaler == 'Standard':
            self.copyscaler = StandardScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'MinMax':
            self.copyscaler = MinMaxScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'Robust':
            self.copyscaler = RobustScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'Combine':
            self.check_gaussian(X_)
            self.copyscaler = ColumnTransformer(
                [('std_scaler',StandardScaler(),self.gaussian_columns),('minmax_scaler',MinMaxScaler(),self.non_gaussian_columns)],remainder='passthrough',n_jobs=1)
            self.copyscaler.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective scalers.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        if self.scaler != 'Combine':
            X_ = pd.DataFrame(
                self.copyscaler.transform(X_), columns = X.columns)
        else:
            X_ = pd.DataFrame(
                self.copyscaler.transform(X_), columns = self.gaussian_columns + self.non_gaussian_columns)
            X_ = X_[X.columns.tolist()]
        return X_


class CatBoostEncodingTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, columns):
        '''
            Method Name: __init__
            Description: This method initializes instance of CatBoostEncodingTransformer class
            Output: None

            Parameters:
            - columns: List of features for catboost encoding
        '''
        super(CatBoostEncodingTransformer, self).__init__()
        self.columns = columns


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method fits dataset onto catboost encoder for categorical data encoding.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        y = y.reset_index(drop=True).copy()
        self.encoder = CatBoostEncoder(cols = self.columns, drop_invariant=True)
        self.encoder.fit(X_, y)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using Catboost encoder for categorical data encoding.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        X_ = self.encoder.transform(X_)
        return X_


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(
            self, method, model, drop_correlated, scaling_indicator= 'no',  number=None):
        '''
            Method Name: __init__
            Description: This method initializes instance of FeatureSelectionTransformer class
            Output: None

            Parameters:
            - method: String that represents method of feature selection (Accepted values are 'Lasso', 'FeatureImportance_ET', 'MutualInformation', 'ANOVA', 'FeatureWiz')
            - model: Model object
            - drop_correlated: String indicator of dropping highly correlated features (yes or no)
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'MinMax' and 'no'). Default value is 'no'
            - number: Integer that represents number of features to select. Minimum value required is 1. Default value is None.

        '''
        self.method = method
        self.model = model
        self.drop_correlated = drop_correlated
        self.scaling_indicator = scaling_indicator
        self.number = number


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method removes features that are highly correlated with other features if applicable and identifies subset of columns from respective feature selection techniques
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        y = y.reset_index(drop=True)
        if self.drop_correlated == 'yes':
            self.correlated_selector = fes.DropCorrelatedFeatures(method='spearman')
            self.correlated_selector.fit(X_)
            X_ = self.correlated_selector.transform(X_)
        if self.method == 'Lasso':
            imp_model = Lasso(random_state=random_state)
            imp_model.fit(X_,y)
            if self.scaling_indicator == 'no':
                result = pd.DataFrame(
                    [pd.Series(X_.columns),pd.Series(np.abs(imp_model.coef_[0])*np.array(X_).std(axis=0))], index=['Variable','Value']).T
            else:
                result = pd.DataFrame(
                    [pd.Series(X_.columns),pd.Series(np.abs(imp_model.coef_[0]))], index=['Variable','Value']).T
            result['Value'] = result['Value'].astype('float64')
            self.sub_columns =  result.loc[result['Value'].nlargest(self.number).index.tolist()]['Variable'].tolist()
        elif self.method == 'FeatureImportance_ET':
            max_features = len(X_.columns.tolist()) if self.number > len(X_.columns.tolist()) else self.number
            fimp_model = ExtraTreesRegressor(random_state=random_state)
            fimportance_selector = SelectFromModel(
                fimp_model,max_features=max_features,threshold=0.0)
            fimportance_selector.fit(X_,y)
            self.sub_columns = X_.columns[fimportance_selector.get_support()].to_list()
        elif self.method == 'MutualInformation':
            values = mutual_info_regression(X_,y,random_state=random_state)
            result = pd.DataFrame(
                [pd.Series(X_.columns),pd.Series(values)], index=['Variable','Value']).T
            result['Value'] = result['Value'].astype('float64')
            self.sub_columns =  result.loc[result['Value'].nlargest(self.number).index.tolist()]['Variable'].tolist()
        elif self.method == 'ANOVA':
            k = 'all' if self.number > len(X_.columns.tolist()) else self.number
            fclassif_selector = SelectKBest(f_regression,k=k)
            fclassif_selector.fit(X_,y)
            self.sub_columns =  X_.columns[fclassif_selector.get_support()].to_list()
        elif self.method == 'FeatureWiz':
            selector = FeatureWiz(verbose=0)
            selector.fit(X_, y)
            self.sub_columns = selector.features
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method removes features that are highly correlated with other features if applicable and identifies subset of columns from respective feature selection techniques.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        if self.drop_correlated == 'yes':
            X_ = self.correlated_selector.transform(X_)
        if self.sub_columns != []:
            X_ = X_[self.sub_columns]
        return X_