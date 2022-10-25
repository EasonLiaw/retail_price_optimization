import streamlit as st
import pandas as pd
import joblib
import numpy as np
from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import os

DATABASE_LOG = "Training_Logs/Training_Main_Log.txt"
DATA_SOURCE = 'Training_Data_FromDB/Training_Data.csv'
PREPROCESSING_LOG = "Training_Logs/Training_Preprocessing_Log.txt"
RESULT_DIR = 'Intermediate_Train_Results/'
TRAINING_LOG = "Training_Logs/Training_Model_Log.txt"

def main():
    st.title("Retail Price Optimization")
    html_temp = """
    <div style="background-color:green;padding:3px">
    <h2 style="color:white;text-align:center;">Retail Price Optimization App </h2>
    <p></p>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    with st.expander("Model Training", expanded=True):
        if st.button("Training Data Validation"):
            trainvalidator = rawtraindatavalidation(
                table_name = 'price', file_object = DATABASE_LOG)
            folders = ['Training_Data_FromDB/','Intermediate_Train_Results/','Caching/','Saved_Models/']
            trainvalidator.initial_data_preparation(
                schemapath = 'schema_training.json', folders = folders, batchfilepath = "Training_Batch_Files/price.csv", compileddatapath= DATA_SOURCE)
            st.success(
                "This step of the pipeline has been completed successfully. Check the local files for more details.")
        if st.button("Exploratory Data Analysis"):
            if 'Training_Data_FromDB' not in os.listdir(os.getcwd()):
                st.error(
                    "Database has not yet inserted. Have u skipped Training Data Validation step?")
            else:
                preprocessor = train_Preprocessor(
                    file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
                preprocessor.eda()
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
        if st.button("Training Data Preprocessing"):
            if 'Training_Data_FromDB' not in os.listdir(os.getcwd()):
                st.error(
                    "Database has not yet inserted. Have u skipped Training Data Validation step?")
            else:
                preprocessor = train_Preprocessor(
                    file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
                preprocessor.data_preprocessing(
                    col_drop_path= 'Columns_Drop_from_Original.csv')
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
        model_names = st.multiselect(
            "Select the following model you would like to train for model selection", options=['HuberRegressor', 'Ridge', 'Lasso', 'ElasticNet',  'LinearSVR', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'HistGradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'])
        if st.button("Model Selection"):
            if not os.path.isdir(RESULT_DIR) or 'X.csv' not in os.listdir(RESULT_DIR):
                st.error(
                    "Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
            else:
                trainer = model_trainer(file_object= TRAINING_LOG)
                X = pd.read_csv(RESULT_DIR + 'X.csv')
                y = pd.read_csv(RESULT_DIR + 'y.csv')
                trainer.model_selection(
                    input = X, output = y, num_trials = 20, folderpath = RESULT_DIR, model_names = model_names)
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
        model_name = st.selectbox(
            "Select the following model you would like to train for final model deployment", options=['HuberRegressor', 'Ridge', 'Lasso', 'ElasticNet',  'LinearSVR', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'HistGradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'])
        if st.button("Final Model Training"):
            if not os.path.isdir(RESULT_DIR) or 'X.csv' not in os.listdir(RESULT_DIR):
                st.error(
                    "Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
            elif not os.path.isdir(RESULT_DIR + model_name):
                st.error(
                    "Model algorithm selection has not been done. Have u skipped model selection step?")
            else:
                trainer = model_trainer(file_object= TRAINING_LOG)
                X = pd.read_csv(RESULT_DIR + 'X.csv')
                y = pd.read_csv(RESULT_DIR + 'y.csv')
                trainer.final_model_tuning(
                    input_data = X, output_data = y, num_trials = 20, folderpath = RESULT_DIR, model_name = model_name)
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
    with st.expander("Model Prediction"):
        if not os.path.isdir(RESULT_DIR) or 'X.csv' not in os.listdir(RESULT_DIR):
            st.error(
                "Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
        else:
            X = pd.read_csv(RESULT_DIR + 'X.csv')
            NAME = st.selectbox('Select Name of Product',list(X['NAME'].unique()))
            ZONE = st.selectbox(
                'Select zone of product',('NORTH','SOUTH','EAST','WEST'))
            Brand = st.selectbox(
                'Select Brand of Product',list(X['Brand'].unique()))
            MC = st.selectbox(
                'Select Material Category of Product',list(X['MC'].unique()))
            NSU = st.number_input('Enter Net Sales Unit',min_value=0.00)
            Sales_at_Cost = st.number_input('Enter cost of sales',min_value=0)
            MRP = st.number_input(
                'Enter maximum possible retail price',min_value=0.01)
            Gross_Sales = MRP * NSU
            Fdate_year = st.selectbox(
                'Select year of transaction',[2017,2018,2019,2020,2021,2022])
            Fdate_month = st.selectbox(
                'Select month of transaction',[1,2,3,4,5,6,7,8,9,10,11,12])
            Fdate_quarter = 1 if Fdate_month in [1,2,3] else 2 if Fdate_month in [4,5,6] else 3 if Fdate_month in [7,8,9] else 4
            if st.button('Predict selling price'):
                if not os.path.isdir('Saved_Models/') or not os.listdir('Saved_Models/'):
                    st.error(
                        "No model has been saved yet. Have u skipped Final Model Training step?")
                else:
                    model = joblib.load('Saved_Models/FinalModel.pkl')
                    pipeline = joblib.load(
                        'Saved_Models/Preprocessing_Pipeline.pkl')
                    inputs = pd.DataFrame({'NAME':[NAME],'ZONE':[ZONE],'Brand':[Brand],'MC':[MC],'NSU':[NSU],'Sales at Cost':[Sales_at_Cost],'Gross Sales':[Gross_Sales],'MRP':[MRP],'Fdate_year':[Fdate_year],'Fdate_month':[Fdate_month],'Fdate_quarter':[Fdate_quarter]})
                    inputs_transformed = pipeline.transform(inputs)
                    predicted_value = model.predict(inputs_transformed)
                    st.write(
                        "Predicted selling price of given product is $",str(np.round(predicted_value[0],2)))

if __name__=='__main__':
    main()