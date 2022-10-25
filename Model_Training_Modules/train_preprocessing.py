'''
Author: Liaw Yi Xian
Last Modified: 25th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
import numpy as np
from feature_engine.datetime import DatetimeFeatures
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.figure_factory as ff
from tqdm import tqdm
from dython.nominal import associations

random_state=120

class train_Preprocessor:


    def __init__(self, file_object, datapath, result_dir):
        '''
            Method Name: __init__
            Description: This method initializes instance of train_Preprocessor class
            Output: None

            Parameters:
            - file_object: String path of logging text file
            - datapath: String path where compiled data is located
            - result_dir: String path for storing intermediate results from running this class
        '''
        self.file_object = file_object
        self.datapath = datapath
        self.result_dir = result_dir
        self.log_writer = App_Logger()


    def extract_compiled_data(self):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start reading compiled data from database")
        try:
            data = pd.read_csv(self.datapath)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to read compiled data from database with the following error: {e}")
            raise Exception(
                f"Fail to read compiled data from database with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish reading compiled data from database")
        return data


    def data_cleaning(self,data):
        '''
            Method Name: data_cleaning
            Description: This method performs initial data cleaning on a given pandas dataframe, while removing certain data anomalies.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        data = data[data['Sales at Cost']>=0]
        data[['NSU','MRP','SP']] = data[['NSU','MRP','SP']].applymap(
            lambda x: abs(x))
        data['Gross Sales'] = np.round(data['NSU'] * data['MRP'],2)
        data = data[data['SP']<=data['MRP']]
        feat_extract_Fdate = DatetimeFeatures(
            variables=['Fdate'],missing_values='ignore',features_to_extract=['year','month','quarter'])
        data = feat_extract_Fdate.fit_transform(data)
        data[['Fdate_year','Fdate_month','Fdate_quarter']] = data[['Fdate_year','Fdate_month','Fdate_quarter']].applymap(lambda x: str(x))
        return data


    def remove_irrelevant_columns(self, data, cols):
        '''
            Method Name: remove_irrelevant_columns
            Description: This method removes columns from a pandas dataframe, which are not relevant for analysis.
            Output: A pandas DataFrame after removing the specified columns. In addition, columns that are removed will be stored in a separate csv file.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - cols: List of irrelevant columns to remove from dataset
        '''
        self.log_writer.log(
            self.file_object, "Start removing irrelevant columns from the dataset")
        try:
            data = data.drop(cols, axis=1)
            result = pd.concat(
                [pd.Series(cols, name='Columns_Removed'), pd.Series(["Irrelevant column"]*len(cols), name='Reason')], axis=1)
            result.to_csv(self.result_dir+self.col_drop_path, index=False)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
            raise Exception(
                f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish removing irrelevant columns from the dataset")
        return data


    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, "Start handling duplicated rows in the dataset")
        if len(data[data.duplicated()]) == 0:
            self.log_writer.log(
                self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                data[data.duplicated()].to_csv(
                    self.result_dir+'Duplicated_Records_Removed.csv', index=False)
                data = data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"Fail to remove duplicated rows with the following error: {e}")
                raise Exception(
                    f"Fail to remove duplicated rows with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish handling duplicated rows in the dataset")
        return data
    

    def features_and_labels(self,data,target_col):
        '''
            Method Name: features_and_labels
            Description: This method splits a pandas dataframe into two pandas objects, consist of features and target labels.
            Output: Two pandas/series objects consist of features and labels separately.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - target_col: Name of target column
        '''
        self.log_writer.log(
            self.file_object, "Start separating the data into features and labels")
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to separate features and labels with the following error: {e}")
            raise Exception(
                f"Fail to separate features and labels with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish separating the data into features and labels")
        return X, y


    def eda(self):
        '''
            Method Name: eda
            Description: This method performs exploratory data analysis on the entire dataset, while generating various plots/csv files for reference.
            Output: None
        '''
        self.log_writer.log(
            self.file_object, 'Start performing exploratory data analysis')
        path = os.path.join(self.result_dir, 'EDA')
        if not os.path.exists(path):
            os.mkdir(path)
        scat_path = os.path.join(path, 'High_Correlation_Scatterplots')
        if not os.path.exists(scat_path):
            os.mkdir(scat_path)
        data = self.extract_compiled_data()
        data = self.data_cleaning(data)
        # Remove irrelevant columns from analysis
        data = data.drop(
            ['UID', 'NSV', 'GST Value', 'NSV-GST', '(NSV-GST)-SALES AT COST', 'Gross RGM(P-L)', 'Gross Margin %(Q/P*100)', 'MARGIN%', 'DIS', 'DIS%'], axis=1)
        # Extract basic information about dataset
        pd.DataFrame({"name": data.columns, "non-nulls": len(data)-data.isnull().sum().values, "type": data.dtypes.values}).to_csv(self.result_dir + "EDA/Data_Info.csv",index=False)
        # Extract summary statistics about dataset
        data.describe().T.to_csv(
            self.result_dir + "EDA/Data_Summary_Statistics.csv")
        # Extract information related to number of unique values for every feature of dataset
        nunique_values = []
        for col in data.columns:
            nunique_values.append(data[col].nunique())
        pd.DataFrame(
            [data.columns, nunique_values], index=['Variable','Number']).T.to_csv(self.result_dir + "EDA/Number_Unique_Values.csv", index=False)
        for col in tqdm(data.columns):
            col_path = os.path.join(path, col)
            if not os.path.exists(col_path):
                os.mkdir(col_path)
            if data[col].dtype == 'object' or 'Fdate' in col:
                data[col].value_counts().to_csv(
                    self.result_dir+f'EDA/{col}/{col}_nunique.csv')
                if data[col].nunique() < 50:
                    plt.figure(figsize=(24, 16),dpi=100)
                    countplot = sns.countplot(data = data, y = col)
                    for rect in countplot.patches:
                        width = rect.get_width()/len(data)*100
                        plt.text(
                            rect.get_width(), rect.get_y()+0.5*rect.get_height(), '%.2f' % width + '%', ha='left', va='center')
                    plt.title(f'{col} distribution')
                    plt.savefig(self.result_dir+f'EDA/{col}/{col}_nunique.png', bbox_inches='tight', pad_inches=0.2)
                    plt.clf()
                    boxplot = sns.boxplot(data = data, y = col, x='SP')
                    plt.title(f'{col} distribution by Selling Price')
                    plt.savefig(
                        self.result_dir+f'EDA/{col}/{col}_boxplot_by_Selling_Price.png', bbox_inches='tight', pad_inches=0.2)
                    plt.clf()
            else:
                # Plotting boxplot of features
                fig2 = px.box(data,x=col,title=f"{col} Boxplot")
                fig2.write_image(
                    self.result_dir + f"EDA/{col}/{col}_Boxplot.png")
                # Plotting kdeplot of features
                fig3 = ff.create_distplot(
                    [data[col]], [col], show_hist=False,show_rug=False)
                fig3.layout.update(
                    title=f'{col} Density curve (Skewness: {np.round(data[col].skew(),4)})')
                fig3.write_image(
                    self.result_dir + f"EDA/{col}/{col}_Distribution.png")
                # Plotting scatterplot between given feature and target variable SP
                fig4 = px.scatter(
                    data,x=col,y='SP',title=f"Scatterplot of {col} vs SP")
                fig4.write_image(
                    self.result_dir + f"EDA/{col}/Scatterplot_{col}_vs_SP.png")
        # Plotting scatterplot between features that are highly correlated (>0.8) with each other based on absolute value of spearman correlation
        corr_matrix = data.corr(method='spearman')
        c1 = corr_matrix.stack().sort_values(ascending=False).drop_duplicates()
        high_cor = c1[c1.values!=1]
        results = high_cor[(high_cor>0.8) | (high_cor<-0.8)].reset_index()
        for col1, col2 in tqdm(zip(results['level_0'],results['level_1'])):
            fig5 = px.scatter(
                data,x=col1,y=col2,title=f"Scatterplot of {col1} vs {col2} (Spearman corr.: {np.round(corr_matrix.loc[col1, col2],4)})")
            fig5.write_image(
                self.result_dir + f"EDA/High_Correlation_Scatterplots/Scatterplot_{col1}_vs_{col2}.png")
        # Plot correlation heatmap
        r = associations(data, num_num_assoc='spearman', compute_only=True)
        correlation = r['corr']
        plt.figure(figsize=(12,8))
        sns.heatmap(
            correlation, annot=True, mask = np.triu(np.ones_like(correlation, dtype=bool)),fmt='.2f',annot_kws={'fontsize':12})
        plt.title('Spearman correlation heatmap',size=16)
        plt.savefig(
            self.result_dir+f'EDA/Correlation_Heatmap.png', bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        self.log_writer.log(
            self.file_object, 'Finish performing exploratory data analysis')


    def data_preprocessing(self, col_drop_path):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: None
            
            Parameters:
            - col_drop_path: String path that stores list of columns that are removed from the data
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        self.col_drop_path = col_drop_path
        data = self.extract_compiled_data()
        data = self.data_cleaning(data)
        cols_to_remove = ['UID', 'NSV', 'GST Value', 'NSV-GST', '(NSV-GST)-SALES AT COST', 'Gross RGM(P-L)', 'Gross Margin %(Q/P*100)', 'MARGIN%', 'DIS', 'DIS%']
        data = self.remove_irrelevant_columns(
            data = data, cols = cols_to_remove)
        data = self.remove_duplicated_rows(data = data)
        X, y = self.features_and_labels(data = data, target_col='SP')
        X.to_csv(self.result_dir+'X.csv',index=False)
        y.to_csv(self.result_dir+'y.csv',index=False)
        self.log_writer.log(self.file_object, 'End of data preprocessing')
