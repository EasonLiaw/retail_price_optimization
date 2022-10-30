'''
Author: Liaw Yi Xian
Last Modified: 30th October 2022
'''

import json
import pandas as pd
import psycopg2
from Application_Logger.logger import App_Logger
from Application_Logger.exception import CustomException
import os, sys
import shutil
import DBConnectionSetup as login


class rawtraindatavalidation:


    def __init__(self, table_name, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of rawtraindatavalidation class
            Output: None

            Parameters:
            - table_name: String name of table within a given PostgreSQL database
            - file_object: String path of logging text file
        '''
        self.tablename = table_name
        self.file_object = file_object
        self.host = login.logins['host']
        self.user = login.logins['user']
        self.password = login.logins['password']
        self.dbname = login.logins['dbname']
        self.log_writer = App_Logger()


    def newDB(self, schema):
        '''
            Method Name: newDB
            Description: This method creates a new database and table in PostgreSQL database based on a given schema object.
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - schema: JSON object file related to schema database
        '''
        self.log_writer.log(
            self.file_object, f"Start creating new table({self.tablename}) in SQL database ({self.dbname})")
        try:
            # Comment out line 51-60 for Heroku deployment
            conn = psycopg2.connect(
                host=self.host,user=self.user,password=self.password)
            conn.autocommit = True
            mycursor = conn.cursor()
            mycursor.execute(
                f"SELECT COUNT(*) FROM pg_database WHERE datname = '{self.dbname}'")
            if mycursor.fetchone()[0] == 0:
                mycursor.execute(f"CREATE DATABASE {self.dbname}")
            conn.commit()
            conn.close()
            conn = psycopg2.connect(
                host=self.host,user=self.user,password=self.password,database=self.dbname)
            mycursor = conn.cursor()
            for name, type in zip(schema['ColName'].keys(),schema['ColName'].values()):
                mycursor.execute(
                    f"""SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.tablename}'""")
                if mycursor.fetchone()[0] == 1:
                    try:
                        mycursor.execute(
                            f"ALTER TABLE {self.tablename} ADD \"{name}\" {type}")
                        conn.commit()
                        self.log_writer.log(
                            self.file_object, f"Column {name} added into {self.tablename} table")
                    except:
                        conn.rollback()
                        self.log_writer.log(
                            self.file_object, f"Column {name} already exists in {self.tablename} table")
                else:
                    mycursor.execute(
                        f"CREATE TABLE {self.tablename} (\"{name}\" {type})")
                    conn.commit()
                    self.log_writer.log(
                        self.file_object, f"{self.tablename} table created with column {name}")
        except ConnectionError:
            self.log_writer.log(
                self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        conn.close()
        self.log_writer.log(
            self.file_object, f"Finish creating new table({self.tablename}) in SQL database ({self.dbname})")


    def data_insert(self):
        '''
            Method Name: data_insert
            Description: This method inserts data from existing csv file into PostgreSQL database
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start inserting new good training data into SQL database")
        try:
            data = self.characters_single_quotes()
            data = self.blank_with_null_replacement(data)
            conn = psycopg2.connect(
                host=self.host,user=self.user,password=self.password,database=self.dbname)
            mycursor = conn.cursor()
            for index in range(len(data)):
                try:
                    newstring = str(data.iloc[index].tolist()).replace("\"","").replace("'null'","null")[1:-1]
                    mycursor.execute(
                        f"INSERT INTO {self.tablename} VALUES ({newstring})")
                    conn.commit()
                except Exception as e:
                    self.log_writer.log(
                        self.file_object, f'Row {index} could not be inserted into database for price.csv file')
                    conn.rollback()
                    raise Exception(
                        f"The following error occured when connecting to SQL database: {e}")
            self.log_writer.log(
                self.file_object, f"Price.csv file added into database")
        except ConnectionError:
            self.log_writer.log(
                self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        conn.close()
        self.log_writer.log(
            self.file_object, "Finish inserting new good training data into SQL database")


    def compile_data_from_DB(self):
        '''
            Method Name: compile_data_from_DB
            Description: This method compiles data from PostgreSQL database into csv file for further data preprocessing.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start writing compiled good training data into a new CSV file")
        try:
            conn = psycopg2.connect(
                host=self.host,user=self.user,password=self.password,database=self.dbname)
            data = pd.read_sql(
                f'''SELECT DISTINCT * FROM {self.tablename};''', conn)
            data.to_csv(self.compileddatapath, index=False)
        except ConnectionError:
            self.log_writer.log(
                self.file_object, "Error connecting to PostgreSQL database")
            raise Exception("Error connecting to PostgreSQL database")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        conn.close()
        self.log_writer.log(
            self.file_object, "Finish writing compiled good training data into a new CSV file")


    def file_initialize(self):
        '''
            Method Name: file_initialize
            Description: This method creates the list of folders mentioned in the filelist if not exist. If exist, this method deletes the existing folders and creates new ones. Note that manual archiving will be required if backup of existing files is required.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start initializing folder structure")
        for folder in self.folders:
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(os.path.dirname(folder), exist_ok=True)
                self.log_writer.log(
                    self.file_object, f"Folder {folder} has been initialized")
            except Exception as e:
                self.log_writer.log(
                    self.file_object, str(CustomException(e,sys)))
                raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish initializing folder structure")


    def load_train_schema(self):
        '''
            Method Name: load_train_schema
            Description: This method loads the schema of the training data from a given JSON file for creating tables in PostgreSQL database.
            Output: JSON object
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start loading train schema")
        try:
            with open(self.schemapath, 'r') as f:
                schema = json.load(f)
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(self.file_object, "Finish loading train schema")
        return schema
    

    def characters_single_quotes(self):
        '''
            Method Name: characters_single_quotes
            Description: This method adds single quotes to all string related data types in a given CSV file. Not adding single quotes to string data types will result in error when inserting data into PostgreSQL table.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start handling single quotes on characters")
        try:
            data = pd.read_csv(self.batchfilepath)
            char_df = data.select_dtypes('object')
            for column in char_df.columns:
                data[column] = data[column].map(lambda x: x.replace("'","''"))
                data[column] = data[column].map(lambda x: f'\'{x}\'')
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish handling single quotes on characters")
        return data
    

    def blank_with_null_replacement(self, data):
        '''
            Method Name: blank_with_null_replacement
            Description: This method replaces blanks with null values in a given CSV file.
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - data: Pandas dataframe
        '''
        self.log_writer.log(
            self.file_object, "Start replacing missing values with null keyword")
        try:
            data = data.fillna('null')
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish replacing missing values with null keyword")
        return data
    
    
    def initial_data_preparation(
            self, folders, schemapath, batchfilepath, compileddatapath):
        '''
            Method Name: initial_data_preparation
            Description: This method performs all the preparation tasks for the data to be ingested into PostgreSQL database.
            Output: None

            Parameters:
            - folders: List of string file paths for initializing folder structure
            - schemapath: String path where JSON object file related to schema database is located
            - batchfilepath: String file path for specified folder
            - compileddatapath: String path where good quality data is compiled from database
        '''
        self.log_writer.log(self.file_object, "Start initial data preparation")
        self.folders = folders
        self.schemapath = schemapath
        self.batchfilepath = batchfilepath
        self.compileddatapath = compileddatapath
        self.file_initialize()
        schema = self.load_train_schema()
        self.newDB(schema)
        self.data_insert()
        self.compile_data_from_DB()
        self.log_writer.log(self.file_object, "Finish initial data preparation")