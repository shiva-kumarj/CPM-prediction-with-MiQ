import pandas as pd
import numpy as np
import ast
import datetime as dt
import pickle
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error


def read_data(test_path, train_path):
    # read train and test data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # print shape of train and test data
    # change dtype of date in train
    test['Date'] = pd.to_datetime(test['Date'], format = '%d-%m-%Y')
    target_date = str(test['Date'].iloc[0].date())

    train['Date'] = pd.to_datetime(train['Date'], format = '%d-%m-%Y')
    validation_date = str(pd.Series(train['Date'].sort_values().unique()).iloc[-1].date())
    # return train and test dataset.
    return train, test, target_date, validation_date

def clean_unknowns(train, test):
    # remove rows with unknown app_id .
    unknown_app_ids = train[train['App/URL ID'] == 'Unknown'].index
    train.drop(unknown_app_ids, axis = 0, inplace = True)

    unknown_app_ids = test[test['App/URL ID'] == 'Unknown'].index
    test.drop(unknown_app_ids, axis = 0, inplace = True)    

    # perform transformation in place.
    train.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    # No need to return dataframe
    print("Shape of train set after cleansing: {}".format(train.shape))
    print("Shape of test set after cleansing: {}".format(test.shape))


def prep_dataset(train, test):
    # Concatenate train and test into one dataframe for encoding. 
    final_df = pd.concat([train, test]).reset_index(drop = True)         
    
    # drop redundant columns ['IO_ID', 'Line Item ID', 'row_no'] inplace,
    final_df.drop(['IO_ID', 'Line Item ID', 'row_no'], axis = 1, inplace = True)
    
    # rename columns 
    final_df.columns = ['date', 'app_url_id', 'isp_or_carrier_id',
                   'device_type', 'exchange_id', 'operating_system', 'browser',
                   'creative_size', 'advertiser_currency', 'impressions', 'cpm']
    # Change app_id to numeric type
    final_df['app_url_id'] = pd.to_numeric(final_df['app_url_id'])

    # make day_of_week column.
    final_df['day_of_week'] = final_df['date'].dt.day_name()

    # fillna in 'cpm'
    final_df['cpm'].fillna(0, inplace = True)
    
    # return final_df
    return final_df

def ordinal_encoder(categorical, df):
    # Perform label encoding
    enc = OrdinalEncoder()
    # encode the categorical columns of the final_df
    df[categorical] = enc.fit_transform(df[categorical])
    # return the transformed df
    return df

def train_test_split(target_date, df, validation_date):
    print("Target Date: {}".format(target_date))

    # Take test date to make test dataset.
    df_test = df[df['date'] == target_date].copy()
    test_indices = list(df_test.index)

    # Take validation date into validation dataset.
    df_val = df[df['date'] == validation_date].copy()
    val_indices = list(df_val.index)
    test_indices.extend(val_indices)

    # Take train date make into train.
    df_train = df.drop(test_indices)
    y_train = df_train['cpm']
    df_train.drop(['date', 'cpm'], inplace = True, axis = 1)

    y_val = df_val['cpm']
    df_val.drop(['date', 'cpm'], inplace = True, axis = 1)

    df_test.drop(['date', 'cpm'], axis = 1, inplace = True)

    df_train[categorical] = df_train[categorical].apply(pd.to_numeric, downcast = 'integer')
    df_val[categorical] = df_val[categorical].apply(pd.to_numeric, downcast = 'integer')
    df_test[categorical] = df_test[categorical].apply(pd.to_numeric, downcast = 'integer')

    # return test_df, train_df, validation_df, y_train, y_val
    return df_train, y_train, df_val, y_val, df_test

def param_eval(df_train, y_train, categorical):
    # Read the hyperparameter csv
    hyp_tuning_results = pd.read_csv(r'code\Hyperparameter Tuning\out_file_2', \
        names = ['loss', 'params', 'iteration', 'estimators', 'time'])

    # get best hyperparameter from csv.
    hyp_tuning_results.sort_values('loss', ascending = True, inplace = True)

    # Change task_type to "CPU"
    best_random_params = ast.literal_eval(hyp_tuning_results['params'].iloc[0]).copy()
    best_random_params['task_type'] = 'CPU'

    # set best params for regressor
    cbr = CatBoostRegressor(logging_level='Silent', random_state=100, 
                            early_stopping_rounds =10, **best_random_params)

    # fit model to df_train, ,y_train
    cbr.fit(df_train, y_train, cat_features = categorical, logging_level = 'Verbose')

    # Return model
    return cbr

if __name__ == '__main__':
    # Take user input for test and train data path
    root = tk.Tk()
    root.withdraw()
    # Select the train and test data from their respective path
    print("Select test.csv file")
    test_path = filedialog.askopenfilename()
    print("Select train.csv file")
    train_path= filedialog.askopenfilename()

    # Read dataframe
    train, test, target_date, validation_date = read_data(test_path, train_path)

    # clear unknown from test, train data
    clean_unknowns(train, test)

    # prepare final dataframe, concat train and test 
    final_df = prep_dataset(train, test)

    # ordinal encoding
    categorical = ['app_url_id', 'isp_or_carrier_id', 'device_type',\
                'exchange_id','operating_system', 'browser', \
                    'creative_size', 'advertiser_currency','day_of_week']
    
    print("Shape of final_df: ", final_df.shape)
    
    final_df = ordinal_encoder(categorical, final_df)

    # split data to train, validation and test set
    df_train, y_train, df_val, y_val, df_test = train_test_split(target_date, final_df, validation_date)

    print("Shape of train set: {}, y_train: {}".format(df_train.shape, y_train.shape))
    print("Shape of test set: {}".format(df_test.shape))
    print("Shape of validation set: {}, y_val: {}".format(df_val.shape, y_val.shape))
    
    # Get the best parameter and train model with it
    cbr_model = param_eval(df_train, y_train, categorical)

    print("Model: ", cbr_model.get_param)

    # Write model to disc
    pickle.dump(cbr_model, open(r'code\output\catboostmodel_2.pkl', 'wb'))

    print("Trained model at : {}".format(r'code\output\catboostmodel_2.pkl'))

    input("press any key to exit")

    



    



