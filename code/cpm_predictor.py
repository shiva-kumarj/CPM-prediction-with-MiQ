# Import required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import datetime as dt
import seaborn as sns
import pickle
import tkinter as tk
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

def read_dataset(train_path, test_path):
    # Read the train dataset as well. no need to take as user input.
    train = pd.read_csv(train_path)
    # Read the path of the test dataset
    test = pd.read_csv(test_path)

    # convert dtype of date
    # Infer the date from the input.
    test['Date'] = pd.to_datetime(test['Date'], format = '%d-%m-%Y')
    target_date = str(test['Date'].iloc[0].date())

    train['Date'] = pd.to_datetime(train['Date'], format = '%d-%m-%Y')
    validation_date = str(pd.Series(train['Date'].sort_values().unique()).iloc[-1].date())
    
    # print some information such as shape of input dataframe, 
    print("Shape of train data: {}".format(train.shape))
    print("Shape of test data: {}".format(test.shape))
    
    # and the inferred target date 
    print("Target date: {}".format(target_date))
    return train, test, target_date, validation_date


def clean_unknowns(train, test):
    # test set
    # impute  rows with unknown app_id .
    test.loc[test[test['App/URL ID'] == 'Unknown'].index, 'App/URL ID'] = 9999

    # train set
    train.loc[train[train['App/URL ID'] == 'Unknown'].index, 'App/URL ID'] = 9999    

   
def prep_dataset(train, test):
    # Concatenate train and test into one dataframe for encoding. reset_index(drop = True)
    final_df = pd.concat([train, test]).reset_index(drop = True)
         
    # drop redundant columns ['IO_ID', 'Line Item ID', 'row_no'] inplace,
    final_df.drop(['IO_ID', 'Line Item ID', 'row_no'], axis = 1, inplace = True)
    
    # rename columns 
    final_df.columns = ['date', 'app_url_id', 'isp_or_carrier_id', \
        'device_type', 'exchange_id','operating_system', 'browser',\
            'creative_size', 'advertiser_currency', 'impressions', 'cpm']
    
    # make day_of_week column.
    final_df['day_of_week'] = final_df['date'].dt.day_name()
    
    # fillna in 'cpm'
    final_df['cpm'].fillna(0, inplace = True)

    # convert app_id to numeric
    final_df['app_url_id'] = pd.to_numeric(final_df['app_url_id'])

    # return final_df
    return final_df

def ordinal_encoding(df, categorical):
    
    enc = OrdinalEncoder()
    # encode the categorical columns of the final_df
    df[categorical] = enc.fit_transform(df[categorical])

    # return the transformed df
    return df

def train_test_split(target_date, df, categorical):
    print("Target Date: {}".format(target_date))

    # Take test date to make test dataset.
    df_test = df[df['date'] == target_date].copy()
    test_indices = list(df_test.index)

    # Take train date make into train.
    df_train = df.drop(test_indices)
    y_train = df_train['cpm']
    df_train.drop(['date', 'cpm'], inplace = True, axis = 1)

    #y_val = df_val['cpm']
    #df_val.drop(['date', 'cpm'], inplace = True, axis = 1)

    df_test.drop(['date', 'cpm'], axis = 1, inplace = True)

    df_train[categorical] = df_train[categorical].apply(pd.to_numeric, downcast = 'integer')
    #df_val[categorical] = df_val[categorical].apply(pd.to_numeric, downcast = 'integer')
    df_test[categorical] = df_test[categorical].apply(pd.to_numeric, downcast = 'integer')

    # return test_df, train_df, validation_df, y_train, y_val
    #return df_train, y_train, df_val, y_val, df_test
    return df_train, y_train, df_test

def model_prediction(df, model_path):
    # load the model (catboostmodel.pkl)
    cbr = pickle.load(open(model_path, 'rb'))

    # print the MAE for train and test
    # print R2 for  train and test.

    # predict the cpm using model
    predicted_cpm = pd.Series(cbr.predict(df))
    # merge the output series with the original test dataset.
    test['predicted_cpm'] = predicted_cpm

    return test

if __name__ == '__main__':
    # take user input to the path of the test dataset.
    # Select the train and test data from their respective path
    print("Select test.csv file")
    test_path = filedialog.askopenfilename()
    print("Select train.csv file")
    train_path= filedialog.askopenfilename()
    print("Select model to use")
    model_path = filedialog.askopenfilename()
    print("Select the output file path")
    output_file = filedialog.asksaveasfilename()

    train, test, target_date, validation_date = read_dataset(train_path, test_path)
    
    # clean the "unknowns" present in train and test set
    clean_unknowns(train, test)
    print("Shape of train set after cleansing: {}".format(train.shape))
    print("Shape of test set after cleansing: {}".format(test.shape))


    # Prep dataset for encoding,
    # rename columns, fillna  etc
    final_df = prep_dataset(train, test)

    # Categorical Columns 
    categorical = ['app_url_id', 'isp_or_carrier_id', 'device_type',\
        'exchange_id','operating_system', 'browser', \
            'creative_size', 'advertiser_currency','day_of_week']

    # ordinal encoding the data
    final_df = ordinal_encoding(final_df, categorical)
    
    # dataset splitting
    df_train, y_train, df_test = train_test_split(target_date, final_df, categorical)

    result = model_prediction(df_test, model_path)
    
    # write the output dataset to csv and return path to output.
    result.to_csv(output_file, index = False)
    print("Predictions file at output path: {}".format(output_file))






