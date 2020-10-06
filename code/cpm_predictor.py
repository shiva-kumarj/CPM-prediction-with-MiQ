# Import required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import datetime as dt
import seaborn as sns
import pickle
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error

def read_dataset():
    # Read the train dataset as well. no need to take as user input.
    train = pd.read_csv(r"C:\Users\blahb\OneDrive\Documents\ML and Cloud UPGRAD\CAPSTONE PROJECT\CPM-prediction-with-MiQ\data\train.csv")
    
    # Read the path of the test dataset
    test = pd.read_csv(r"C:\Users\blahb\OneDrive\Documents\ML and Cloud UPGRAD\CAPSTONE PROJECT\CPM-prediction-with-MiQ\data\test.csv")

    # convert dtype of date
    # Infer the date from the input.
    test['Date'] = pd.to_datetime(test['Date'], format = '%d-%m-%Y')
    target_date = str(test['Date'].iloc[0].date())

    train['Date'] = pd.to_datetime(train['Date'], format = '%d-%m-%Y')
    
    # print some information such as shape of input dataframe, 
    print("Shape of train data: {}".format(train.shape))
    print("Shape of test data: {}".format(test.shape))
    
    # and the inferred target date 
    print("Target date: {}".format(target_date))
    return train, test, target_date


def clean_unknowns(train, test):
    # take both and train and test dataset as input parameter
    # app_ids contain "Unknown" in both. Remove these entries

    # test set
    unknown_app_ids = test[test['App/URL ID'] == 'Unknown'].index
    test.drop(unknown_app_ids, inplace = True)

    # train set
    unknown_app_ids = train[train['App/URL ID'] == 'Unknown'].index
    train.drop(unknown_app_ids, inplace = True)

    # Perform inplace transformation, no need to return dataframe
    # print the shape of train and test after removing "Unknown".
    print("Shape of train set after cleansing: {}".format(train.shape))
    print("Shape of test set after cleansing: {}".format(test.shape))

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

def ordinal_encoding(df):
    categorical = ['app_url_id', 'isp_or_carrier_id', 'device_type',\
        'exchange_id','operating_system', 'browser', \
            'creative_size', 'advertiser_currency','day_of_week']

    enc = OrdinalEncoder()
    # encode the categorical columns of the final_df
    df[categorical] = enc.fit_transform(df[categorical])

    # return the transformed df
    return df


def train_test_split(df, target_date):
    # create df_test from the target_date
    print(target_date)
    df_test = df[df['date'] == target_date].copy()
  
    # create validation dataset frm train data.
    df_val = df[df['date'] == '2020-08-27'].copy()
    
    # drop "date" and "cpm".
    df_test.drop(['cpm', 'date'], inplace = True, axis = 1)
    df_val.drop(['date', 'cpm'], axis = 1, inplace = True)

    print("Shape of validation set: {}".format(df_val.shape))
    print("Shape of test set: {}".format(df_test.shape))
    
    # return the df_test, df_validation.
    return df_test, df_val

def model_prediction(df):
    # load the model (catboostmodel.pkl)
    cbr = pickle.load(open(r'C:\Users\blahb\OneDrive\Documents\ML and Cloud UPGRAD\CAPSTONE PROJECT\CPM-prediction-with-MiQ\code\output\catboostmodel.pkl', 'rb'))

    # print the MAE for train and test
    # print R2 for  train and test.

    # predict the cpm using model
    predicted_cpm = pd.Series(cbr.predict(df))
    # merge the output series with the original test dataset.
    test['predicted_cpm'] = predicted_cpm

    # write the output dataset to csv and return path to output.
    test.to_csv(r"../data/prediction.csv")
    print("Predictions file at output path")

if __name__ == '__main__':
    # take user input to the path of the test dataset.
    #test_path = input()
    train, test, date = read_dataset()
    
    # clean the "unknowns" present in train and test set
    clean_unknowns(train, test)

    # Prep dataset for encoding,
    # rename columns, fillna  etc
    final_df = prep_dataset(train, test)

    # ordinal encoding the data
    final_df = ordinal_encoding(final_df)

    # dataset splitting
    df_test, df_val = train_test_split(final_df, date)

    # model prediction from saved location
    print("Shape of df_test: ",df_test.head())

    #model_prediction(df_test)





