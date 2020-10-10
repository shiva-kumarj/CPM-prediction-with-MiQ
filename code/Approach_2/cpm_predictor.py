import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cgb
import datetime as dt
import pickle
import random
import tkinter as tk
from tkinter import filedialog
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error

root = tk.Tk()
root.withdraw()

def read_dataset(train_path, test_path):
    # take user input for path to test and train data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # change type of date 
    test['Date'] = pd.to_datetime(test['Date'], format = '%d-%m-%Y')
    target_date = str(test['Date'].iloc[0].date())

    train['Date'] = pd.to_datetime(train['Date'], format = '%d-%m-%Y')
    
    # and the inferred target date 
    print("Target date: {}".format(target_date))
    return train, test, target_date    

def clean_unknown(train, test):
    # change the "unknown" to 99999
    # test set
    test.loc[test[test['App/URL ID'] == 'Unknown'].index, 'App/URL ID'] = 9999
    # train set
    train.loc[train[train['App/URL ID'] == 'Unknown'].index, 'App/URL ID'] = 9999    
    # transform in place, do not return

def prep_dataset(train, test, ):
    # Concat the train and test into one for encoding purposes
    final_df = pd.concat([train, test]).reset_index(drop = True)
         
    # drop redundant columns
    final_df.drop(['Line Item ID'], axis = 1, inplace = True)
    
    # rename columns 
    final_df.columns = ['date', 'app_url_id', 'isp_or_carrier_id',\
        'device_type', 'exchange_id', 'operating_system', 'browser',\
            'creative_size', 'advertiser_currency', 'impressions',\
                'io_id', 'cpm', 'row_no']
    
    # fillna in 'cpm'
    final_df['cpm'].fillna(0, inplace = True)

    # convert app_id to numeric
    final_df['app_url_id'] = pd.to_numeric(final_df['app_url_id'])

    # return final_df
    return final_df
    
def ordinal_endoder(df, categorical):
    # ordinal encoding of the colummns
    enc = OrdinalEncoder()
    df[categorical] = enc.fit_transform(final_df[categorical])
    # return transformed df
    return df

def train_test_split(df, target_date, ):
    # split dataset by io_id 
    dfx = {io: rows for io, rows in final_df.groupby('io_id')}

    # take target_date as train set
    dataset_structure = {}
    for io in dfx:
        temp_df = dfx[io]
        df_test = temp_df[temp_df['date'] == target_date].copy()
        test_indices = list(df_test.index)
        df_train = temp_df.drop(test_indices)
        
        y_train = df_train['cpm']
        df_train.drop(['date', 'cpm', 'io_id', 'row_no'], axis = 1, inplace = True)

        df_test.drop(['cpm', 'date', 'io_id'], axis = 1, inplace = True)

        # Encoded values must all be integral not float point
        df_train[:] = df_train[:].apply(pd.to_numeric, downcast = 'integer')
        df_test[:] = df_test[:].apply(pd.to_numeric, downcast = 'integer')
        dataset_structure[io] = (df_train, df_test, y_train)
        
    # return the dataset
    return dataset_structure

def model_prediction(trained_models, test, dataset_structure):
    # use model to generate predictions
    test_set_prediction = {}
    for io in dataset_structure:
        y_pred_test = pd.Series(trained_models[io].predict(dataset_structure[io][1]))
        # take the predictions in a dict
        test_set_prediction[io] = y_pred_test
    
    # merge predictions with the preprocessed test data
    for io in dataset_structure:
        dataset_structure[io][1].reset_index(drop = True, inplace=True)
        dataset_structure[io][1]['predicted_cpm'] = test_set_prediction[io]
    
    test_prediction = pd.concat([
        dataset_structure[1][1],\
        dataset_structure[2][1],
        dataset_structure[3][1],
        dataset_structure[4][1],
        dataset_structure[5][1],
        dataset_structure[6][1]]).reset_index(drop = True)

    test_prediction = pd.merge(test, \
        test_prediction[['row_no', 'predicted_cpm']], \
        on = 'row_no', how = 'left')
    
    # merge intermediate test predictions data with original test dataframe
    test_prediction.loc[test_prediction[test_prediction['predicted_cpm'] < 0].index, 'predicted_cpm'] = 0.010000
    
    # return transformed test dataframe.
    return test_prediction

if __name__ == '__main__':
    # take user input for test and train data path
    print("Select test.csv file")
    test_path = filedialog.askopenfilename()
    
    print("Select train.csv file")
    train_path= filedialog.askopenfilename()
    
    print("Select the directory where trained models are saved")
    model_path = filedialog.askdirectory()
    
    #print("Select the output file path")
    #output_file = filedialog.asksaveasfilename()

    train, test, target_date = read_dataset(train_path, test_path)

    clean_unknown(train, test)
    print("Shape of train set after cleansing: {}".format(train.shape))
    print("Shape of test set after cleansing: {}".format(test.shape))

    # Prep dataset for encoding,
    # rename columns, fillna  etc
    final_df = prep_dataset(train, test)

    categorical = [
        'app_url_id', 'isp_or_carrier_id', 'device_type', 
        'exchange_id','operating_system', 'browser', 
        'creative_size', 'advertiser_currency']

    # ordinal encoding
    final_df = ordinal_endoder(final_df, categorical)

    dataset_structure = train_test_split(final_df, target_date)

    i = 1
    trained_models = {}
    for io in dataset_structure:
        trained_models[io] = \
        pickle.load(open(model_path + "/trained_cbr_models_%s.pkl"%i, 'rb'))
        i+=1

    result = model_prediction(trained_models, test, dataset_structure)

    # Write resultant dataframe as csv
    print("Select the output file path")
    output_file = filedialog.asksaveasfilename()
    result.to_csv(output_file)

    print("The prediction CSV is here: {}".format(output_file))




