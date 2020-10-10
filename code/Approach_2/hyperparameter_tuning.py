import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import pickle
import random
import tkinter as tk
from tkinter import filedialog
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


def read_data(train_path, test_path):
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

def train_test_split(df, target_date, validation_date):
    dataset_structure = {}
    dfx = {io: rows for io, rows in final_df.groupby('io_id')}
    for io in dfx:
        temp_df = dfx[io]
        df_val = temp_df[temp_df['date'] == validation_date].copy()
        df_test = temp_df[temp_df['date'] == target_date].copy()
        
        val_indices = list(df_val.index)
        test_indices = list(df_test.index)
        test_indices.extend(val_indices)
        df_train = temp_df.drop(test_indices)
        
        y_train = df_train[['cpm']]
        df_train.drop(['date', 'cpm', 'io_id'], axis = 1, inplace = True)

        y_val = df_val[['cpm']]
        df_val.drop(['cpm', 'date', 'io_id'], axis = 1, inplace = True)

        df_test.drop(['cpm', 'date', 'io_id'], axis = 1, inplace = True)

        # Encoded values must all be integral not float point
        df_train[categorical] = df_train[categorical].apply(pd.to_numeric, downcast = 'integer')
        df_val[categorical] = df_val[categorical].apply(pd.to_numeric, downcast = 'integer')
        df_test[categorical] = df_test[categorical].apply(pd.to_numeric, downcast = 'integer')
        dataset_structure[io] = (df_train, df_val, df_test, y_train, y_val)

    return dataset_structure

def hyper_tuner(dataset_structure):
    grid_cbr_io_models = {}

    parameters = {
        'depth': [6,8,10],
        'learning_rate' : np.round(np.linspace(0.05, 0.1, 5), 2).tolist(),
        'l2_leaf_reg':[10, 50, 10]
        }
    
    for io in dataset_structure:
        print("##### IO_ID {} #####".format(io))
        cv_model = GridSearchCV(CatBoostRegressor(
            ctr_target_border_count=1,iterations = 200,
            logging_level="Silent", random_state=100,
            loss_function='MAE', eval_metric='MAE', 
            bootstrap_type='Bayesian', bagging_temperature=5,
            thread_count = -1)
            ,parameters, n_jobs = -1,cv = 5, verbose = 3)
        cv_model.fit(dataset_structure[io][0], dataset_structure[io][3]['cpm'])
        grid_cbr_io_models[io] = cv_model
    return grid_cbr_io_models

if __name__ == '__main__':
    print("Select test.csv file")
    test_path = filedialog.askopenfilename()
    
    print("Select train.csv file")
    train_path= filedialog.askopenfilename()

    train, test, target_date, validation_date = read_data(train_path, test_path)

    # clean unknowns
    clean_unknown(train, test)
    print("Shape of train set after cleansing: {}".format(train.shape))
    print("Shape of test set after cleansing: {}".format(test.shape))

    # prepare final dataframe, concat train and test 
    final_df = prep_dataset(train, test)

    categorical = [
        'app_url_id', 'isp_or_carrier_id', 'device_type', 
        'exchange_id','operating_system', 'browser', 
        'creative_size', 'advertiser_currency']
    
    # ordinal encoding
    final_df = ordinal_endoder(final_df, categorical)

    dataset_structure = train_test_split(final_df, target_date, validation_date)

    grid_search_models = hyper_tuner(dataset_structure)

    # dump grid search models to disk
    print("Path where you want to store the trained models")
    trained_model_path = filedialog.askdirectory()
    i = 1
    for io in grid_search_models:
        pickle.dump(grid_search_models[io], open(trained_model_path+"/grid_cbr_io_models_%s.pkl" % i, 'wb'))
        i+=1
    
    print("Done.....")
    print("Trained Models are at this path")
    for io in grid_search_models:
        print(trained_model_path+"/grid_cbr_io_models_%s.pkl"%io, 'wb')
    
    input("Press any key to exit")
