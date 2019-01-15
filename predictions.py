# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:39:20 2019

@author: janpa
"""
#import functions
import pandas as pd
import numpy as np
import pylab
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import seaborn
import math
import scipy
pd.options.mode.chained_assignment = None  # default='warn'

from datetime import datetime
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.append('C:/Users/janpa/Desktop/Uni/Bachelorarbeit/gathered_stats')
sys.path.append('..')
#import Data
BTC_USD_data = pd.read_hdf('C:/Users/janpa/Desktop/Uni/Bachelorarbeit/gathered_stats/BTC_USD_all_dbs_BitFinex_corrected_vola.h5')

#choose features, target and lag
Features = ['VWAP_10_n_ask','VWAP_10_n_bid', 'VWAP_20_n_ask', 'VWAP_20_n_bid', \
            'Vola_last_10',
            'av_bid_volume_price_ratio', 'av_ask_volume_price_ratio', \
            'five_percent_mid_dev_volume_a', \
            'number_asks', 'number_bids', \
            'spread', \
            'ten_last_added_volume_a','ten_last_added_volume_b', 'ten_percent_mid_dev_volume_a', \
       'ten_percent_mid_dev_volume_b', 'ten_percent_volume_a', \
       'ten_percent_volume_b', 'top_five_volume_a', 'top_five_volume_b', \
       'top_ten_volume_a', 'top_ten_volume_b', 'top_twenty_volume_a', \
       'top_twenty_volume_b', 'twenty_last_added_volume_a', \
       'twenty_last_added_volume_b', 'twenty_percent_mid_dev_volume_a', \
       'twenty_percent_mid_dev_volume_b', 'twenty_percent_volume_a', \
       'twenty_percent_volume_b']

Target = 'VWAP_10_n_ask'
Lag = 10
Write = False

#normalized features

#needed for some data type correction for filer_to_minutes
def pandas_time_object_to_seconds(pt):
    
    if type(pt) == pd._libs.tslibs.timestamps.Timestamp:
        return(pt.to_pydatetime().time().second)
    
    if type(pt) == pd.core.series.Series:
        if type(pt.iloc[0]) == pd._libs.tslibs.timestamps.Timestamp:
            return(pt.iloc[0].to_pydatetime().time().second)
        if type(pt.iloc[0]) == datetime:
            return(pt.iloc[0].time().second)
        
#filers the entries from a Dataframe that are not within -10 or 10 seconds of the whole minute
def filter_to_minutes(df):
    check = 0
    temp = df
    if temp.index.name == 'Timestamp':
        check = 1
        temp.reset_index(inplace = True)
    temp['temp'] = temp['Timestamp']
    temp['temp'] = temp['temp'].apply(pandas_time_object_to_seconds) 
    temp = temp.loc[(temp['temp'] <= 10) | (temp['temp'] >= 50)]
    temp.drop(['temp'], axis = 1, inplace = True)
    if check ==1:
        temp.set_index('Timestamp', inplace = True)
    return(temp)

#needed for some datatype correction in the Timestamps before applying sort
def unlist_things(X):
    if type(X) == pd.core.series.Series:
        X = X.iloc[0]
    return(X)
    
#builds rolling averages for a given set of features 
def build_ra(df, features, lag):
    for i in features:
        df[i + '_ra'] = df[i].rolling(10).mean().shift(-lag)
    df = df[:-10]
    return(df)

#scales normalized data so it doesn't drop below machine precision 
def scale_data(df):
    df.astype(float)
    normalized_columns = ['VWAP_10_n_ask', 'VWAP_10_n_bid','VWAP_10_n_spread',\
                     'VWAP_20_n_ask', 'VWAP_20_n_bid','VWAP_20_n_spread',\
                    'Vola_last_10']
    small_columns = []
    for i in normalized_columns:
        if i in df.columns:
            small_columns.append(i)

    df[small_columns] =  df[small_columns].apply(lambda x: x*10000)
    return df


#applys a filter, a sort, a dropna, the scale for the normalized data, the build RAs and makes sure
#the datatype are just floats 
def prepare_data(df, features, target, lag):
    temp = filter_to_minutes(df[features])
    temp.reset_index()['Timestamp'].apply(unlist_things).sort_values(inplace = True, ascending = True)
    temp.dropna(inplace = True)
    temp = scale_data(temp)
    temp = build_ra(temp, Features, 10)
    temp['target'] = temp[target].shift(-lag)
    temp = temp[0:-lag].astype(float)
    return(temp)

#prepare the Data to work with
Working_data = prepare_data(df = BTC_USD_data, \
                            features = Features, \
                            target = Target, \
                            lag = Lag)



#Action!
#returns a list of points at wich the training data sets start
def get_training_starting_points(df):
    length = df.shape[0]
    nr_of_batches = round((length-90*1440)/(30*1440))
    starting_points = []
    for i in range(nr_of_batches):
        starting_points.append(1440*30*i)
    return(starting_points)

#applys either a RandomForestRegressor or a LinearRegressor to the df, and returns either the 
# mean squared errors or the mean squared errors and the actuall predictions
def apply_model(df, method, ret_pred, debug):
    
    
    n_est = 1000
    if debug == 1:
        n_est = 152
    
    df.dropna(inplace = True)

    #df.sort_values(by = 'Timestamp', ascending = True, inplace = True)
    #df.reset_index()['Timestamp'].apply(unlist_things).sort_values(inplace = True, ascending = True)
    points = get_training_starting_points(df)
    
    results = []
    Data = {
        'prediction':[],
        'benchmark':[],
        'benchmark_2':[],
        'Value':[],
        'Timestamp':[]
    }
    for i in points:
        
        if method == 0:
            model = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=n_est, n_jobs=-1)
        if method == 1:
            model = LinearRegression(n_jobs = -1)
            
        train_df = df[i:i+1440*90]
        Y_train = train_df['target']
        X_train = train_df[Features]
        
        model.fit(X_train,Y_train)
        
        test_df = df[i+1440*90:i+1440*90 +1440*30]
        Y_test = test_df['target']
        X_test = test_df[Features]
        Timestamp = X_test.index.tolist()
        
        prediction_test = model.predict(X_test).tolist()
        prediction_train = model.predict(X_train).tolist()
        
        benchmark_2_test = [sum(Y_test)/len(Y_test)] * 43200
        benchmark_2_train = [sum(Y_train)/len(Y_train)] * 129600

        mse_train = mean_squared_error(Y_train, prediction_train)
        mse_test = mean_squared_error(Y_test, prediction_test)
        mse_benchmark_train = mean_squared_error(Y_train, X_train[Target])
        mse_benchmark_test = mean_squared_error(Y_test, X_test[Target])
        mse_benchmark_2_train = mean_squared_error(Y_train, benchmark_2_train)
        mse_benchmark_2_test = mean_squared_error(Y_test, benchmark_2_test)
        
        result = {
            'i': i,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'benchmark_mse_train': mse_benchmark_train,
            'benchmark_mse_test': mse_benchmark_test,
            'benchmark_2_mse_train': mse_benchmark_2_train,
            'benchmark_2_mse_test': mse_benchmark_2_test
        }
        if method == 0:
            result['feature_imp'] =  model.feature_importances_
        elif method == 1:
            result['Coefficients'] = model.coef_
            
        results.append(result)
        if ret_pred == 1:
            
            Data['prediction'] = Data['prediction'] + prediction_test
            Data['benchmark'] = Data['benchmark'] + X_test[Target].values.tolist()
            Data['benchmark_2'] = Data['benchmark_2'] + benchmark_2_test
            Data['Value'] = Data['Value'] + Y_test.values.tolist()
            Data['Timestamp'] = Data['Timestamp'] + Timestamp
        
    if ret_pred == 1:
        prediction_frame = pd.DataFrame(data = Data)
        prediction_frame.set_index('Timestamp', inplace = True)
        return(prediction_frame, pd.DataFrame(results))
    return(results)

#apply the function 
res_pre, res_mse = apply_model(df = Working_data, \
            method= 0, \
            ret_pred = 1, \
            debug = 0)

#if Write == 1 while executing the whole script it will save the results to a hdf file
if Write == 1:
    filename_pred = 'C:/Users/janpa/Desktop/Uni/Bachelorarbeit/gathered_stats/BTC_USD_BitFinex_'\
            +Target+'lag_'+str(Lag)+'_prediction'+'.h5'

    filename_errors = 'C:/Users/janpa/Desktop/Uni/Bachelorarbeit/gathered_stats/BTC_USD_BitFinex_'\
            +Target+'lag_'+str(Lag)+'_errors'+'.h5'


    results[0].to_hdf(filename_pred, key = '', mode = 'w')
    results[1].to_hdf(filename_errors, key = '', mode = 'w')

#spread normalisieren