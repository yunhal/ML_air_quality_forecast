import pylab
#pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
#matplotlib
import matplotlib.pyplot as plt
#datetime
import datetime as dt
#import numpy
import numpy as np
#sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#gaussian_kde
from scipy.stats import gaussian_kde
#anchored text
from matplotlib.offsetbox import AnchoredText 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold
import time
import glob
import pickle
import os
import re

#open our ML functions
exec(open(r"./ML_Functions.py").read())


##################################

# Constants for model selection
MODEL_A = 'RF_classifier'
MODEL_B = 'RF_2phase_regressor' 

# Set this constant to the model you want to use
SELECTED_MODEL = MODEL_A

# Constants file path and names
INPUT_DIR = '/Users/yunhalee/Desktop/DLair/data/final_input' #  pre-processed data of obs O3, AIRPACT O3, and WRF met datasets from 2017 to 2020
OUTPUT_DIR = '../results/multi_site_results'

##################################

def preprocess_data(df_site, pollutant):

    if SELECTED_MODEL == MODEL_A: #random forest classifier

        if pollutant == 'O3':
            labels_to_drop = ['O3_obs', 'O3_ap', 'O3avg8hr', 'O3avg8hr_ap', 'Past_hr_O3', 'PBL_m'] 
            target_column = ['AQI_class', 'O3avg8hr']

        elif pollutant == 'PM25':
            labels_to_drop = ['PM2.5_obs', 'PM2.5_mod','PMavg24hr', 'PMavg24hr_ap','Past_hr_PM', 'PBL_m']
            target_column = ['AQI_class', 'PMavg24hr']
        else:
            raise ValueError("Unsupported pollutant type. Only O3 and PM25 are available.")

    elif SELECTED_MODEL == MODEL_B: 
    
        if pollutant == 'O3':
            labels_to_drop = ['O3_obs', 'O3_ap', 'O3avg8hr', 'O3avg8hr_ap', 'Past_8hr_O3', 'AQI_class', 'PBL_m']
            target_column = 'O3_obs'

        elif pollutant == 'PM25':
            labels_to_drop = ['PM2.5_obs', 'PM2.5_mod', 'PMavg24hr', 'PMavg24hr_ap', 'AQI_class', 'Past_24hr_PM', 'PBL_m']
            target_column = 'PM2.5_obs'
        else:
            raise ValueError("Unsupported pollutant type. Only O3 and PM25 are available.")


    else:
        raise ValueError("Unsupported MODEL. Please check SELECTED_MODEL")

    features = df_site.drop(columns=labels_to_drop)
    processed_features = pd.DataFrame(preprocess('MAS', features))
    processed_features.columns = features.columns
    processed_features.index = features.index

    return processed_features, df_site[target_column]


def get_data_subsets(pred1, pred2, y, sep1, sep2):
    low_index = np.where(pred1 < sep1)
    med_index = np.where((pred1 >= sep1) & (pred1 <= sep2))
    high_index = np.where(pred1 > sep2)

    X_low = list(zip(pred1[low_index], pred2[low_index]))
    Y_low = y[low_index]

    X_med = list(zip(pred1[med_index], pred2[med_index]))
    Y_med = y[med_index]

    X_high = list(zip(pred1[high_index], pred2[high_index]))
    Y_high = y[high_index]

    return (X_low, Y_low), (X_med, Y_med), (X_high, Y_high)

def make_adjusted_predictions(model_RF, model_RF2, X, y, coefs):
    pred1 = model_RF.predict(X)
    pred2 = model_RF2.predict(X)

    sep1 = np.percentile(pred1, 33)
    sep2 = np.percentile(pred1, 67)

    pred_adjusted = np.empty_like(pred1)
    # Calculate coefficients for different ranges
    low_coef, med_coef, high_coef = coefs  # Assuming coefs = [(low1, low2), (med1, med2), (high1, high2)]

    # Initialize adjusted predictions array
    pred_adjusted = np.empty_like(pred1)

    # Apply coefficients based on the percentiles
    low_indices = pred1 < sep1
    med_indices = (pred1 >= sep1) & (pred1 <= sep2)
    high_indices = pred1 > sep2

    pred_adjusted[low_indices] = low_coef[0] * pred1[low_indices] + low_coef[1] * pred2[low_indices]
    pred_adjusted[med_indices] = med_coef[0] * pred1[med_indices] + med_coef[1] * pred2[med_indices]
    pred_adjusted[high_indices] = high_coef[0] * pred1[high_indices] + high_coef[1] * pred2[high_indices]

    # Correct negative predictions
    pred_adjusted[pred_adjusted < 0] = pred1[pred_adjusted < 0]

    return pred_adjusted

def process_pm25_data(df_split, pm25_pred_column, counter):

    pm25_pred_final = 'PMavg24hr_ML'
    

    if SELECTED_MODEL == MODEL_A:

        target_columns = ['PMavg24hr', 'PMavg24hr_ap', pm25_pred_column]
        # subset the dataframe
        df_sub = df_split[target_columns]

        # Reindexing to hourly data and applying shifts
        df_reindexed = df_sub.reindex(pd.date_range(start=df_sub.index.min(), end=df_sub.index.max(), freq='1H'))
        df_reindexed['PMavg24hr'] = df_reindexed['PMavg24hr'].shift(-7)
        df_reindexed['PMavg24hr_ap'] = df_reindexed['PMavg24hr_ap'].shift(-7)
        df_reindexed[pm25_pred_final] = df_reindexed[pm25_pred_column].shift(-7)

    elif SELECTED_MODEL == MODEL_B:

        target_columns = ['PM2.5_obs', 'PM2.5_ap', pm25_pred_column]
        # subset the dataframe
        df_sub = df_site[target_columns]
    
        # Reindexing to hourly data and computing rolling means
        df_reindexed = df_sub.reindex(pd.date_range(start=df_sub.index.min(), end=df_sub.index.max(), freq='1H'))
        df_reindexed['PMavg24hr'] = df_reindexed['PM2.5_obs'].rolling(24, min_periods=18).mean().shift(-23)
        df_reindexed['PMavg24hr_ap'] = df_reindexed['PM2.5_mod'].rolling(24, min_periods=18).mean().shift(-23)
        df_reindexed[pm25_pred_final] = df_reindexed[pm25_pred_column].rolling(24, min_periods=18).mean().shift(-23)

    # Selecting data at 00:00 hour for daily data
    df_daily = df_reindexed[df_reindexed.index.hour == 0]

    # AQI calculations
    aqi_bins = [-np.inf, 12, 35.4, 55.4, 150.4, 250.4, np.inf]
    aqi_labels = [1, 2, 3, 4, 5, 6]
    df_daily['AQI_day'] = pd.cut(df_daily['PMavg24hr'], bins=aqi_bins, labels=aqi_labels)
    df_daily['AQI_pred_day'] = pd.cut(df_daily[pm25_pred_final], bins=aqi_bins, labels=aqi_labels)
    df_daily['AQI_ap_day'] = pd.cut(df_daily['PMavg24hr_ap'], bins=aqi_bins, labels=aqi_labels)

    # Adjust index to reflect the day the data represents
    df_daily.index += pd.DateOffset(hours=5)

    # Custom AQI Calculation
    aqi_low = [0, 51, 101, 151, 201, 301, 401]
    aqi_high = [50, 100, 150, 200, 300, 400, 500]
    aqi_lowc = [0, 12.1, 35.5, 55.5, 150.5, 250.5, 350.5]
    aqi_highc = [12, 35.4, 55.4, 150.4, 250.4, 350.4, 500]
    df_daily['AQI'] = np.nan
    for i in range(len(df_daily)):
        if np.isnan(df_daily[pm25_pred_final][i]) or df_daily[pm25_pred_final][i] < 0:
            continue
        aqi_class = df_daily['AQI_pred_day'][i] - 1
        df_daily.at[df_daily.index[i], 'AQI'] = (
            (aqi_high[aqi_class] - aqi_low[aqi_class]) /
            (aqi_highc[aqi_class] - aqi_lowc[aqi_class]) *
            (round(df_daily[pm25_pred_final][i], 1) - aqi_lowc[aqi_class]) +
            aqi_low[aqi_class]
        )
        df_daily.at[df_daily.index[i], 'AQI'] = round(df_daily.at[df_daily.index[i], 'AQI'], 1)

    return df_daily, counter + 1

def process_o3_data(df_split, o3_pred_column, counter):

    o3_pred_final = 'O3avg8hr_ML'

    if SELECTED_MODEL == MODEL_A:

        target_columns = ['O3avg8hr', 'O3avg8hr_ap', o3_pred_column]

        # subset the dataframe
        df_sub = df_split[target_columns]

        # Reindexing to hourly data and applying shifts
        df_reindexed = df_sub.reindex(pd.date_range(start=df_split.index.min(), end=df_split.index.max(), freq='1H'))
        df_reindexed['O3avg8hr'] = df_reindexed['O3avg8hr'].shift(-7)
        df_reindexed['O3avg8hr_ap'] = df_reindexed['O3avg8hr_ap'].shift(-7)
        df_reindexed[o3_pred_final] = df_reindexed[o3_pred_column].shift(-7)

    elif SELECTED_MODEL == MODEL_B:
        target_columns = ['O3_obs', 'O3_ap', o3_pred_column]
        # subset the dataframe
        df_sub = df_split[target_columns]

        # Reindexing to hourly data and computing rolling means
        df_reindexed = df_sub.reindex(pd.date_range(start=df_split.index.min(), end=df_split.index.max(), freq='1H'))
        df_reindexed['O3avg8hr'] = df_reindexed['O3_obs'].rolling(8, min_periods=6).mean().shift(-7)
        df_reindexed['O3avg8hr_ap'] = df_reindexed['O3_ap'].rolling(8, min_periods=6).mean().shift(-7)
        df_reindexed[o3_pred_final] = df_reindexed[o3_pred_column].rolling(8, min_periods=6).mean().shift(-7)

    # Calculating daily max 8-hour average and shifting
    avg_columns = ['O3avg8hr', 'O3avg8hr_ap', o3_pred_final]
    for col in avg_columns:
        df_reindexed[f'{col}_maxdaily8hravg'] = df_reindexed[col].rolling(17, min_periods=13).max().shift(-16)

    # Selecting data at 07:00 hour for daily data
    df_daily = df_reindexed[df_reindexed.index.hour == 7]

    # AQI calculations
    aqi_bins = [0, 54, 70, 85, 105, 200, np.inf]
    aqi_labels = [1, 2, 3, 4, 5, 6]
    for col in avg_columns:
        df_daily[f'AQI_{col}_day'] = pd.cut(df_daily[f'{col}_maxdaily8hravg'], bins=aqi_bins, labels=aqi_labels)

    # Custom AQI Calculation
    aqi_low = [0, 51, 101, 151, 201, 301]
    aqi_high = [50, 100, 150, 200, 300, 500]
    aqi_lowc = [0, 55, 71, 86, 106, 201]
    aqi_highc = [54, 70, 85, 105, 200, 600]
    df_daily['AQI'] = np.nan
    for i in range(len(df_daily)):
        if np.isnan(df_daily[f'{o3_pred_final}_maxdaily8hravg'][i]):
            continue
        aqi_class = df_daily[f'AQI_{o3_pred_final}_day'][i] - 1
        df_daily.at[df_daily.index[i], 'AQI'] = (
            (aqi_high[aqi_class] - aqi_low[aqi_class]) /
            (aqi_highc[aqi_class] - aqi_lowc[aqi_class]) *
            (round(df_daily[f'{o3_pred_final}_maxdaily8hravg'][i]) - aqi_lowc[aqi_class]) +
            aqi_low[aqi_class]
        )
        df_daily.at[df_daily.index[i], 'AQI'] = round(df_daily.at[df_daily.index[i], 'AQI'])

    # Adjust index to reflect the day the data represents
    df_daily.index += pd.DateOffset(hours=5)

    return df_daily, counter + 1


def train_and_predict_RF_2phase(df, X, y, pollutant, all_dates):
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=12883823)

    all_results = {}

    for i, (train_index, test_index) in enumerate(rkf.split(all_dates)):
        train_datetime_index = pd.to_datetime(df.index.date).isin(all_dates[train_index])
        test_datetime_index = pd.to_datetime(df.index.date).isin(all_dates[test_index])
        X_train, X_test = X[train_datetime_index], X[test_datetime_index]
        y_train, y_test = y[train_datetime_index], y[test_datetime_index]

        # Create a new DataFrame for training data
        new_train = pd.DataFrame(X_train)
        new_train['obs'] = y_train
        new_train['pred'] = model_RF.predict(X_train)
        new_train['diff'] = abs(new_train['pred'] - new_train['obs'])

        # Filter out rows where the difference is greater than 5
        new_train = new_train[new_train['diff'] > 5]

        # Prepare second training set
        X_train2 = new_train.drop(columns=['obs', 'pred', 'diff']).to_numpy()
        y_train2 = y_train[new_train.index]

        # Train two sets of RandomForest model
        model_RF = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=137)
        model_RF.fit(X_train, y_train)
        model_RF2 = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=137)
        model_RF2.fit(X_train2, y_train2)

        # Predictions for segregation
        pred1 = model_RF.predict(X_train)
        pred2 = model_RF2.predict(X_train)

        # Calculate percentiles for classification
        sep1 = np.percentile(pred1, 33)
        sep2 = np.percentile(pred1, 67)

        coefs = []
        data_subsets = get_data_subsets(pred1, pred2, y_train, sep1, sep2)

        for X_subset, Y_subset in data_subsets:
            model_LR = LinearRegression(fit_intercept=False)
            model_LR.fit(X_subset, Y_subset)
            coefs.append(model_LR.coef_)


        pred3 = make_adjusted_predictions(model_RF, model_RF2, np.array(X_test), y_train, coefs)

        df_split = df[test_datetime_index].copy()


        dict_name = f'iteration_{counter}'

        if pollutant == 'O3':
            df_split = df_split.assign(O3_pred=pred3)
            delta = np.timedelta64(7,'h')  #7 hr shift to compute DMA8 O3 (starts from 7 am)
            df_split.index += delta
            # Currently, saving df_split is turned off. 
            # all_df_split[dict_name] = df_split
            df_split_processed, counter = process_o3_data(df_split, 'O3_pred')

        elif pollutant == 'PM25':
            df_split = df_split.assign(PM25_pred=pred3)
                        
            # Currently, saving df_split is turned off. 
            # all_df_split[dict_name] = df_split

            df_split_processed, counter = process_pm25_data(df_split, 'PM25_pred')

        # Store processed data in the dictionary
        all_results[dict_name] = df_split_processed

    return all_results

def train_and_predict_RF_classifier(df, X, y, pollutant, all_dates):

    # preparing data for training
    y1 = y['AQI_class'].to_numpy() 
    y2 = y['O3avg8hr'].to_numpy()
    
    X1 = X.drop(labels=['AQI_class'], axis=1).to_numpy()  # Convert to numpy array

    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=12883823)

    all_results = {}
    counter = 0

    for i, (train_index, test_index) in enumerate(rkf.split(all_dates)):
        train_datetime_index = pd.to_datetime(df.index.date).isin(all_dates[train_index])
        test_datetime_index = pd.to_datetime(df.index.date).isin(all_dates[test_index])


        # Splitting the data
        X_train1, X_test1 = X1[train_datetime_index], X1[test_datetime_index]
        y_train1, y_test1 = y1[train_datetime_index], y1[test_datetime_index]


        # Random Forest Model for AQI Classification
        model_RF = RandomForestClassifier(n_estimators=100, max_depth=7, max_features='sqrt',
                                        random_state=137, class_weight='balanced_subsample')
        model_RF.fit(X_train1, y_train1)
        pred_AQI = model_RF.predict(X_test1)

        # Preparing data for Linear Regression
        X_dat2 = X.copy()
        X_dat2.loc[test_datetime_index, 'AQI_class'] = pred_AQI / df['AQI_class'].max()
        X2 = X_dat2.to_numpy() # Convert to numpy array directly

        X_train2, X_test2 = X2[train_datetime_index], X2[test_datetime_index]
        y_train2, y_test2 = y2[train_datetime_index], y2[test_datetime_index]

        # Linear Regression Model for Prediction
        model_LR = LinearRegression()
        rfe = RFE(model_LR, n_features_to_select=5)
        rfe.fit(X_train2, y_train2)
        pred = rfe.predict(X_test2)

        # Storing results
        df_split = df[test_datetime_index].copy()
        dict_name = f'iteration_{counter}'

        if pollutant == 'O3':
            df_split = df_split.assign(O3_pred=pred)
            delta = np.timedelta64(7,'h')  #7 hr shift to compute DMA8 O3 (starts from 7 am)
            df_split.index += delta
            df_split_processed, counter = process_o3_data(df_split, 'O3_pred')

        elif pollutant == 'PM25':
            df_split = df_split.assign(PM25_pred=pred)
            df_split_processed, counter = process_pm25_data(df_split, 'PM25_pred')

        # Store processed data in the dictionary
        all_results[dict_name] = df_split_processed

    return all_results

def process_and_save_o3_data(result_dict, outputdir, site):

    target_columns = ['O3avg8hr_maxdaily8hravg', 'O3avg8hr_ap_maxdaily8hravg', 'OO3avg8hr_ML_maxdaily8hravg']

    # Concatenate the results and select relevant columns
    big_df = pd.concat(result_dict)[['datetime'] + target_columns]
    big_df = big_df.groupby('datetime').mean().dropna(how='all')

    # Calculate AQI for observed, predicted, and modeled O3 data
    aqi_bins = [0, 54, 70, 85, 105, 200, np.inf]
    aqi_labels = [1, 2, 3, 4, 5, 6]
    big_df['AQI_day'] = pd.cut(round(big_df['O3avg8hr_maxdaily8hravg']), bins=aqi_bins, labels=aqi_labels)
    big_df['AQI_pred_day'] = pd.cut(round(big_df['O3avg8hr_ML_maxdaily8hravg']), bins=aqi_bins, labels=aqi_labels)
    big_df['AQI_ap_day'] = pd.cut(round(big_df['O3avg8hr_ap_maxdaily8hravg']), bins=aqi_bins, labels=aqi_labels)

    # Save the DataFrame to a CSV file
    file_path = f'{outputdir}/O3_{SELECTED_MODEL}_{start_month}_{end_month}_at_{site}_8hrmax.csv'
    big_df.to_csv(file_path)

def process_and_save_pm25_data(result_dict, outputdir, site):

    target_columns = ['PMavg24hr_obs', 'PMavg24hr_ap', 'PMavg24hr_ML']

    # Concatenate the results and select relevant columns
    big_df = pd.concat(result_dict)[['datetime'] + target_columns]
    big_df = big_df.groupby('datetime').mean().dropna(how='all')
    
    # Calculate AQI for observed, predicted, and modeled PM2.5 data
    aqi_bins = [-np.inf, 12, 35.4, 55.4, 150.4, 250.4, np.inf]
    aqi_labels = [1, 2, 3, 4, 5, 6]

    big_df['AQI_day'] = pd.cut(round(big_df['PMavg24hr_org']), bins=aqi_bins, labels=aqi_labels)
    big_df['AQI_pred_day'] = pd.cut(round(big_df['PMavg24hr_ML']), bins=aqi_bins, labels=aqi_labels)
    big_df['AQI_ap_day'] = pd.cut(round(big_df['PMavg24hr_ap']), bins=aqi_bins, labels=aqi_labels)

    # Save the DataFrame to a CSV file
    file_path = f'{outputdir}/PM25_{SELECTED_MODEL}_{start_month}_{end_month}_at_{site}.csv'
    big_df.to_csv(file_path)



##################################
# main starts here
##################################
    
for pollutant in ['O3', 'PM25']:

    # set the month duration 
    global start_month, end_month
    if pollutant == 'O3':
        start_month = 4
        end_month = 10

    elif pollutant == 'PM25':
        start_month = 10
        end_month = 3

    #create the output dir if it doesn't exist
    outputdir=os.path.join(OUTPUT_DIR, pollutant)
    os.makedirs(outputdir, exist_ok=True)

    # loop over each site
    for file in glob.glob(INPUT_DIR+'/Final_PNW_ML_{pollutant}_{start_month}_{end_month}_*.csv'):
        site = re.findall("[0-9]{4,}", file)[0]
        print(file, "site #: ", site)

        if os.path.isfile(outputdir+f'/{pollutant}_{SELECTED_MODEL}_{start_month}_{end_month}_at_{site}.csv'): continue

        # read csv files
        df_site = pd.read_csv(INPUT_DIR+'/Final_PNW_ML_{pollutant}_{start_month}_{end_month}_at_{site}+.csv', index_col=0)

        df_site.index = pd.to_datetime(df_site.index)  # Converting the index to date
        
        if pollutant == 'O3':
            delta = np.timedelta64(7,'h')  #7 hr shift to compute DMA8 O3 (starts from 7 am)
            df_site.index -= delta
        
        df_site = df_site.dropna()
        all_dates = np.array(sorted(set(df_site.index.date)))


        # preprocess the data
        X_features, y_target = preprocess_data(df_site, pollutant)

        if SELECTED_MODEL == MODEL_A:
            result_dict= train_and_predict_RF_classifier(df_site, X_features, y_target, pollutant, all_dates)
        elif SELECTED_MODEL == MODEL_B:
            result_dict= train_and_predict_RF_2phase(df_site, X_features, y_target, pollutant, all_dates)
        else:
            raise ValueError("Unsupported MODEL. Please check SELECTED_MODEL")

        # Set index for the output dataframe
        for k in result_dict.keys():
            result_dict[k]['datetime']=result_dict[k].index

        if pollutant == 'O3': 
            process_and_save_o3_data(result_dict, outputdir, site)
        elif pollutant == 'PM25':
            process_and_save_pm25_data(result_dict, outputdir, site)



