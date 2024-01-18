

import pandas as pd
import numpy as np
import os
import pickle
import glob
from datetime import datetime

# custom functions defined in a sepearate file 
from ML_Functions import *

# Constants
INPUT_DIR = '../data'
WRF_PICKLE_ALL_DATA = '/WRF_pkl/multi_sites_full_o3_1720.pkl'
OUTPUT_DIR = '../results'
GTM_OFFSET_FILE = './gmtoff.csv'
AIRNOW_OBS_URL_TEMPLATE = "http://lar.wsu.edu/R_apps/2020ap5/data/byAQSID/{}.apan"
AQS_FILE_PATTERN = '/AQS/download/44201/{}_{}_88101.csv'


def set_season_months(pollutant):
    global start_month, end_month

    if pollutant == 'O3':
        start_month = 4
        end_month = 10

    elif pollutant == 'PM':
        start_month = 10
        end_month = 3

def preprocess_airnow_data(data, pollutant):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)
    data.index -= np.timedelta64(8, 'h')  # Adjust for timezone

    if pollutant == 'O3':
        data['O3_obs'] = data['OZONEan']
        data['O3_mod'] = data['OZONEap']

        data = data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(), freq='1H'))
        for col in ['O3_obs','O3_mod']:
            suffix = '' if col == 'O3_obs' else '_ap'
            data[f'O3avg8hr{suffix}'] = data[col].rolling(8, min_periods=6).mean()
            data[f'AQI_class{suffix}'] = pd.cut(round(data[f'O3avg8hr{suffix}'].fillna(-1)),
                                              [0, 54, 70, 85, 105, 200, np.inf],
                                              labels=[1, 2, 3, 4, 5, 6])
            if col == 'O3_obs':
                data[f'Past_8hr_O3{suffix}'] = data[f'O3avg8hr{suffix}'].shift(24)
                data['Past_hr_O3'] = data['Observed'].shift(24)

    elif pollutant == 'PM':
        data['PM2.5_obs'] = data['PM2.5an']
        data['PM2.5_mod'] = data['PM2.5ap']

        data = data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(), freq='1H'))
        for col in ['PM2.5_obs', 'PM2.5_mod']:
            suffix = '' if col == 'PM2.5_obs' else '_ap'
            data[f'PMavg24hr{suffix}'] = data[col].rolling(24, min_periods=18).mean()
            data[f'AQI_class{suffix}'] = pd.cut(round(data[f'PMavg24hr{suffix}'].fillna(-1)),
                                              [0, 12, 35.4, 55.4, 150.4, 250.4, np.inf],
                                              labels=[1, 2, 3, 4, 5, 6])
            if col == 'PM2.5_obs':
                data[f'Past_24hr_PM{suffix}'] = data[f'PMavg24hr{suffix}'].shift(24)
                data['Past_hr_PM'] = data['PM2.5_obs'].shift(24)

    return data

def load_2020_pickle_data(site_id):
    try:
        with open(f"{INPUT_DIR}/WRF_pkl/data_wrf2020_{site_id}.pkl", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f'Pickle file not found for site {site_id}')
        return None

def replace_airnow_with_aqs_data(df_tmp, site_id, pollutant, gmtoff):
    if pollutant == 'O3':
        aqs_files = glob.glob(f'{INPUT_DIR}/AQS/download/44201/{site_id}_*_88101.csv')
        observed_column = 'O3_obs'
        sample_measurement_column = 'OZONE_obs'
    elif pollutant == 'PM':
        aqs_files = glob.glob(f'{INPUT_DIR}/AQS/download/88101/{site_id}_*_88101.csv') + \
                    glob.glob(f'{INPUT_DIR}/AQS/download/88502/{site_id}_*_88502.csv')
        observed_column = 'PM2.5_obs'
        sample_measurement_column = 'PM2.5_obs'

    if aqs_files:
        for file_path in aqs_files:
            aqs_data = pd.read_csv(file_path)
            aqs_data = aqs_data[aqs_data['sample_duration'] == '1 HOUR']
            if not aqs_data.empty:
                break

        if not aqs_data.empty:
            aqs_data.index = pd.to_datetime(aqs_data['Datetime'])
            aqs_data = aqs_data.drop_duplicates(subset='Datetime', keep="first")
            tz_delta = np.timedelta64(abs(gmtoff.loc[gmtoff['AQSID'] == site_id, 'GMToff'].values[0]), 'h')
            aqs_data.index -= tz_delta
            aqs_data.rename(columns={"sample_measurement": observed_column}, inplace=True)
            df_tmp = pd.concat([df_tmp, aqs_data[[observed_column]]], axis=1)
        else:
            print(f'No valid AQS data found for {site_id}')
    else:
        print(f'NO AQS data for {site_id}')
    return df_tmp


def process_data(site_id, pollutant, data_dict, gmtoff):        
    
    single_site = data_dict[site_id]
    single_site = single_site.query(f'{start_month} < index.month < {end_month}')

    if not single_site.empty:
        obs_ap_tmp = None
        try:
            obs_ap_tmp = pd.read_csv(AIRNOW_OBS_URL_TEMPLATE.format(site_id))
        except FileNotFoundError as e:
            print(f'Error loading data for site {site_id}: {e}')
            return

        if obs_ap_tmp is None:
            return

        obs_ap = preprocess_airnow_data(obs_ap_tmp)
        if obs_ap.dropna().empty:
            return

        dict_his2 = load_2020_pickle_data(site_id)
        if dict_his2 is None:
            return

        df_tmp = single_site.combine_first(dict_his2.get('WRFRT', pd.DataFrame()))
        df_tmp = df_tmp.combine_first(obs_ap)
        df_tmp.update(obs_ap)
        df_tmp = replace_airnow_with_aqs_data(df_tmp, site_id, gmtoff)

    # Additional processing
        if pollutant == 'O3':
            df_tmp['O3avg8hr'] = df_tmp['O3_obs'].rolling(8, min_periods=6).mean()
            df_tmp['AQI_class'] = pd.cut(round(df_tmp['O3avg8hr'].fillna(-1)),
                                        [0, 54, 70, 85, 105, 200, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])
            df_tmp['Past_hr_O3'] = df_tmp['O3_obs'].shift(24)
            df_tmp['Weekday'] = df_tmp.index.dayofweek
            df_tmp['Month'] = df_tmp.index.month
            df_tmp['Hour'] = df_tmp.index.hour

            df_tmp = df_tmp[['Past_hr_O3','O3_obs','O3_ap','O3avg8hr', 'O3avg8hr_ap', 'PBL_m', 'Surface_pres_Pa', 'Temp_K', 'U_m_s', 'V_m_s',
                            'RH_pct', 'Past_8hr_O3','Month','Hour','Weekday','AQI_class']].copy()
            # Final filtering for the O3 season
            df_tmp = df_tmp.query(f'{start_month} < index.month < {end_month}')

        elif pollutant == 'PM':
            df_tmp['PMavg24hr'] = df_tmp['PM2.5_obs'].rolling(24, min_periods=18).mean()
            #fill na with ML forecasting
            #obs_ap['PMavg24hr']=obs_ap['PMavg24hr'].fillna(dict_his_tmp['mean'][dict_his_tmp['mean'].index.date<d]['PM2.5_pred'])
            df_tmp['AQI_class'] = pd.cut(round(df_tmp['PMavg24hr'].fillna(-1)),
                [0, 12, 35.4, 55.4, 150.4, 250.4, np.inf],
                labels=[1, 2, 3, 4, 5, 6])
            df_tmp['Past_hr_PM'] = df_tmp['PM2.5_obs'].shift(24)
            
            df_tmp['Weekday']=df_tmp.index.dayofweek
            df_tmp['Month'] = df_tmp.index.month
            df_tmp['Hour'] = df_tmp.index.hour
            df_tmp = df_tmp[['PM2.5_obs','PM2.5_mod','Past_hr_PM','PMavg24hr', 'PMavg24hr_ap', 'PBL_m', 'Surface_pres_Pa', 'Temp_K', 'U_m_s', 'V_m_s',
                                    'RH_pct', 'Past_24hr_PM','Month','Hour','Weekday','AQI_class']]
            df_tmp = df_tmp.query(f'{start_month} < index.month < {end_month}')

        return df_tmp
    else:   
        print(f'Data is not available at this site: {site_id}')

# Main script execution
os.makedirs(OUTPUT_DIR, exist_ok=True)
gmtoff = pd.read_csv(GTM_OFFSET_FILE)

with open(os.path.join(INPUT_DIR, WRF_PICKLE_ALL_DATA), 'rb') as fp:
    data_dict = pickle.load(fp)

for pollutant in ['O3', 'PM']:

    # set a particular months you are interested
    set_season_months(pollutant)
    print(f'Start month: {start_month}, End month: {end_month}')

    for site in data_dict:
        output_file = os.path.join(OUTPUT_DIR,'/data_2RF_{pollutant}_{start_month}_{end_month}_at_{site}.csv')
        if not os.path.isfile(output_file):
            final_df = process_data(site, data_dict, gmtoff, pollutant)
            if final_df is not None:
                final_df.to_csv(os.path.join(INPUT_DIR, '/final_input/Final_PNW_ML_{pollutant}_{start_month}_{end_month}_at_{site}.csv'))