# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:31:53 2025

@author: eldenbreejen
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress

def get_feature_input_table(dataset, tabular_data, prediction_mode, observation_window, gap_window,  date, feature_windows, feature_selection=False, beta=False):
    '''
    Function to obtain DataFrame with label and input features per sample.

    Parameters
    ----------
    dataset : str
        Dataset name, folder with patient files.
    tabular_data : DataFrame
        Table with tabular data and event/control samples per record.
    prediction_mode : str
        Ventilation mode during prediction.
    observation_window : int
        Observation window length in hours.
    gap_window : int
        Horizon length in hours.
    date : str
        Current datetime, used in input feature table filename.
    feature_windows : int
        Number of feature windows within observation window, used to calculate aggregate features over.
    feature_selection : Bool, optional
        Whether to get only the selected features. The default is False.
    beta : Bool, optional
        Whether to get features for model 1B: before switch, after switch, and diff. The default is False.

    Returns
    -------
    features_df : DataFrame
        Table with label and input features per sample

    '''
    
    # Set patient ID as index and calculate BMI
    tabular_data        = tabular_data.set_index('PatientID')
    tabular_data        = tabular_data.copy()
    tabular_data['BMI'] = tabular_data.Gewicht / (tabular_data.Lengte/100)**2
    tabular_data        = tabular_data.copy()
    
    # Get features from tabular data for each sample
    features = get_tabular_features(tabular_data, prediction_mode, gap_window)
    
    # Get aggregate features from timeseries data for each sample
    if beta is False:
        features_df = get_timeseries_features(dataset, features, prediction_mode, observation_window, gap_window,feature_windows, feature_selection)   
    else:
        features_df = get_timeseries_features2(dataset, features, prediction_mode, observation_window, gap_window,feature_windows, feature_selection)
                
    # Save input feature table
    features_df.to_csv(f'P:/Emmelieve/input_feature_tables_{dataset}/input_feature_values_{date}.csv', index=None)
    print('Feature input table saved')
    print('')
    
    return features_df
        
def get_tabular_features(tabular_data, prediction_mode, gap_window): 
    '''
    Function to get tabular features per sample.    

    Parameters
    ----------
    tabular_data : DataFrame
        Dataframe with tabular data and events/controls per record.
    prediction_mode : str
        Ventilation mode during prediction.
    gap_window : int
        Horizon length in hours.

    Returns
    -------
    None.

    '''  
     
    input_features  = {}
    i               = 0     # Count nr of events
    
    for patient_id, row in tabular_data.iterrows():
        
        # Get event and control timestamps
        events_idx  = row.index[row.index.str.match(r"^event")]
        events      = row[events_idx].dropna()
    
        if len(events) > 0:
            
            # Get features per event
            for index, timestamp in events.items():
                
                event   = 0 if index.startswith('event_control') else 1                 # Get event label (0: control, 1: event)
                gender  = 1 if tabular_data.loc[patient_id, 'Geslacht'] == 'M' else 0   # Get gender
                
                # Save metadata, label, features
                features = {
                    'patient_id'        : patient_id,
                    'hospital_number'   : tabular_data.loc[patient_id, 'HospitalNumber'],
                    'timestamp'         : timestamp,
                    #'switch             : pd.to_datetime(tabular_data.loc[patient_id, f'switch_{index}']),
                    'prediction_window' : tabular_data.loc[patient_id, f'pred_{index}'],
                    'event'             : event,
                    'age'               : tabular_data.loc[patient_id, 'Leeftijd'],
                    'gender'            : gender,
                    'bmi'               : tabular_data.loc[patient_id, 'BMI']
                    }
                
                # Add assisted ventilation duration as feature for predictions during assisted ventilation
                if prediction_mode == 'assisted':
                    features['spont_duration'] = tabular_data.loc[patient_id, f'duration_{index}'].total_seconds() /3600 - gap_window
                    
                input_features[i] = features
                i += 1
    
    return input_features

def get_timeseries_features(dataset, features_per_event, prediction_mode, observation_window, gap_window, feature_windows, feature_selection): 
    '''
    Function to get aggregate features form timeseries data.

    Parameters
    ----------
    dataset : str
        Name of dataset and folder with patient files.
    features_per_event : dict
        Dictionary with tabular features per event.
    prediction_mode : str
        Ventilation mode during predicitions.
    observation_window : int
        Observation window length in hours.
    gap_window : int
        Horizon length in hours.
    feature_windows : int
        Number of windows within observation window to calculate aggregate features from.
    feature_selection : Bool
        Wheter to calculate only the selected features.

    Returns
    -------
    features_df : DataFrame
        DataFrame with input feature values per sample.

    '''
    
    # Parameters to calculate
    parameters_df = pd.read_csv('P:/Emmelieve/feature_variables.csv', delimiter=';')
    
    if feature_selection is True:
        parameters_df = parameters_df[parameters_df[f'{prediction_mode}_selected'] == 1]
    else:
        parameters_df = parameters_df[parameters_df[prediction_mode] == 1]
    
    
    # Get feature values per sample
    previous_patient_id = None
    
    for idx, event in features_per_event.items():
        
        features            = {}
        patient_id          = event['patient_id']
        event_time          = event['timestamp']
        prediction_window   = pd.to_timedelta(event['prediction_window'])
        start_window        = event_time - pd.Timedelta(hours=(observation_window+gap_window)) - prediction_window
        end_window          = event_time - pd.Timedelta(hours=(gap_window)) - prediction_window
        
        # Read new patient file
        if patient_id != previous_patient_id:
            timeseries              = pd.read_csv(f'P:/Emmelieve/{dataset}/{patient_id}.csv', index_col=0)
            timeseries.index.name   = 'Timestamp'
            timeseries.index        = pd.to_datetime(timeseries.index, format="%Y-%m-%d %H:%M:%S")
            previous_patient_id     = patient_id
        
        # Loop over parameters and calculate feature values
        for parameter in parameters_df.itertuples(index=False):

            try:
                # Check if variable is present in data
                keys    = [parameter.parameter_0, parameter.parameter_1, parameter.parameter_2, parameter.parameter_3]
                present = [key for key in keys if key in timeseries.columns]
                
                # Extract/compute column with parameter data
                # Use empty Series if parameter is not present
                if len(present) == 0:
                    timestamps  = pd.date_range(start=start_window, end=end_window, freq="1min")
                    data        = pd.Series(index=timestamps, name=parameter.saveas, dtype=float)
                    
                # Calculate IE, PF, RSBI or SF ratio
                elif parameter.saveas in ['ie_ratio', 'pf_ratio', 'rsbi', 'sf_ratio']:
                    data        = timeseries[parameter.parameter_0] / timeseries[parameter.parameter_1]
        
                # Calculate and save MV duration    
                elif parameter.saveas == 'mv_duration':
                    data        = timeseries[parameter.parameter_0][timeseries.index[0]:(start_window + pd.Timedelta(hours=observation_window))].notna()
                    features[parameter.saveas]  = data.sum()
                    
                # Merge columns with same parameter
                elif len(present) > 1:
                    data        = timeseries[present].bfill(axis=1).iloc[:,0]
                    timeseries[parameter.saveas] = data
                
                # Get parameter column
                else:
                    data = timeseries[present[0]]
                    timeseries[parameter.saveas] = data
    
                # Calculate aggregate feature values
                # Set 0 if medication was not administred (pump rate = 0)
                if parameter.type == 'medication':
                    data = data.fillna(0)
                    
                # Save last parameter value as feature value
                if parameter.last is True:
                    features[parameter.saveas] = data.dropna().iloc[-1]
                
                # Save mean, std, and trend as feature value
                if parameter.mean is True:
               
                    if feature_windows == 1:        # Save mean, std, and trend for 1 feature window
                    
                        # Slice segment and remove infite values
                        segment = data[end_window-pd.Timedelta(hours=feature_windows):end_window]
                        segment = segment.replace([np.inf, -np.inf], np.nan)
                        
                        # Count perc available data points
                        datapoints = segment.notna().sum()/len(segment)
                        
                        # Save mean
                        if datapoints < 0.5:    # Put Nan if more than 50% missing data
                            features[f'{parameter.saveas}_mean']    = np.nan
                        else:
                            features[f'{parameter.saveas}_mean'] = segment.mean()
                            
                            # Save std
                            if parameter.std is True:
                                if datapoints < 0.5:
                                    features[f'{parameter.saveas}_std'] = np.nan
                                else:
                                    features[f'{parameter.saveas}_std'] = segment.std()
                                
                            # Save trend
                            if parameter.trend is True:
                                if datapoints < 0.5:
                                    features[f'{parameter.saveas}_trend'] = np.nan
                                else:
                                    segment = segment.dropna()
                                    x = (segment.index - segment.index[0]).total_seconds()
                                    y = segment.values
                                    slope, _, _, _, _ = linregress(x,y)
                                    features[f'{parameter.saveas}_trend'] = slope
                        
                    else:                   # Save mean value for multiple feature windows

                        # Calculate feature window length
                        interval = observation_window/feature_windows
                        
                        # Save mean for each featue window
                        for i in range(feature_windows):
                            
                            segment     = data[start_window+pd.Timedelta(hours=i*interval):start_window+pd.Timedelta(hours=(i+1)*interval)]
                            datapoints  = segment.notna().sum()/len(segment)
                            
                            if datapoints < 0.5:
                                features[f'{parameter.saveas}_mean_{i}'] = np.nan
                            else:
                                features[f'{parameter.saveas}_mean_{i}'] = segment.mean()

            except Exception as e:
                    print('Error:', e, 'in patient:', patient_id, event_time, parameter.saveas)
                    
                    continue
                    
        features_per_event.setdefault(idx, {}).update(features)     # Add aggregate feature values to sample in feature dictionary
        
    features_df = pd.DataFrame(features_per_event).transpose()      # Convert to DataFrame
        
    return features_df

def get_timeseries_features2(dataset, features_per_event, prediction_mode, observation_window, gap_window, feature_windows, feature_selection): 
    '''
    Function to get aggregate features form timeseries data. Calculates mean from 1 hour before & 1 hour after switch, and relative & absolute change.

    Parameters
    ----------
    dataset : str
        Name of dataset and folder with patient files.
    features_per_event : dict
        Dictionary with tabular features per event.
    prediction_mode : str
        Ventilation mode during predicitions.
    observation_window : int
        Observation window length in hours.
    gap_window : int
        Horizon length in hours.
    feature_windows : int
        Number of windows within observation window to calculate aggregate features from.
    feature_selection : Bool
        Wheter to calculate only the selected features.

    Returns
    -------
    features_df : DataFrame
        DataFrame with input feature values per sample.

    '''
    parameters_df = pd.read_csv('P:/Emmelieve/feature_variables.csv', delimiter=';')
    if feature_selection is True:
        parameters_df = parameters_df[parameters_df[f'{prediction_mode}_selected'] == 1]
    else:
        parameters_df = parameters_df[parameters_df[prediction_mode] == 1]
    previous_patient_id = None
    
    for idx, event in features_per_event.items():
        features            = {}
        patient_id          = event['patient_id']
        event_time          = event['timestamp']
        switch              = event['switch']
        start_window        = switch - pd.Timedelta(hours=1)
        end_window          = switch + pd.Timedelta(hours=1)
        
        if patient_id != previous_patient_id:
            timeseries = pd.read_csv(f'P:/Emmelieve/{dataset}/{patient_id}.csv', index_col=0)
        previous_patient_id = patient_id
        timeseries.index.name = 'Timestamp'
        timeseries.index  = pd.to_datetime(timeseries.index, format="%Y-%m-%d %H:%M:%S")
        
        for parameter in parameters_df.itertuples(index=False):    # loop through input features to calculate

            try:
                # Check if variable is present in data
                keys    = [parameter.parameter_0, parameter.parameter_1, parameter.parameter_2, parameter.parameter_3]
                present = [key for key in keys if key in timeseries.columns]
                
                # Extract/compute column with parameter data
                # Use empty Series if parameter is not present
                if len(present) == 0:
                    timestamps  = pd.date_range(start=start_window, end=end_window, freq="1min")
                    data        = pd.Series(index=timestamps, name=parameter.saveas, dtype=float)
                    
                # Calculate IE, PF, RSBI or SF ratio
                elif parameter.saveas in ['ie_ratio', 'pf_ratio', 'rsbi', 'sf_ratio']:
                    data        = timeseries[parameter.parameter_0] / timeseries[parameter.parameter_1]
        
                # Calculate and save MV duration    
                elif parameter.saveas == 'mv_duration':
                    data        = timeseries[parameter.parameter_0][timeseries.index[0]:(start_window + pd.Timedelta(hours=observation_window))].notna()
                    features[parameter.saveas]  = data.sum()
                    
                # Merge columns with same parameter
                elif len(present) > 1:
                    data        = timeseries[present].bfill(axis=1).iloc[:,0]
                    timeseries[parameter.saveas] = data
                
                # Get parameter column
                else:
                    data = timeseries[present[0]]
                    timeseries[parameter.saveas] = data
                    
                # Calculate aggregate feature values
                # Set 0 if medication was not administred (pump rate = 0)
                if parameter.type == 'medication':
                    data = data.fillna(0)
                    
                # Save last parameter value as feature value
                if parameter.last is True:
                    features[parameter.saveas] = data.dropna().iloc[-1]
                
                # Save mean before, after and change
                if parameter.mean is True:
               
                    for i in [0,1]:
                        
                        segment = data[start_window+pd.Timedelta(hours=i):start_window+pd.Timedelta(hours=i+1)]
                        segment = segment.replace([np.inf, -np.inf], np.nan)
                        
                        datapoints = segment.notna().sum()/len(segment)
                        
                        if datapoints < 0.5:
                            features[f'{parameter.saveas}_mean_{i}']    = np.nan
                        else:
                            features[f'{parameter.saveas}_mean_{i}']    = segment.mean()
                            

                    before_value    = features[f'{parameter.saveas}_mean_0']
                    after_value     = features[f'{parameter.saveas}_mean_1']
                    
                    features[f'{parameter.saveas}_abs_change']          = (after_value - before_value)
                    
                    if before_value != 0:
                        features[f'{parameter.saveas}_rel_change']      = (after_value - before_value)/before_value
                    else:
                        features[f'{parameter.saveas}_rel_change']      = np.nan
                                                    
                        
            except Exception as e:
                    print('Error:', e, 'in patient:', patient_id, event_time, parameter.saveas)
                    
                    continue
                    
        features_per_event.setdefault(idx, {}).update(features)
        
    features_df = pd.DataFrame(features_per_event).transpose()
        
    return features_df