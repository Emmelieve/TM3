# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:05:23 2025

@author: eldenbreejen
"""
import numpy as np
import pandas as pd
import os
import warnings

def get_events(dataset, observation_window, gap_window, prediction_window, prediction_mode, fio2_threshold=40, beta=False):
    '''
    Function to detect event and control samples.

    Parameters
    ----------
    dataset : str
        Name of the dataset folder where patient files are stored.
    observation_window : int
        Observation window length in hours.
    gap_window : int
        Horizon window length in hours.
    prediction_window : int
        Prediction window length in hours.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.
    fio2_threshold : int, optional
        Minimum FiO2 in hour before and after event. The default is 40.
    beta : bool, optional
        Indicating wheter events are detected for model 1B. If True events wihtin 1 hour after switch are excluded. The default is False.

    Returns
    -------
    dataset_overview : DataFrame
        Overview with events and controls per record, merged with tabular data.

    '''
    
    # Get patient file names and initiatie dataset overview list
    data_files          = os.listdir(f'P:/Emmelieve/{dataset}')
    dataset_overview    = []
    
    # Get tabular data and tracheostomy insertion times for the right dataset
    if dataset == 'dataset4':
        tabular_data    = pd.read_csv(f'P:/Emmelieve/tabular_data_{dataset}.csv', delimiter=',', index_col='PatientID')
        trach_times     = pd.read_csv(f'P:/Emmelieve/trach_start_times_{dataset}.csv')
    
    elif (dataset == 'train3') or (dataset == 'test3'):
        tabular_data    = pd.read_csv('P:/Emmelieve/tabular_data_dataset3.csv', delimiter=';', index_col='PatientID')
        trach_times     = pd.read_csv('P:/Emmelieve/trach_start_times_dataset3.csv')
        
    trach_times['StartTime']  = pd.to_datetime(trach_times['StartTime'])
        
    # Get events and controles for each patient file
    for file in data_files:
        
        patient_id      = file[:-4]     # get patient ID
        
        try:
        
            # Read patient file
            data            = pd.read_csv(f'P:/Emmelieve/{dataset}/{file}', index_col=0)
            data.index.name = 'Timestamp'
            data.index      = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
            
            # Get tracheostomy time if applicable
            try:
                trach_time      = trach_times.loc[trach_times['PatientID'] == int(patient_id), 'StartTime'][0]
            except:
                trach_time = np.nan
            
            # Get start and end time of ICU measurements (based on heart rate) and limit datapoints to this window 
            heart_rate      = data['Hartfrequentie'].dropna()
            admission_time  = heart_rate.index[0]
            demission_time  = heart_rate.index[-1]
            data            = data.loc[admission_time:demission_time]
            
            # Create dictionary with patient ID, admission, demission and death datetimes
            patient_overview = {
                'patientID': patient_id,
                'admission': pd.Timestamp(tabular_data.loc[int(patient_id),'AddmissionDate']),
                'demission': pd.Timestamp(tabular_data.loc[int(patient_id),'DemissionDate']),
                'death':     pd.Timestamp(tabular_data.loc[int(patient_id),'death'])
                }
            
            # Get time periods with ventilation by Hamilton ventilator
            hamilton        = data['Adem minuutvolume (contr) Hamilton C6'].notna()
            
            # Get ventilation modes: 'assisted' or 'controlled'
            ventilation_mode_df = get_ventilation_mode(data['Ademfrequentie spontaan (contr)'], hamilton)
    
            # Get FiO2 settings over time
            fio2             = data['FiO2 (inst)']
            
            # Get events and controls for late/psili events or early/unreadiness events and add to patient_overview dict
            if prediction_mode      == 'assisted':
                
                patient_overview        = psili_event_detection(ventilation_mode_df, fio2, patient_overview, trach_time, observation_window, gap_window, prediction_window, fio2_threshold)
            
            elif prediction_mode    == 'controlled':
                
                if beta is False:
                    patient_overview    = unready_event_detection(ventilation_mode_df, fio2, patient_overview, trach_time, observation_window, fio2_threshold)
                else:
                    patient_overview    = unready2_event_detection(ventilation_mode_df, fio2, patient_overview, trach_time, observation_window, fio2_threshold)
                
     
            # Add patient overview with events to to dataset overview
            dataset_overview.append(patient_overview)
            
            
        except Exception as e:
            print('Error:', e, 'in patient:', patient_id)
            continue
    
    # Convert dataset overview to DataFrame         
    dataset_overview = pd.DataFrame(dataset_overview) 
    
    return dataset_overview
    

def get_ventilation_mode(rr_spontaneous, hamilton):
    '''
    Function to obtain ventilation modes (assisted or controlled) based on spontaneous respiratory rate.
    Periods without Hamilton ventilation are removed.

    Parameters
    ----------
    rr_spontaneous : Series
        Spontaneous respiratory rate per minute.
    hamilton : Series
        Presence of Hamilton ventilation per minute.

    Returns
    -------
    ventilation_mode_df : DataFrame
        Table with ventilation modes per segment with start and end time.

    '''
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Get ventilation mode from spontaneous respiratory rate (RR)
    ventilation_mode = rr_spontaneous.where(rr_spontaneous.isna(), rr_spontaneous > 5)            # Make shift at RR > 5
    ventilation_mode = ventilation_mode.where(hamilton, np.nan)                                   # Remove segments without Hamilton
    ventilation_mode = adaptive_median(ventilation_mode)                                          # Apply adaptive median filter
    ventilation_mode = ventilation_mode.fillna('NaN')
    
    # Create segments with start and end time
    groups              = (ventilation_mode != ventilation_mode.shift()).cumsum()                           # Increase group number when mode changes
    segments            = ventilation_mode.groupby(groups).apply(lambda g: (g.index[0], g.index[-1] + pd.Timedelta(minutes=1), g.iloc[0]))  # Get start time, end time, and ventilation mode per group
    ventilation_mode_df             = pd.DataFrame(segments.tolist(), columns=['Start', 'End', 'Label'])    # Convert to DataFrame
    ventilation_mode_df['Label']    = ventilation_mode_df['Label'].map({
        True: 'assisted', 
        False: 'controlled',
        'NaN': 'NaN'})
    
    ventilation_mode_df = ventilation_mode_df.copy()
    
    # Merge segments with missing data shorter than 1 hour with previous ventilation mode
    # Get missing data segment <= 1 h
    mask = ((ventilation_mode_df['Label'] == 'NaN')
            & ((ventilation_mode_df['End'] - ventilation_mode_df['Start']) <= pd.Timedelta(hours=1)))
    
    # Change labels
    if mask.sum() > 0:
        for i in mask[mask].index:
            if i != 0:      # skip first segment
                ventilation_mode_df.loc[i,'Label'] = ventilation_mode_df.loc[i-1,'Label']
     
    # Merge consecutive segments with identical labels
    new_block = (ventilation_mode_df['Label'] != ventilation_mode_df['Label'].shift()) | (ventilation_mode_df['Start'] != ventilation_mode_df['End'].shift())
    group = new_block.cumsum()
    ventilation_mode_df = ventilation_mode_df.groupby([group, 'Label'], as_index=False).agg({
        "Start": "min",
        "End": "max"
    })
    
    # Add Value and Parameter columns for timeline function
    ventilation_mode_df['Value'] = pd.Series(dtype=float)  # lege kolom met naam 'Value'
    ventilation_mode_df['Parameter'] = 'Ventilation mode' 
    
    # Add previous and next columns for event detection functions
    ventilation_mode_df['prev'] = ventilation_mode_df['Label'].shift(1)
    ventilation_mode_df['next'] = ventilation_mode_df['Label'].shift(-1)
    
    return ventilation_mode_df

def adaptive_median(series, base_window=15, min_periods=3, max_expand=10):
    '''
    Function to apply an adaptive median filter. Calculutes the rolling median first, and then increases the window if median is 0.5. 

    Parameters
    ----------
    series : Series
        Series with booleans per minute, True for RR > 5 and False for RR < 5.
    base_window : int, optional
        Base window length in minutes, rolling window = base window * 2 + 1. The default is 15.
    min_periods : int, optional
        Minimum observations in rolling window. The default is 3.
    max_expand : int, optional
        Maximum observation to add to rolling window. The default is 10.

    Returns
    -------
    medians : Series
        Median filtered ventilation mode per minute.

    '''
    
    # Apply standard rolling window median filter
    medians     = series.rolling(2*base_window + 1, center=True, min_periods=min_periods).median()

    # Identify indices with median == 0.5
    mask_half   = medians == 0.5
    half_idx    = np.where(mask_half)[0]

    # Recaculate 0.5 medians with expanded window
    for i in half_idx:
        expand  = 1
        med     = 0.5
        n   = len(series)

        while med == 0.5 and expand <= max_expand:
            left    = max(0, i - (base_window + expand))
            right   = min(n, i + (base_window + expand) + 1)
            window  = series.iloc[left:right].dropna()

            if len(window) < min_periods:
                break

            med     = np.median(window)
            expand  += 1

        # Set to 0 if median is still 0.5
        if med == 0.5:
            med = int(round(med))
        medians.iat[i] = med

    return medians

def psili_event_detection(ventilation_mode_df, fio2, patient_overview, trach_time, observation_window, gap_window, prediction_window, fio2_threshold=40):
    '''
    Function to label PSILI event samples and control samples.

    Parameters
    ----------
    ventilation_mode_df : DataFrame
        Table with ventilation mode segments with start and end times.
    fio2 : Series
        FiO2 settings per per minute.
    patient_overview : Dict
        Dictionary with patient ID, admission, demission, and death datetimes, events and control events will be added to this overview.
    trach_time : DateTime
        Datetime of tracheostomy, NaN if not applicable.
    observation_window : int 
        Observation window length in hours.
    gap_window : int
        Horizon length in hours.
    prediction_window : int
        Prediction window length in hours.
    fio2_threshold : int, optional
        Minimum FiO2 in hour before and after event. The default is 40.

    Returns
    -------
    patient_overview : Dict
        Patient overview supplemented with datetimes of events and controls.

    '''
    event_times = []
    
    # Get transistions from assisted to controlled
    mask = (
    (ventilation_mode_df['Label'] == 'assisted')
    & (ventilation_mode_df['next'] == 'controlled'))
        
    # Count nr of events and controls
    event_nr        = 0
    non_event_nr    = 0
    
    # Check if transistions are real events
    if mask.sum() > 0:
        for i in mask[mask].index:
            assisted_start  = ventilation_mode_df.loc[i,'Start']        # Start of assisted mode segment (switch)
            event_time      = ventilation_mode_df.loc[i,'End']          # Transistion from assisted to controlled
            control_end     = ventilation_mode_df.loc[i+1, 'End']       # End of controlled mode segment after event
            
            prediction_mode_length = event_time - assisted_start        # Length of assisted mode segment
            
            if prediction_mode_length >= pd.Timedelta(hours=(gap_window+observation_window)):   # Check if observation window fits within assisted mode segment
            
                if (((fio2[event_time - pd.Timedelta(hours=1) : event_time + pd.Timedelta(hours=1)] >= fio2_threshold).any())       # Check if FiO2 >= threshold
                    and (control_end - event_time >= pd.Timedelta(hours=3))                                                         # Check if controlle duration is >= 3 h
                    and ((pd.isna(trach_time)) or (trach_time < event_time-pd.Timedelta(hours=1)) or (trach_time > event_time+pd.Timedelta(hours=3)))):    # Check if no tracheostomy was performed around event                                                                                     # check line changes 
                            
                        # Get max nr of events that fit in prediction window (if prediction window == 2, max is 3 events at 0, 1, 2 h)
                        prediction_mode_length_res  = prediction_mode_length - pd.Timedelta(hours=observation_window+gap_window)     
                        nr_events                   = int(np.floor((prediction_mode_length_res.total_seconds() / 3600))) + 1                          
                        events                      = range(nr_events)
                        
                        if nr_events > prediction_window+1:         # Restrict to max nr of events witin prediction window
                            events = range(prediction_window+1)
                            
                        # Save event time, duration from switch, and prediction window for each event
                        for j in events:                       
                            patient_overview[f'event_{event_nr}']           = event_time
                            patient_overview[f'duration_event_{event_nr}']  = ventilation_mode_df.loc[i,'End'] - ventilation_mode_df.loc[i,'Start']
                            patient_overview[f'pred_event_{event_nr}']      = pd.Timedelta(hours=j)
                            event_times.append(event_time)
                            event_nr += 1
                            
    # Get control samples
    mask = (ventilation_mode_df['Label'] == 'assisted')
    
    if mask.sum() > 0:
        for i in mask[mask].index:
            segment_start   = ventilation_mode_df.loc[i,'Start']        # Start of assisted mode segment
            segment_end     = ventilation_mode_df.loc[i,'End']          # End of assisted mode segment
            prediction_mode_length = segment_end - segment_start - pd.Timedelta(hours=gap_window)   # Length of assisted mode segment min horizon
            
            # Only use data from before event if end of segment is true event
            if segment_end in event_times:
                prediction_mode_length = prediction_mode_length - pd.Timedelta(hours=prediction_window)

            # Check if observation window fits within prediciton mode length
            if prediction_mode_length > pd.Timedelta(hours=observation_window):
                
                try:
                    # Get end time of previous controlled segment (switch)
                    controlled_end  = ventilation_mode_df.loc[:i].query("Label == 'controlled'").iloc[-1]['End']
                    
                    # Get number of nr of control samples that fit within prediction mode lengh
                    nr_events       = int(np.floor((prediction_mode_length.total_seconds() / 3600)))
                    events          = range(nr_events)
                    
                    # Save timestamp, time from switch and prediction window for each control sample
                    for j in events:                       
                        event_time = segment_start + pd.Timedelta(hours=observation_window+gap_window) + pd.Timedelta(hours=j) 
                        
                        if (pd.isna(patient_overview['death'])) or (patient_overview['death'] - event_time > pd.Timedelta(hours=24)):   # Check if control sample is not within 24 before death
                            patient_overview[f'event_control_{non_event_nr}'] = event_time
                            patient_overview[f'duration_event_control_{non_event_nr}'] = event_time - controlled_end
                            patient_overview[f'pred_event_control_{non_event_nr}'] = pd.Timedelta(hours=0)
                            non_event_nr += 1
                            
                except Exception as e:
                    # print(e)
                    # No switch to assisted ventilation detected
                    continue
                    
    return patient_overview

def unready_event_detection(ventilation_mode_df, fio2, patient_overview, trach_time, observation_window, fio2_threshold):
    '''
    Function to label unreadiness event samples and control samples.

    Parameters
    ----------
    ventilation_mode_df : DataFrame
        Table with ventilation mode segments with start and end times.
    fio2 : Series
        FiO2 settings per per minute.
    patient_overview : Dict
        Dictionary with patient ID, admission, demission, and death datetimes, events and control events will be added to this overview.
    trach_time : DateTime
        Datetime of tracheostomy, NaN if not applicable.
    observation_window : int 
        Observation window length in hours.
    fio2_threshold : int, optional
        Minimum FiO2 in hour before and after event. The default is 40.

    Returns
    -------
    patient_overview : Dict
        Patient overview supplemented with datetimes of events and controls.

    '''

    # Get transistions from assisted to controlled
    mask = (
    (ventilation_mode_df['Label'] == 'assisted')
    & (ventilation_mode_df['prev'] == 'controlled')
    & (ventilation_mode_df['next'] == 'controlled'))
     
    # Count nr of events and controls
    event_nr = 0
    non_event_nr = 0
    
    # Check if transistions are real events, otherwise use them as control
    if mask.sum() > 0:
        for i in mask[mask].index:
            event_time              = ventilation_mode_df.loc[i,'End']      # Transistion from assisted to controlled
            control_end             = ventilation_mode_df.loc[i+1, 'End']   # End of controlled mode segment after event
            
            prediction_mode_length  = ventilation_mode_df.loc[i-1,'End'] - ventilation_mode_df.loc[i-1,'Start'] # Length of assisted ventilation segment
            event_duration          = ventilation_mode_df.loc[i,'End'] - ventilation_mode_df.loc[i,'Start']     # Time to failure after switch
            
            if (prediction_mode_length >= pd.Timedelta(hours=observation_window)) and (event_duration >= pd.Timedelta(minutes=15)):     # Check if observation window fits within prediction mode segment and if time to failure >= 15 min
                
                if (((fio2[event_time - pd.Timedelta(hours=1) : event_time + pd.Timedelta(hours=1)] >= fio2_threshold).any()) and       # Check if FiO2 is >= threshold
                    (control_end - event_time >= pd.Timedelta(hours=3)) and                                                             # Check if controlled mode after events >= 3 h
                    (event_duration < pd.Timedelta(hours=6)) and                                                                        # Check if event is within 6 h after switch        
                    ((pd.isna(trach_time)) or  (trach_time < event_time-pd.Timedelta(hours=1)) or (trach_time > event_time+pd.Timedelta(hours=3)))):    # Check if no tracheostomy was performed around event
                        
                    # Save event datetime and duration
                    patient_overview[f'event_{event_nr}']           = event_time
                    patient_overview[f'duration_event_{event_nr}']  = event_duration
                    patient_overview[f'pred_event_{event_nr}']      = event_duration
                    event_nr += 1
                                
                # If not an event, save as control sample    
                elif (pd.isna(patient_overview['death'])) or (patient_overview['death'] - event_time > pd.Timedelta(hours=24)):     # Check if control sample is not within 24 h before death
                    patient_overview[f'event_control_{non_event_nr}']           = event_time
                    patient_overview[f'duration_event_control_{non_event_nr}']  = event_duration
                    patient_overview[f'pred_event_control_{non_event_nr}']      = event_duration
                    non_event_nr += 1
                    
    # Save control samples followed by detubation
    mask = (
        (ventilation_mode_df['Label'] == 'assisted')
        & (ventilation_mode_df['prev'] == 'controlled')
        & (ventilation_mode_df['next'] == 'NaN'))
    
    if mask.sum() > 0:
        for i in mask[mask].index:
            
            event_time              = ventilation_mode_df.loc[i,'End']                                              # End of assisted ventilation segment
            prediction_mode_length  = ventilation_mode_df.loc[i-1,'End'] - ventilation_mode_df.loc[i-1,'Start']     # Controlled mode length
            spont_nan_duration      = ventilation_mode_df.loc[i+1, 'End'] - ventilation_mode_df.loc[i,'Start']      # Duration of assisted and nan segments combined
            
            if ((prediction_mode_length >= pd.Timedelta(hours=observation_window))                                  # Check if observation window fits within controlled segment
                and (spont_nan_duration >= pd.Timedelta(hours=6))                                                   # Check if assisted and nan duration is >= 6 h
                and ((pd.isna(patient_overview['death'])) or (patient_overview['death'] - event_time > pd.Timedelta(hours=24)))):   # Check if control sample is not within 24 h before death
                
                # Save control sample 
                patient_overview[f'event_control_{non_event_nr}']           = event_time
                patient_overview[f'duration_event_control_{non_event_nr}']  = ventilation_mode_df.loc[i,'End'] - ventilation_mode_df.loc[i,'Start']
                patient_overview[f'pred_event_control_{non_event_nr}']      = ventilation_mode_df.loc[i,'End'] - ventilation_mode_df.loc[i,'Start']
                non_event_nr += 1
                    
    return patient_overview

def unready2_event_detection(ventilation_mode_df, fio2, patient_overview, trach_time, observation_window, fio2_threshold):
    '''
    Function to label unreadiness event samples and control samples for model 1B, events within 1 h + 15 min after switch are excluded.

    Parameters
    ----------
    ventilation_mode_df : DataFrame
        Table with ventilation mode segments with start and end times.
    fio2 : Series
        FiO2 settings per per minute.
    patient_overview : Dict
        Dictionary with patient ID, admission, demission, and death datetimes, events and control events will be added to this overview.
    trach_time : DateTime
        Datetime of tracheostomy, NaN if not applicable.
    observation_window : int 
        Observation window length in hours.
    fio2_threshold : int, optional
        Minimum FiO2 in hour before and after event. The default is 40.

    Returns
    -------
    patient_overview : Dict
        Patient overview supplemented with datetimes of events and controls.

    '''

    # Events
    mask = (
    (ventilation_mode_df['Label'] == 'assisted')
    & (ventilation_mode_df['prev'] == 'controlled')
    & (ventilation_mode_df['next'] == 'controlled'))
        
    event_nr = 0
    non_event_nr = 0
    
    if mask.sum() > 0:
        for i in mask[mask].index:
            event_time  = ventilation_mode_df.loc[i,'End']
            switch_time = ventilation_mode_df.loc[i,'Start']
            control_end = ventilation_mode_df.loc[i+1, 'End']
            
            prediction_mode_length = ventilation_mode_df.loc[i-1,'End'] - ventilation_mode_df.loc[i-1,'Start']
            event_duration = ventilation_mode_df.loc[i,'End'] - ventilation_mode_df.loc[i,'Start']
            
            if (prediction_mode_length >= pd.Timedelta(hours=observation_window)) and (event_duration >= pd.Timedelta(hours=1, minutes=15)):
                
                if (((fio2[event_time - pd.Timedelta(hours=1) : event_time + pd.Timedelta(hours=1)] >= fio2_threshold).any()) and
                    (control_end - event_time >= pd.Timedelta(hours=3)) and
                    (event_duration < pd.Timedelta(hours=6)) and
                    ((pd.isna(trach_time)) or  (trach_time < event_time-pd.Timedelta(hours=1)) or (trach_time > event_time+pd.Timedelta(hours=3)))):
                        

                    patient_overview[f'event_{event_nr}']           = event_time
                    patient_overview[f'pred_event_{event_nr}']      = event_duration
                    patient_overview[f'switch_event_{event_nr}']    = switch_time
                    event_nr += 1
                                

                elif (pd.isna(patient_overview['death'])) or (patient_overview['death'] - event_time > pd.Timedelta(hours=24)):
                    patient_overview[f'event_control_{non_event_nr}']           = event_time
                    patient_overview[f'pred_event_control_{non_event_nr}']      = event_duration
                    patient_overview[f'switch_event_control_{non_event_nr}']    = switch_time
                    non_event_nr += 1
                    
    # Non-events at detubation
    mask = (
        (ventilation_mode_df['Label'] == 'assisted')
        & (ventilation_mode_df['prev'] == 'controlled')
        & (ventilation_mode_df['next'] == 'NaN'))
    
    if mask.sum() > 0:
        for i in mask[mask].index:
            event_time = ventilation_mode_df.loc[i,'End']
            switch_time = ventilation_mode_df.loc[i,'Start']
            
            prediction_mode_length = ventilation_mode_df.loc[i-1,'End'] - ventilation_mode_df.loc[i-1,'Start']
            spont_nan_duration = ventilation_mode_df.loc[i+1, 'End'] - ventilation_mode_df.loc[i,'Start']
            
            if ((prediction_mode_length >= pd.Timedelta(hours=observation_window))
                and (spont_nan_duration >= pd.Timedelta(hours=6))
                and ((pd.isna(patient_overview['death'])) or (patient_overview['death'] - event_time > pd.Timedelta(hours=24)))):
                
                patient_overview[f'event_control_{non_event_nr}']           = event_time
                patient_overview[f'pred_event_control_{non_event_nr}']      = ventilation_mode_df.loc[i,'End'] - ventilation_mode_df.loc[i,'Start']
                patient_overview[f'switch_event_control_{non_event_nr}']    = switch_time
                non_event_nr += 1
                    
    return patient_overview