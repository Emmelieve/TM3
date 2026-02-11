# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:28:32 2025

@author: eldenbreejen
"""
import pandas as pd
from datetime import datetime
import os
from event_detection_functions_15 import get_events
from feature_input_functions_12 import get_feature_input_table
from model_training_functions_13 import classification_model_training, performance_per_iteration, backward_feature_selection, compare_models, logres_model_training, end_model_training, plot_learning_curve
from model_testing_functions_10 import model_testing, auroc_auprc, summary, calibration_curve, model_calibration, decision_curve, outcome_distribution_plot, outcome_distribution_plot1
from explainability_functions_10 import shap_summary, tree_visual, local_shaps
from timeline_functions_10 import create_timelines
 
# %% Main function for event detection, feature value calculation and model training

def main(dataset, observation_window, gap_window, prediction_window, feature_windows, prediction_mode, training=True, timelines=False, feature_selection=False, beta=False):
    '''
    Function to execute data labeling, input feature calculation and model training.

    Parameters
    ----------
    dataset : str
        Name of the dataset, corresponding to the folder with patient files to be used.
    observation_window : int
        Observation window length in hours.
    gap_window : int
        Gap window (prediction horizon) length in hours.
    prediction_window : int
        Prediction window length in hours.
    feature_windows : int
        Number of windows within the observation window to calculate summary statistics over as input features.
    prediction_mode : str
        Ventilation mode during prediction, eithter 'assisted' or 'controlled'
    training : bool, optional
        Whether to perform model training, can be set to False for validation datasets. The default is True.
    timelines: bool, optional
        Whether to create timeline files per record. The default is False.
    feature_selection: bool, optional
        Whether to only use the selected subset of features. The default is False.
    beta: bool, optional
        Wheter to train model 1B with input feature from before and after the switch. The default is False.

    Returns
    -------
    None.

    '''
    
    # Print date and window settings. Date is used in filename for feature input tables and cross-validation results.
    date = datetime.today().strftime('%Y-%m-%d_%H-%M')
    
    print(date)
    print('')
    print('observation_window:', observation_window)
    print('gap_window:', gap_window)
    print('prediction_window:', prediction_window)
    print('')
    
    # Data labelling
    # Check whether event file for current window settings exists and read it or create one.
    if beta is True:
        events_file = f'P:/Emmelieve/tabular_data_events_{dataset}/tabular_data_{observation_window}{gap_window}{prediction_window}_b.csv'
    else:
        events_file = f'P:/Emmelieve/tabular_data_events_{dataset}/tabular_data_{observation_window}{gap_window}{prediction_window}.csv'
        
    if os.path.exists(events_file):
        combined_data_inclusions                    = pd.read_csv(events_file, low_memory=False)
        datetime_columns                            = combined_data_inclusions.columns[combined_data_inclusions.columns.str.startswith("event")].tolist()
        combined_data_inclusions[datetime_columns]  = combined_data_inclusions[datetime_columns].apply(pd.to_datetime, format='mixed')
        timedelta_columns                           = combined_data_inclusions.columns[combined_data_inclusions.columns.str.startswith("duration")].tolist()
        combined_data_inclusions[timedelta_columns] = combined_data_inclusions[timedelta_columns].apply(pd.to_timedelta)
        
        print('Events read')
        print('')
        
    else:
        events_per_patient                          = get_events(dataset, observation_window, gap_window, prediction_window, prediction_mode, beta)
        
        # Count and print the nr of detected events
        nr_events = 0
        nr_non_events = 0
        
        for i in events_per_patient.filter(regex=r"^event_\d+").columns.tolist():
            events      = events_per_patient[i].notnull().sum()
            nr_events   += events
            
        for i in events_per_patient.filter(regex=r"^event_control").columns.tolist():
            events          = events_per_patient[i].notnull().sum()
            nr_non_events   += events
            
        print('Events:', nr_events)
        print('Non-Events:', nr_non_events)
        print('')
        
        # Combine events dataframe with tabular data
        tabular_data                 = pd.read_csv(f'P:/Emmelieve/tabular_data_{dataset}.csv')
        events_per_patient           = events_per_patient.drop(['admission'], axis=1)
        events_per_patient           = events_per_patient.rename(columns={'patientID': 'PatientID', 'demission': 'DemissionDate'})
        events_per_patient.PatientID = events_per_patient.PatientID.astype('int64')
        events_per_patient           = events_per_patient.drop(columns=['DemissionDate', 'death'])  # drop duplicate columns
        
        combined_data_inclusions     = tabular_data.merge(events_per_patient, on='PatientID', how='inner')
        combined_data_inclusions.to_csv(events_file, index=False)
        
        print('Events detected and saved')
        print('')
    
    # Compute feature input table
    get_feature_input_table(dataset, combined_data_inclusions, prediction_mode, observation_window, gap_window, date, feature_windows, feature_selection, beta)
    
    # Model training and cross-validation
    if training is True:
        classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=False)
    
    # Create timelines with ventilation mode, respiratory rate and event labels per record
    if timelines is True:
        create_timelines(dataset, combined_data_inclusions)
    


# %% MODEL 1: Unreadiness events
dataset         = 'train3'
prediction_mode = 'controlled'

# 1 OBSERVATION WINDOW
# Perform data labeling, feature generation, model training and cross-validation for different observation windows
for i in [1,2,4,6]:
    main(dataset, observation_window=i, feature_windows=1, prediction_mode=prediction_mode)

# Dates used as filename for feature input tables and training metrics    
# 1 h: '2026-01-13_08-58'
# 2 h: '2026-01-13_09-05'
# 4 h: '2026-01-13_09-10'
# 6 h: '2026-01-13_09-16'

# Repeat training and cross-validation
for date in ['2026-01-13_08-58', '2026-01-13_09-05', '2026-01-13_09-10','2026-01-13_09-16']:
    classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=False)
    
# Wilcoxon signed rank test to compare 1 h observation window to 2,4,6 h
date = '2026-01-13_08-58'

for alt_date in ['2026-01-13_09-05', '2026-01-13_09-10','2026-01-13_09-16']:
    p_auroc, p_auprc = compare_models(date, alt_date, prediction_mode, alternative='greater')

  
# 2 LOGISTIC REGRESSION MODEL
# Train logistic regression model and compare with XGBoost
date = '2026-01-13_08-58'
classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=False, notna=True)
logres_model_training(dataset, date, prediction_mode)

p_auroc, p_auprc = compare_models(f'{date}_notna', f'{date}_logres', prediction_mode)
    

# 3 FEATURE SELECTION
# 3.1 Estimate nr of boosting iterations
date = '2026-01-13_08-58'
params = {'eval_metric': 'auc', 'random_state': 36, 'n_estimators': 100}
performance_per_iteration(dataset, date, prediction_mode, params, step='init')
# ! set n_estimators to selected number in feature_selection() function

# 3.2 Generate feature importance plots and performance curve of importance based backward feature selection based for each feature set

# Set 1: mean, std, trend over 1 hour
date = '2026-01-13_08-58'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Set 2: mean over 20 min
main(dataset, 1, 0, 6, 3, prediction_mode=prediction_mode)

date = '2026-01-13_09-54'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Set 3: observation window 6, mean, std, trend over 6 h
date = '2026-01-13_09-16'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Set 4: observation window 6, mean per 2 h
main(dataset, 6, 0, 6, 3, prediction_mode=prediction_mode)

date = '2026-01-13_10-26'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Save cross-validation metrics with selected features
for date in ['2026-01-13_08-58', '2026-01-13_09-54', '2026-01-13_09-16', '2026-01-13_10-26']:
    classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=True)

# Compare perforamnces of different feature sets 1 vs 2,3,4
date = '2026-01-13_08-58_selected_features'

for alt_date in ['2026-01-13_09-54_selected_features', '2026-01-13_09-16_selected_features', '2026-01-13_10-26_selected_features']:
    p_auroc, p_auprc = compare_models(date, alt_date, prediction_mode, alternative='greater')
    
    
# 4 HYPERPARAMETER TUNING
# Perform manual, stepwise hyperparameter tuning in stepwise_hyperparameter_tuning_controlled.py

# 5 LEARNING CURVES

# Initial learning curve
date    = '2026-01-13_08-58'
title   = 'Initial learning curve'
plot_learning_curve(date, prediction_mode, feature_selection=False, optimized=False, n_estimators=100, title=title)

# Learning curve after reduction of boosting iterations
date    = '2026-01-13_08-58'
title   = 'Learning curve after reduction of boosting iterations'
plot_learning_curve(date, prediction_mode, feature_selection=False, optimized=False, n_estimators=5, title=title)

# Learning curve after feature selection
date    = '2026-01-13_08-58'
title   = 'Learning curve after feature selection'
plot_learning_curve(date, prediction_mode, feature_selection=True, optimized=False, n_estimators=5, title=title)

# Learning curve after hyperparamer tuning
date    = '2026-01-13_08-58'
title   = 'Learning curve after hyperparameter optimisation'
plot_learning_curve(date, prediction_mode, feature_selection=True, optimized=True, title=title)

# 6 LEAKAGE
# Compare cross-validation scores with and without grouping
classification_model_training(dataset, date, prediction_mode, grouped=False, optimized=True, feature_selection=True)
classification_model_training(dataset, date, prediction_mode, grouped=True, optimized=True, feature_selection=True)
p_auroc, p_auprc = compare_models(f'{date}_optimized', f'{date}_leakage', prediction_mode)

# 7 VALIDATION

# 7.1 Generate data labels and feature input tables for test sets
dataset = 'test3'       # COVID test set
main(dataset, 1, 0, 6, 1, 'controlled', training=False) # 2026-01-13_11-32

dataset = 'dataset4'    # non-COVID test set
main(dataset, 1, 0, 6, 1, 'controlled', training=False) # 2026-01-13_11-35

# ! Change feature_input_table names with 'controlled' in stead of the date for train3, test3 and dataset4

# 7.2 Perform cross-validation on non-covid test set
classification_model_training('dataset4', '2026-01-13_11-35', prediction_mode, grouped=True, optimized=True, feature_selection=True)

# 7.3 Train the final optimised model on all training data
X_train, y_train, y_pred, trained_model = end_model_training('train3', 'controlled')
outcome_distribution_plot1(y_train, y_pred, prediction_mode) # probability distribution training set

# 7.4 Get classification metrics for both test sets
X_test1, y_test1, y_pred1, X_test2, y_test2, y_pred2 = model_testing(trained_model,'test3','dataset4','controlled')
threshold1, threshold2  = auroc_auprc(y_test1, y_pred1, y_test2, y_pred2, 0.80, 'specificity', 'controlled')

summary(y_test1, y_pred1, threshold1, prediction_mode)
summary(y_test2, y_pred2, threshold2, prediction_mode)


# 7.5 Decsision curve analysis
# Calibrate model
y_pred_cal  = model_calibration(trained_model, X_train, y_train)
y_pred1_cal = model_calibration(trained_model, X_test1, y_test1)
y_pred2_cal = model_calibration(trained_model, X_test2, y_test2)

# Generate calibration curves for both test sets and train set
calibration_curve(y_train, y_pred, y_test1, y_pred1, y_test2, y_pred2, prediction_mode)
calibration_curve(y_train, y_pred_cal, y_test1, y_pred1_cal, y_test2, y_pred2_cal, prediction_mode, calibrated=True)

# Net-benefit analysis
decision_curve(y_test1, y_pred1_cal, y_test2, y_pred2_cal, prediction_mode)

# Distribution plots
outcome_distribution_plot(y_test1, y_pred1, y_test2, y_pred2, prediction_mode)
outcome_distribution_plot(y_test1, y_pred1_cal, y_test2, y_pred2_cal, prediction_mode, xlim=[0,1], name='calibrated')



# 8 EXPLAINABILITY
# Generate SHAP summary plot
shap_summary(trained_model, X_train, prediction_mode)

# Generate local shap explanation plot for events and controls with highest & lowest prediction scores
local_shaps(trained_model, X_train, y_train, y_pred, prediction_mode)

# Generate tree visualisation of first tree in XGBoost ensemble
tree_visual(trained_model, X_train, y_train, trees=[0], mode=prediction_mode)

# Create timelines with control and event labels
tabular_data_events = pd.read_csv(f'P:/Emmelieve/tabular_data_events_{dataset}/tabular_data_106.csv')
create_timelines(dataset, tabular_data_events)

# %% MODEL 1B: Unreadiness events with before and after switch input

dataset         = 'train3'
prediction_mode = 'controlled'

# 1 FEEATURE SELECTION
# Perform data labeling, feature generation, model training and cross-validation for different observation windows
main(dataset, 1,0,6,1,prediction_mode=prediction_mode, beta=True) #2026-01-20_15-26

date = '2026-01-20_15-26'
backward_feature_selection(dataset, date, prediction_mode, nr_features=20)

# Get performance after feature selection
classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=True, beta=True)

# 2 LEARNING CURVES
# Learning curve after feature selection
date    = '2026-01-20_15-26'
title   = 'Learning curve after feature selection'
plot_learning_curve(date, prediction_mode, feature_selection=True, optimized=False, n_estimators=5, title=title)

# Learning curve after hyperparamer tuning
date    = '2026-01-20_15-26'
title   = 'Learning curve after hyperparameter optimisation'
plot_learning_curve(date, prediction_mode, feature_selection=True, optimized=True, title=title, beta=True)

# Performance after hyperparamer tuning
classification_model_training(dataset, date, prediction_mode, optimized=True, feature_selection=True, beta=True)

# 3 VALIDATION

# 3.1 Generate data labels and feature input tables for test sets
dataset = 'test3'       # COVID test set
main(dataset, 1, 0, 6, 1, 'controlled', training=False, beta=True) # 2026-01-20_16-55

dataset = 'dataset4'    # non-COVID test set
main(dataset, 1, 0, 6, 1, 'controlled', training=False, beta=True) # 2026-01-20_16-58

# ! Change feature_input_table names with 'controlled' in stead of the date for train3, test2 and dataset4

# 3.2 Train the final optimised model on all training data
prediction_mode = 'controlled_1B'
X_train, y_train, y_pred, trained_model = end_model_training('train3', prediction_mode)

# 3.3 Get classification metrics for both test sets

X_test1, y_test1, y_pred1, X_test2, y_test2, y_pred2 = model_testing(trained_model,'test3','dataset4',prediction_mode)
threshold1, threshold2  = auroc_auprc(y_test1, y_pred1, y_test2, y_pred2, 0.80, 'specificity', prediction_mode)

summary(y_test1, y_pred1, threshold1, prediction_mode)
summary(y_test2, y_pred2, threshold2, prediction_mode)


# 4 EXPLAINABILITY
# Generate SHAP summary plot
shap_summary(trained_model, X_train, prediction_mode)


# %% MODEL 2: P-SILI events
dataset         = 'train3'
prediction_mode = 'assisted'

# 1 OBSERVATION WINDOW
# Perform data labeling, feature generation, model training and cross-validation for different observation windows
for i in [1,2,4,6]:
    main(dataset, observation_window=i, gap_window=6, prediction_window=2, feature_windows=1, prediction_mode=prediction_mode)

# 1 h: '2026-01-14_09-34' 
# 2 h: '2026-01-14_09-08' 
# 4 h: '2026-01-14_08-42'  
# 6 h: '2026-01-14_08-16'

# Repeat cross-validation
for date in ['2026-01-14_09-34', '2026-01-14_09-08', '2026-01-14_08-42', '2026-01-14_08-16']:
    classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=False)
   
# Compare model performances with Wilcoxon signed rank test. 4 h vs 1,2,6 h.
date = '2026-01-14_08-42'
for alt_date in ['2026-01-14_09-34', '2026-01-14_09-08', '2026-01-14_08-16']:
    p_auroc, p_auprc = compare_models(date, alt_date, prediction_mode)

# 2 GAP WINDOW (PREDICTION HORIZON)
# Perform data labeling, feature generation, model training and cross-validation for different gap windows
for i in [4, 8, 10]:
    main(dataset, observation_window=4, gap_window=i, prediction_window=2, feature_windows=1, prediction_mode=prediction_mode)
    
# 4 h:  '2026-02-03_16-13'
# 6 h:  '2026-01-14_08-42'
# 8 h:  '2026-02-03_16-43'
# 10 h: '2026-02-03_17-10'

# Repeat cross-validation
for date in ['2026-02-03_16-13', '2026-01-14_08-42', '2026-02-03_16-43', '2026-02-03_17-10']:
    classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=False)

# Compare model performances with Wilcoxon signed rank test. 6 h vs 4,8,10 h.
date = '2026-01-14_08-42'

for alt_date in ['2026-02-03_16-13', '2026-02-03_16-43', '2026-02-03_17-10']:
    p_auroc, p_auprc = compare_models(date, alt_date, prediction_mode)
    
# 3 LOGISTIC REGRESSION MODEL
# Train logistic regression model and compare with XGBoost
date = '2026-01-14_08-42'
classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=False, notna=True)
logres_model_training(dataset, date, prediction_mode)
p_auroc, p_auprc = compare_models(f'{date}_logres', f'{date}_notna', prediction_mode)  


# 4 FEATURE SELECTION
# 4.1 Estimate nr of boosting iterations
date    = '2026-01-14_08-42'
params  = {'eval_metric': 'aucpr', 'random_state': 36, 'n_estimators': 100}
performance_per_iteration(dataset, date, prediction_mode, params, step='init')
# ! set n_estimators to selected number in feature_selection() function

# 4.2 Generate feature importance plots and performance curve of importance based backward feature selection based for each feature set

# Set 1: observation window 2 h: mean, std, trend
date = '2026-01-14_09-08' 
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Set 2: observation window 2 h: mean over 30 min
main(dataset, observation_window=2, gap_window=6, prediction_window=2, feature_windows=4, prediction_mode=prediction_mode)

date = '2026-02-05_09-52'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Set 3: observatin window 4 h: mean, std, trend
date = '2026-01-14_08-42'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Set 4: observation window 4 h: mean over 1 h
main(dataset, observation_window=4, gap_window=6, prediction_window=2, feature_windows=4, prediction_mode=prediction_mode)

date = '2026-02-05_10-15'
backward_feature_selection(dataset, date, prediction_mode, nr_features=25)

# Save cross-validation metrics with selected features
for date in ['2026-01-14_09-08' , '2026-02-05_09-52', '2026-01-14_08-42', '2026-02-05_10-15']:
    classification_model_training(dataset, date, prediction_mode, optimized=False, feature_selection=True)
    
# Compare performances of different feature sets 2 vs 1 and 3
date    = '2026-02-05_09-52_selected_features'

for alt_date in ['2026-01-14_09-08_selected_features', '2026-01-14_08-42_selected_features', '2026-02-05_10-15_selected_features']:
    p_auroc, p_auprc = compare_models(date, alt_date, prediction_mode, alternative='greater')


# 5 HYPERPARAMETER TUNING
# Go to stepwise_hyperparameter_tuning_assisted.py

# 6 LEARNING CURVES
# Initial learning curve
date    = '2026-01-14_08-42'
title   = 'Initial learning curve'
plot_learning_curve(date, prediction_mode, feature_selection=False, optimized=False, n_estimators=100, title=title)

# After reduction of boosting iterations
date    = '2026-01-14_08-42'
title   = 'Learning curve after reduction of boosting iterations'
plot_learning_curve(date, prediction_mode, feature_selection=False, optimized=False, n_estimators=5, title=title)

# After feature selection
date    = '2026-02-05_09-52'
title   = 'Learning curve after feature selection'
plot_learning_curve(date, prediction_mode, feature_selection=True, optimized=False, n_estimators=5, title=title)

# After hyperparameter tuning
date    = '2026-02-05_09-52'
title   = 'Learning curve after hyperparameter optimisation'
plot_learning_curve(date, prediction_mode, feature_selection=True, optimized=True, title=title)

# 7 LEAKAGE
# Compare cross-validation with and without grouping
date    = '2026-02-05_09-52'
classification_model_training(dataset, date, prediction_mode, grouped=False, optimized=True, feature_selection=True)
classification_model_training(dataset, date, prediction_mode, grouped=True, optimized=True, feature_selection=True)
p_auroc, p_auprc = compare_models(f'{date}_optimized', f'{date}_leakage', prediction_mode)

# 8 VALIDATION

# 8.1 Generate data labels and feature input tables for test sets
dataset = 'test3'
main(dataset, 2, 6, 2, 4, prediction_mode, training=False, feature_selection=True)      # 2026-02-05_11-38

dataset = 'dataset4'
main(dataset, 2, 6, 2, 4, prediction_mode, training=False, feature_selection=True)      # 2026-02-05_11-40

# ! Change feature_input_table names with 'assisted' in stead of the date for train3, test3 and dataset4

# 8.2 Train the final optimised model on all training data
X_train, y_train, y_pred, trained_model = end_model_training('train3', prediction_mode, us=True)
outcome_distribution_plot1(y_train, y_pred, prediction_mode) # probability distribution training set

# 8.3 Get classification metrics for both test sets
X_test1, y_test1, y_pred1, X_test2, y_test2, y_pred2 = model_testing(trained_model,'test3','dataset4',prediction_mode)
threshold1, threshold2  = auroc_auprc(y_test1, y_pred1, y_test2, y_pred2, 0.80, 'PPV', prediction_mode)

summary(y_test1, y_pred1, threshold1, prediction_mode)
summary(y_test2, y_pred2, threshold2, prediction_mode)

# 8.4 Decsision curve analysis
# Generate calibration curves for both test sets

# Calibrate model
y_pred_cal  = model_calibration(trained_model, X_train, y_train)
y_pred1_cal = model_calibration(trained_model, X_test1, y_test1)
y_pred2_cal = model_calibration(trained_model, X_test2, y_test2)

# Calibration curve
calibration_curve(y_train, y_pred, y_test1, y_pred1, y_test2, y_pred2, prediction_mode)
calibration_curve(y_train, y_pred_cal, y_test1, y_pred1_cal, y_test2, y_pred2_cal, prediction_mode, calibrated=True)

# Net-benefit
decision_curve(y_test1, y_pred1_cal, y_test2, y_pred2_cal, prediction_mode)

# 8.5 Distribution plot
outcome_distribution_plot(y_test1, y_pred1, y_test2, y_pred2, prediction_mode)
outcome_distribution_plot(y_test1, y_pred1_cal, y_test2, y_pred2_cal, prediction_mode, name='calibrated')

# 9 EXPLAINABILITY
# Generate SHAP summary plot
shap_summary(trained_model, X_train, prediction_mode)

# Generate local shap explanation plot for events and controls with a high and low predicted probability
local_shaps(trained_model, X_train, y_train, y_pred, prediction_mode)

# Generate tree visualisation of the first tree in XGBoost ensemble
tree_visual(trained_model, X_train, y_train, trees=[0], mode=prediction_mode)
