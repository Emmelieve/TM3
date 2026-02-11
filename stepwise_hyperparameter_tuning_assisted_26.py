# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:03:51 2025

@author: eldenbreejen
"""

# iterative hyperparameter tuning 
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import matplotlib.pyplot as plt
from model_training_functions_13 import performance_per_iteration, prepare_inputs

palette = {'blue': '#2D69FF',
           'blue_light': '#E1F0FF',
           'red': '#FF694B',
           'red_light': '#FEE1DB',
           'green': '#35B179',
           'green_light': '#DDFFEF',
           'navy': '#070453'}

dataset     = 'train3'
date        = '2026-02-05_09-52'
prediction_mode = 'assisted'
n_splits    = 10


# Get model inputs
X, y, groups    = prepare_inputs(dataset, date, prediction_mode, feature_selection=True, logres=False)
cv          = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=36)
 
# step 1: n estimators

params      = {
    'eval_metric': 'aucpr',
    'random_state': 36,
    'n_estimators': 100
    }

performance_per_iteration(dataset, date, prediction_mode, params, step='1', feature_selection=True)


# step 2: tree specific parameters

params['n_estimators'] = 15

param_grid = {
    'max_depth': [3,5,7],
    'min_child_weight': [1,3,5]
    }

scoring = {'AUROC': 'roc_auc',
           'AUPRC': 'average_precision'}

model = xgb.XGBClassifier(**params)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='AUPRC', cv=cv, n_jobs=1, verbose=1, return_train_score = True)
grid_search.fit(X, y, groups=groups)

results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_max_depth', 'param_min_child_weight', 'mean_test_AUPRC', 'std_test_AUPRC', 'mean_train_AUPRC', 'std_train_AUPRC']]
results.to_csv(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/hyperparameter_tuning_step2.csv')

# step 3: tune subsampling parameters

params['max_depth'] = 3
params['min_child_weight'] = 5

param_grid = {
        'subsample': [0.6, 0.8, 1.0],           # default = 1 (lower prevents overfitting)
        'colsample_bytree': [0.6, 0.8, 1.0]    # default = 1 (lower prevents overfitting)
        }
    
model = xgb.XGBClassifier(**params)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='AUPRC', cv=cv, n_jobs=1, verbose=1, return_train_score = True)
grid_search.fit(X, y, groups=groups)

results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_subsample', 'param_colsample_bytree', 'mean_test_AUPRC', 'std_test_AUPRC', 'mean_train_AUPRC', 'std_train_AUPRC']]
results.to_csv(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/hyperparameter_tuning_step3.csv')

# step 4: Fine-tuning regularization parameters

params['subsample'] = 0.6
params['colsample_bytree'] = 0.6

param_grid = {
    'alpha': [0, 0.01, 0.1, 1, 100],
    'lambda': [0, 0.01, 0.1, 1, 100]
    }

model = xgb.XGBClassifier(**params)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='AUPRC', cv=cv, n_jobs=1, verbose=1, return_train_score = True)
grid_search.fit(X, y, groups=groups)

results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_alpha', 'param_lambda', 'mean_test_AUPRC', 'std_test_AUPRC', 'mean_train_AUPRC', 'std_train_AUPRC']]
results.to_csv(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/hyperparameter_tuning_step4.csv')

# step 5: lower  learning rate & re-run step 1
params['alpha'] = 0.01
params['lambda'] = 1

params['learning_rate'] = 0.1
params['n_estimators'] = 100

performance_per_iteration(dataset, date, prediction_mode, params, step='2')

# conclusion
params ={'eval_metric': 'aucpr',
 'random_state': 36,
 'n_estimators': 20,
 'max_depth': 3,
 'min_child_weight': 5,
 'subsample': 0.6,
 'colsample_bytree': 0.6,
 'alpha': 0.01,
 'lambda': 1,
 'learning_rate': 0.1}