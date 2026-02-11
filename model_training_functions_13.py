from sklearn.model_selection import cross_validate, train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Global variables

# Color palette for plots
palette = {'blue': '#2D69FF',
           'red': '#FF694B',
           'green': '#35B179',
           'navy': '#070453'}

# Optimised hyperparameter settings for XGBoost model 1

params_controlled = {'eval_metric': 'auc',
 'random_state': 36,
 'n_estimators': 20,
 'max_depth': 3,
 'min_child_weight': 1,
 'subsample': 1,
 'colsample_bytree': 0.6,
 'alpha': 0,
 'lambda': 1,
 'learning_rate': 0.05}


# Initial hyperparameter settings for XGBoost model 1
params_controlled_init =  {
            'eval_metric': 'auc',
            'random_state': 36
            }

# Optimised hyperparameter settings for XGBoost model 1B
params_controlled_beta = {'eval_metric': 'auc',
                         'random_state': 36,
                         'n_estimators': 25,
                         'max_depth': 3,
                         'min_child_weight': 1,
                         'subsample': 0.6,
                         'colsample_bytree': 0.8,
                         'alpha': 1,
                         'lambda': 0.1,
                         'learning_rate': 0.01}

# Optimised hyperparameter settings for XGBoost model 2
params_assisted = {'eval_metric': 'aucpr',
 'random_state': 36,
 'n_estimators': 20,
 'max_depth': 3,
 'min_child_weight': 5,
 'subsample': 0.6,
 'colsample_bytree': 0.6,
 'alpha': 0.01,
 'lambda': 1,
 'learning_rate': 0.1}

# Initial hyperparameter settings for XGBoost model 2
params_assisted_init = {
            'eval_metric': 'aucpr',
            'random_state': 36
    }

# 10-fold cross-validation iterator for stratified and grouped splits
cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=36)


def undersampling(data, prediction_mode):
    '''
    Function to perform undersampling of control samples to obtain 1:3 ratio for model 1 and 1:10 ratio for model 2.

    Parameters
    ----------
    data : DataFrame
        Dataframe with feature input values and event labels (columns) per sample (rows).
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    data : DataFrame
        Original DataFrame with less control samples (rows).

    '''

    # Define applicable ratio
    if prediction_mode == 'assisted':   
        k = 10
    if prediction_mode == 'controlled':
        k = 3
    
    # Count number of event and control samples
    nr_events   = (data.event == 1).sum()
    nr_controls = nr_events*k
    
    # Seperate events and controls
    event_data      = data[data.event == 1]
    control_data    = data[data.event == 0]
    
    # Sample controls
    control_data_sampled = control_data.sample(n=nr_controls, random_state=36)
    
    # Combine events and sampled controls in one dataframe    
    data = pd.concat([event_data, control_data_sampled])
    
    return data


def prepare_inputs(dataset, date, prediction_mode, feature_selection=False, logres=False):
    '''
    Function to get X (feature input), y (labels), groups (group numbers for cross-validation)

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    date : str
        Datetime of feature input table generation.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    feature_selection : bool, optional
        Whether to use only selected features or all features in the feature table. The default is False.
    logres : bool, optional
        Whether the model to be trained is a Logistic regression model. The default is False.
    
    Returns
    -------
    X : DataFrame
        Table with feature inputs (columns) per sample (rows).
    y : Series
        Event label per sample.
    groups : Series
        Group numbers for grouped cross-validation.

    '''
    # Read table with input feature values    
    data = pd.read_csv(f'P:/Emmelieve/input_feature_tables_{dataset}/input_feature_values_{date}.csv')
    
    # Remove infinity values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns and rows with missing values for logistic regression model
    if logres is True:
        print(data.shape)
        data = data.dropna(thresh=data.shape[1] - 3)             
        print(data.shape)
        print(data.columns[data.isna().any()])
        data = data.dropna(axis=1) 
        print(data.shape)
        
    
    # Apply undersampling
    data = undersampling(data, prediction_mode) 
    
    # Get model input features
    if feature_selection is True:
        selected_features   = pd.read_csv(f'P:/Emmelieve/input_feature_tables_{dataset}/selected_features_{date}.csv', header=None)[0].to_list()
        X                   = data[selected_features].copy()
    else:
        X                   = data.drop(columns=['hospital_number', 'patient_id', 'event','timestamp', 'prediction_window', 'switch'], errors='ignore')
    
    # Scale feature values for logistic regression mode
    if logres is True:
        scaler  = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    # Get event labels
    y = data.event
    y = y.astype(int)
    
    # Get group numbers
    groups = data.hospital_number
    
    return X, y, groups


def classification_model_training(dataset, date, prediction_mode, grouped=True, optimized=False, feature_selection=False, beta=False, notna=False):
    '''
    Function to train and test an XGBoost model using cross-validation.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    date : str
        Datetime of feature input table generation.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    grouped : bool, optional
        Whether to perfrom grouped cross-validation or not. The default is True.
    optimized : bool, optional
        Wheter to use optimised hyperparameter settings or not. The default is False.
    feature_selection : bool, optional
        Whether to use only selected features or all features in the feature table. The default is False.

    Returns
    -------
    None.

    '''
    
    # Get model inputs, labels and group numbers
    X, y, groups = prepare_inputs(dataset, date, prediction_mode, feature_selection, logres=notna)

    # Prepare results table    
    index       = list(range(10)) + ['mean', 'std', 'median', 'q1', 'q3']
    metrics     = pd.DataFrame(index=index)
    
    # Set scoring metrics
    scoring = {'AUROC': 'roc_auc',
               'AUPRC': 'average_precision'}
    
    # Get hyperparameters
    if prediction_mode=='controlled':
        if optimized is True:
            if beta is True:
                params = params_controlled_beta
            else:
                params = params_controlled
            
        else:
            params = params_controlled_init
            if feature_selection is True:
                params['n_estimators'] = 5
        
    elif prediction_mode == 'assisted':
        if optimized is True:
            params = params_assisted
        else:
            params = params_assisted_init
            if feature_selection is True:
                params['n_estimators'] = 5

    # Prepare model
    model = xgb.XGBClassifier(**params)
    
    # Perform grouped or ungrouped cross-validation
    if grouped is True:
        scores  = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring, return_train_score=False)
    else:
        cv2     = StratifiedKFold(10)
        scores  = cross_validate(model, X, y, cv=cv2, scoring=scoring, return_train_score=False)
    
    # Fill results table
    for metric in scoring.keys():
        score_list      = list(scores[f'test_{metric}'])
        mean_score      = [np.mean(score_list), np.std(score_list)]
        median_score    = [np.median(score_list), np.quantile(score_list, 0.25), np.quantile(score_list, 0.75)] 
        metrics[metric] = score_list + mean_score + median_score
        
    # Put nr of events and controls in metrics table    
    metrics['events']           = (y == 1).sum()
    metrics['control events']   = (y == 0).sum()
    
    # Save results
    if notna is True:
        metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}_notna.xlsx')
    if grouped is False:
        metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}_leakage.xlsx')
    elif optimized is True:
        metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}_optimized.xlsx')
    elif feature_selection is True:
        metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}_selected_features.xlsx')
    else:
        metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}.xlsx')
    
    # Print result
    print(metrics)
    print('')
    

def get_feature_importance(model, X, y, groups, dataset, mode, date, save=True, figure=True):
    
    feature_importance_df = pd.DataFrame(index=X.columns)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X,y, groups=groups)):
        X_train  = X.iloc[train_idx]
        y_train  = y.iloc[train_idx]
        
        model.fit(X_train, y_train)
        
        booster                     = model.get_booster()
        feature_importance          = booster.get_score(importance_type='gain')
        feature_importance_df[fold] = feature_importance
        
    # Get mean feature importance over 10 folds
    feature_importance_df           = feature_importance_df.fillna(0)
    feature_importance_df['mean']   = feature_importance_df.mean(axis=1)
    feature_importance_df           = feature_importance_df.sort_values(by='mean', ascending=False)
    if save == True:
        feature_importance_df.to_csv(f'P:/Emmelieve/output_{dataset}/{mode}/feature_importance_{date}.csv')
    
    # Generate feature importance plot and save figure
    if figure == True:
        fig, ax = plt.subplots(figsize=(10,10))
        feature_importance_df.iloc[0:25].plot.barh(y='mean', rot=0, color=palette['blue'], ax=ax, legend=False)
        ax.invert_yaxis()
        ax.grid(color='black', linewidth=0.5, linestyle=':')
        ax.set_title('XGBoost feature importance')
        ax.set_xlabel('Gain')
        ax.set_ylabel('Feature')
        plt.savefig(f'P:/Emmelieve/output_{dataset}/{mode}/figures/feature_importance_{date}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return feature_importance_df
        

def decisiontree_training(dataset, date, prediction_mode, feature_selection=False):
    '''
    Function to train and test a Decision Tree model using cross-validation.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    date : str
        Datetime of feature input table generation.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    feature_selection : bool, optional
        Whether to use only selected features or all features in the feature table. The default is False.

    Returns
    -------
    None.

    '''
    
    # Get model inputs
    X, y, groups = prepare_inputs(dataset, date, prediction_mode, feature_selection)

    # Prepare results table    
    index       = list(range(10)) + ['mean', 'std', 'median', 'q1', 'q3']
    metrics     = pd.DataFrame(index=index)
    
    # Set scoring metrics
    scoring     = {'AUROC': 'roc_auc',
                   'AUPRC': 'average_precision'}
    
    # Prepare model
    model       = DecisionTreeClassifier(random_state=36)
    
    # Perform grouped or ungrouped cross-validation
    scores      = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring, return_train_score=False)
    
    # Fill results table
    for metric in scoring.keys():
        score_list      = list(scores[f'test_{metric}'])
        mean_score      = [np.mean(score_list), np.std(score_list)]
        median_score    = [np.median(score_list), np.quantile(score_list, 0.25), np.quantile(score_list, 0.75)] 
        metrics[metric] = score_list + mean_score + median_score
        
    # Put nr of events and controls in metrics table    
    metrics['events']           = (y == 1).sum()
    metrics['control events']   = (y == 0).sum()
    
    # Save results
    metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}_decisiontree.xlsx')
    
    # Print result
    print(metrics)
    print('')
    

def logres_model_training(dataset, date, prediction_mode, feature_selection=False):
    '''
    Function to train and test a Logistic regression model using cross-validation.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    date : str
        Datetime of feature input table generation.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    feature_selection : bool, optional
        Whether to use only selected features or all features in the feature table. The default is False.

    Returns
    -------
    None.

    '''
    # Get model inputs
    X, y, groups = prepare_inputs(dataset, date, prediction_mode, feature_selection, logres=True)
    
    # Prepare results table    
    index       = list(range(10)) + ['mean', 'std', 'median', 'q1', 'q3']
    metrics     = pd.DataFrame(index=index)
    
    # Set scoring metrics
    scores      = {'AUROC': [],
                   'AUPRC': []}
    
    # Prepare model       
    model           = LogisticRegression()
    
    coefficients_df = pd.DataFrame(index=X.columns)
    
    # Grouped cross validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X,y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        coefficients_df[fold] = model.coef_[0]
        
        y_pred  = model.predict_proba(X_val)[:,1]
        auroc   = roc_auc_score(y_val, y_pred)
        auprc   = average_precision_score(y_val, y_pred)
        scores['AUROC'].append(auroc)
        scores['AUPRC'].append(auprc)
    
    # Calculate mean coefficient
    coefficients_df['mean']     = coefficients_df.mean(axis=1)
    coefficients_df['mean_abs'] = coefficients_df['mean'].abs()
    coefficients_df             = coefficients_df.sort_values(by='mean_abs', ascending=False)
    coefficients_df.to_csv(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/logres_coefficients_{date}.csv')
    
    # Generate feature importance plot and save figure
    fig, ax = plt.subplots(figsize=(10,10))
    coefficients_df.iloc[0:25].plot.barh(y='mean', rot=0, color=palette['blue'], ax=ax, legend=False)
    ax.invert_yaxis()
    ax.grid(color='black', linewidth=0.5, linestyle=':')
    ax.set_title('Logistic regression coefficients')
    ax.set_xlabel('Logistic regression coefficient')
    ax.set_ylabel('Feature')
    plt.savefig(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/figures/logres_coefficients_{date}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Fill results table
    for metric in scores.keys():
        score_list      = list(scores[metric])
        mean_score      = [np.mean(score_list), np.std(score_list)]
        median_score    = [np.median(score_list), np.quantile(score_list, 0.25), np.quantile(score_list, 0.75)] 
        metrics[metric] = score_list + mean_score + median_score
        
    # Put nr of events and controls in metrics table    
    metrics['events']           = (y == 1).sum()
    metrics['control events']   = (y == 0).sum()
    
    # Save results
    metrics.to_excel(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/metrics_{date}_logres.xlsx')
    
    # Print result
    print(metrics)
    print('')
    
  
def performance_per_iteration(dataset, date, prediction_mode, params, step, feature_selection=False):
    '''
    Function to generate a plot with the performance of the XGBoost model per iteration.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    date : str
        Datetime of feature input table generation.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    params : dict
        XGBoost hyperparameter settings.

    Returns
    -------
    None.

    '''
    
    # Get model inputs
    X, y, groups    = prepare_inputs(dataset, date, prediction_mode, feature_selection, logres=False)

    # Prepare scoring
    test_scores     = []
    train_scores    = []
    
    metric          = params['eval_metric']
    
    # Prepare model
    model           = xgb.XGBClassifier(**params, early_stopping_rounds = 50)

    # Fit XGBoost model with early stopping for ten-folds
    for train_index, test_index in cv.split(X,y,groups=groups):
        
        # Get model inputs for current fold
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        # Fit model
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_train_fold, y_train_fold),(X_test_fold, y_test_fold)], verbose=False)

        # Save train and test scores        
        test_scores.append(model.evals_result()['validation_1'][metric])
        train_scores.append(model.evals_result()['validation_0'][metric])

    # Get 5th, 25th, 50th, 75th, 95th percentiles for train and test scores
    p5_test  = []
    p25_test = []
    p50_test = []
    p75_test = []
    p95_test = []

    p5_train  = []
    p25_train = []
    p50_train = []
    p75_train = []
    p95_train = []
    
    # Get 10 scores for iteration 1 to 50
    for i in range(50):
        test_scores_i   = []
        train_scores_i  = []
        
        for j in range(10):
            test_scores_i.append(test_scores[j][i])
            train_scores_i.append(train_scores[j][i])
        
        p5_test.append(np.percentile(test_scores_i, 5))
        p25_test.append(np.percentile(test_scores_i, 25))
        p50_test.append(np.percentile(test_scores_i, 50))
        p75_test.append(np.percentile(test_scores_i, 75))
        p95_test.append(np.percentile(test_scores_i, 95))
        
        p5_train.append(np.percentile(train_scores_i, 5))
        p25_train.append(np.percentile(train_scores_i, 25))
        p50_train.append(np.percentile(train_scores_i, 50))
        p75_train.append(np.percentile(train_scores_i, 75))
        p95_train.append(np.percentile(train_scores_i, 95))

    # Generate figure with train and test performance scores per iteration
    x = range(1,50+1)
      
    plt.figure(figsize=(10,6))

    plt.plot(x, np.array(p50_train), label='Train score', color=palette['red'])
    plt.fill_between(x, np.array(p5_train), np.array(p95_train), label='_none', color=palette['red'], alpha=0.2)
    plt.fill_between(x, np.array(p25_train), np.array(p75_train), label='_none', color=palette['red'], alpha=0.2)

    plt.plot(x, np.array(p50_test), label='Validation score', color=palette['blue'])
    plt.fill_between(x, np.array(p5_test), np.array(p95_test), label='_none', color=palette['blue'], alpha=0.2)
    plt.fill_between(x, np.array(p25_test), np.array(p75_test), label='_none', color=palette['blue'], alpha=0.2)
    
    plt.title('Performance over iterations')
    plt.xlabel('Nr of iterations')
    
    if prediction_mode == 'assisted':
        plt.ylabel('AUPRC')
        plt.ylim(0,1.02)
    elif prediction_mode == 'controlled':
        plt.ylabel('AUROC')
        plt.ylim(0.4,1.02)
    
    plt.legend(loc='upper right')
    plt.grid(color='black', linewidth=0.5, linestyle=':')

    plt.savefig(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/figures/hyperparameter_tuning_nrtrees_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()

  
def feature_selection(dataset, date, prediction_mode, nr_features=20):
    '''
    Function to get feature importance scores from XGBoost model and perform importance based forward feature selection.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    date : str
        Datetime of feature input table generation.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    nr_features : int, optional
        The maximum number of features to be added during forward feature evaluation. The default is 20.

    Returns
    -------
    None.

    '''
    
    # Get model inputs
    X, y, groups    = prepare_inputs(dataset, date, prediction_mode)  

    # Get hyperparameter settings
    if (prediction_mode=='controlled'):
        params                 = params_controlled_init
        params['n_estimators'] = 5
        scoring                = 'roc_auc'
        
    elif (prediction_mode == 'assisted'):
        params                  = params_assisted_init
        params['n_estimators']  = 5
        scoring                 = 'average_precision'
        
    # Prepare XGBoost model    
    model       = xgb.XGBClassifier(**params) 
    
    # Get feature importance
    feature_importance_df = get_feature_importance(model, X, y, groups, dataset, prediction_mode, date)
    
    # Perform forward feature evaluation based on importance
    selected    = [] # List with names of selected features

    # Prepare results table
    results     = {'p5 train':  [],
                   'p25 train': [],
                   'p50 train': [],
                   'p75 train': [],
                   'p95 train': [],
                   'p5 test':   [],
                   'p25 test':  [],
                   'p50 test':  [],
                   'p75 test':  [],
                   'p95 test':  []}
      
    # Add feature and perform cross-validation
    for feature in feature_importance_df.index.to_list():
        selected.append(feature)        
        scores = cross_validate(model, X[selected], y, groups=groups, cv=cv, scoring=scoring, return_train_score=True, error_score='raise')
               
        # Save score percentiles
        for score in ['train', 'test']:
            for percentile in [5,25,50,75,95]:
                results[f'p{percentile} {score}'].append(np.percentile(list(scores[f'{score}_score']), percentile))

        # Stop if the max number of features to be added is reached
        if len(selected) == nr_features:
            break

    # Save results
    results_df = pd.DataFrame(results, index=range(1,nr_features+1))
    results_df.to_csv(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/feature_selection_performances_{date}.csv')
    
    # Plot feature performance curve
    x       = range(1,nr_features+1)
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(x, np.array(results['p50 train']), label='Train score', color=palette['red'])
    ax.fill_between(x, np.array(results['p5 train']), np.array(results['p95 train']), label="_none", color=palette['red'], alpha = 0.2)
    ax.fill_between(x, np.array(results['p25 train']), np.array(results['p75 train']), label="_none", color=palette['red'], alpha = 0.2)
                  
    ax.plot(x, np.array(results['p50 test']), label='Validation score', color=palette['blue'])
    ax.fill_between(x, np.array(results['p5 test']), np.array(results['p95 test']), label="_none", color=palette['blue'], alpha = 0.2)
    ax.fill_between(x, np.array(results['p25 test']), np.array(results['p75 test']), label="_none", color=palette['blue'], alpha = 0.2)

    ax.set_title('Feature curve')
    ax.set_xlabel('Nr of features')
    ax.set_xticks(ticks=range(1,nr_features+1,1), labels=range(1,nr_features+1,1))
    
    # Add second x-axis with added feature name
    secax = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
    ticks = ax.get_xticks()
    secax.set_xticks(ticks)
    secax.set_xticklabels(selected)
    secax.set_xlabel('Added feature')
    secax.spines['bottom'].set_position(('outward', 40))
    
    fig.canvas.draw()
    
    for label in secax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_va('top')
        label.set_rotation_mode('anchor')
        
    if scoring == 'roc_auc': 
        ax.set_ylim(0.4,1)
        ax.set_ylabel('AUROC')
    elif scoring == 'average_precision': 
        ax.set_ylim(0,1)
        ax.set_ylabel('AUPRC')
    
    ax.legend(loc='upper left')
    ax.grid(color='black', linewidth=0.5, linestyle=':')
    
    plt.savefig(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/figures/feature_curve_{date}.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_models(date1, date2, prediction_mode, alternative='greater'):
    '''
    Function to perform Wilcoxon signed rank test to compare performance results of 2 models.

    Parameters
    ----------
    date1 : str
        Datetime of metrics table generation of sample 1.
    date2 : str
        Datetime of metrics table generation of sample 2.
    prediction_mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.
    alternative : str, optional
        Defines the alternative hypothesis: 'two-sided' for equal distributions, 'less' for a lower median of first sample, 'greater' for a greater median of first sample. The default is 'greater'.

    Returns
    -------
    p_auroc : float
        P-value associated with the given alternative for AUROC.
    p_auprc : float
        P-value associated with the given alternative for AUPRC.

    '''
    
    # Read metrics table for both samples
    metrics1    = pd.read_excel(f'P:\\Emmelieve\\output_train3\\{prediction_mode}\\metrics_{date1}.xlsx', index_col=0)
    metrics2    = pd.read_excel(f'P:\\Emmelieve\\output_train3\\{prediction_mode}\\metrics_{date2}.xlsx', index_col=0)

    # Perform wilcoxon signed rank test for AUROC
    auroc1      = metrics1.loc[0:9,'AUROC']
    auroc2      = metrics2.loc[0:9,'AUROC']
    #_, p_auroc = stats.ttest_rel(auroc1, auroc2, alternative=alternative)  # paired t-test
    _, p_auroc  = stats.wilcoxon(auroc1, auroc2, alternative=alternative)
    print('AUROC p:', p_auroc)

    # Perform wilcoxon signed rankt test for AUPRC
    auprc1      = metrics1.loc[0:9,'AUPRC']
    auprc2      = metrics2.loc[0:9,'AUPRC']
    # _, p_auprc = stats.ttest_rel(auprc1, auprc2, alternative=alternative) # paired t-test
    _, p_auprc  = stats.wilcoxon(auprc1, auprc2, alternative=alternative)
    print('AUPRC p:', p_auprc)
    print('')
    
    return p_auroc, p_auprc

  
def end_model_training(train_set, mode, us=False):
    '''
    Function to train and return the final model after feature selection and hyperparameter optimization.

    Parameters
    ----------
    train_set : str
        Name of the training dataset.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    us : bool, optional
        Wheter to apply undersampling on control samples. The default is False.

    Returns
    -------
    X_train : DataFrame
        Model input.
    y_train : Series
        Data labels.
    y_pred : array
        Predicted probabilities.
    model : xgboost.sklearn.XGBClassifier
        Trained XGBoost model.

    '''
        
    # Get model inputs    
    selected_features = pd.read_csv(f'P:/Emmelieve/input_feature_tables_train3/selected_features_{mode}.csv', header=None)[0].to_list()
    train             = pd.read_csv(f'P:/Emmelieve/input_feature_tables_train3/input_feature_values_{mode}.csv')
    
    # Apply undersampling
    if us is True:
        train = undersampling(train, mode)
    
    # Feature values
    X_train = train[selected_features].copy()
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True) 
    
    # Labels
    y_train = train.event
    y_train = y_train.astype(int)
            
    # Get hyperparameter settings
    if mode == 'controlled':
        params = params_controlled
    elif mode == 'controlled_1B':
        params = params_controlled_beta
    else:
        params = params_assisted
    
    # Fit and save model
    model   = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    model.save_model(f'xgboost_final_{mode}.json')
    
    # Get predicted probabilities
    y_pred  = model.predict_proba(X_train)
    y_pred  = y_pred[:,1]
    
    # Save predictions on training samples
    combined_df = train[['patient_id', 'hospital_number', 'timestamp', 'prediction_window', 'event', 'age', 'gender'] + selected_features].copy()
    combined_df['prediction'] = y_pred
    combined_df.to_csv(f'P:/Emmelieve/output_test/{mode}/covid_train.csv', index=False)
    
    return X_train, y_train, y_pred, model
 
   
def plot_learning_curve(date, mode, feature_selection, optimized, n_estimators=30, title='Learning curve', beta=False):
    '''
    Function to generate a learning curve.

    Parameters
    ----------
    date : str
        Datetime of feature input table generation.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'. 
    feature_selection : bool, optional
        Whether to use only selected features or all features in the feature table. The default is False.
    optimized : bool, optional
        Wheter to use optimised hyperparameter settings or not. The default is False.
    n_estimators : int, optional
        The number of boosting iterations. The default is 30.
    title : str, optional
        Figure title. The default is 'Initial learning curve'.

    Returns
    -------
    None.

    '''
    
    # Get model input
    train = pd.read_csv(f'P:/Emmelieve/input_feature_tables_train3/input_feature_values_{date}.csv')
    train.replace([np.inf, -np.inf], np.nan, inplace=True)  # remove infinity values
    
    # Apply undersampling for Model 2
    if mode == 'assisted':    
        train = undersampling(train, mode)
        
    # Get selected features
    if feature_selection is True:
        selected_features = pd.read_csv(f'P:/Emmelieve/input_feature_tables_train3/selected_features_{date}.csv', header=None)[0].to_list()
            
    # Get hyperparameter settings and scoring
    if mode =='controlled':
        
        if optimized is True:
            params      = params_controlled
            if beta is True:
                params  = params_controlled_beta
        else:
            params      = params_controlled_init
            params['n_estimators'] = n_estimators            
            
        scoring     = 'roc_auc'
        metric      = 'AUROC'
        ylim        = [0.4,1.02]
        
    elif mode =='assisted':
        
        if optimized is True:
            params  = params_assisted
        else:
            params  = params_assisted_init
            params['n_estimators'] = n_estimators
        
        scoring     = 'average_precision'
        metric      = 'AUPRC'
        ylim        = [0,1.02]

    # Prepare model
    model = xgb.XGBClassifier(**params)
    
    # Prepare cross-validation for different training set sizes
    scores_train = []
    sd_train     = []
    scores_test  = []
    sd_test      = []
    
    median_train = []
    p5_train     = []
    p25_train    = []
    p75_train    = []
    p95_train    = []
    
    median_test  = []
    p5_test      = []
    p25_test     = []
    p75_test     = []
    p95_test     = []
    
    sizes        = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    for size in sizes:
        # Sample training data
        
        if size == 1:
            train_subset    = train.copy()
        else:
            train_subset, _ = train_test_split(train, train_size=size, stratify=train['event'], random_state=36)
        
        # Get model inputs
        if feature_selection is True:
            X   = train_subset[selected_features]
        else:
            X   = train_subset.drop(columns=['hospital_number', 'patient_id', 'event','timestamp', 'prediction_window'])
        
        # Get labels
        y       = train_subset.event
        y       = y.astype(int)
        
        # Get group number for grouped cross validation        
        groups  = train_subset.hospital_number
        
        # Perform cross validation
        scores  = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring, return_train_score=True, error_score='raise')
        
        # Save train and tes scores
        scores_train.append(np.mean(list(scores['train_score'])))
        sd_train.append(np.std(list(scores['train_score'])))
        scores_test.append(np.mean(list(scores['test_score'])))
        sd_test.append(np.std(list(scores['test_score'])))
        
        median_train.append(np.median(list(scores['train_score'])))
        p5_train.append(np.percentile(list(scores['train_score']),10))
        p25_train.append(np.percentile(list(scores['train_score']),25))
        p75_train.append(np.percentile(list(scores['train_score']),75))
        p95_train.append(np.percentile(list(scores['train_score']),90))
        
        median_test.append(np.median(list(scores['test_score'])))
        p5_test.append(np.percentile(list(scores['test_score']),10))
        p25_test.append(np.percentile(list(scores['test_score']),25))
        p75_test.append(np.percentile(list(scores['test_score']),75))
        p95_test.append(np.percentile(list(scores['test_score']),90))
        
    
    # Generate figure with performance per sample size
    x               = [size*len(train) for size in sizes]
    scores_train    = np.array(scores_train)
    sd_train        = np.array(sd_train)
    scores_test     = np.array(scores_test)
    sd_test         = np.array(sd_test)
    
    plt.figure(figsize=(10,6))
    plt.plot(x, np.array(median_test), label='Validation score', color=palette['blue'])
    plt.fill_between(x, np.array(p5_test), np.array(p95_test), color=palette['blue'], alpha=0.2)
    plt.fill_between(x, np.array(p25_test), np.array(p75_test), color=palette['blue'], alpha=0.2)
    
    plt.plot(x, np.array(median_train), label='Train score', color=palette['red'])
    plt.fill_between(x, np.array(p5_train), np.array(p95_train), color=palette['red'], alpha=0.2)
    plt.fill_between(x, np.array(p25_train), np.array(p75_train), color=palette['red'], alpha=0.2)
       
    plt.title(title)
    plt.xlabel('Nr of training samples')
    plt.ylabel(metric)
    plt.ylim(ylim)
    plt.legend(loc='lower right')
    plt.grid(color='black', linewidth=0.5, linestyle=':')

    if optimized is True:
        plt.savefig(f'P:/Emmelieve/output_train3/{mode}/figures/learning_curve_{date}_optimized.png', dpi=300, bbox_inches='tight')
    elif feature_selection is True:
        plt.savefig(f'P:/Emmelieve/output_train3/{mode}/figures/learning_curve_{date}_selected_features.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'P:/Emmelieve/output_train3/{mode}/figures/learning_curve_{date}_{n_estimators}.png', dpi=300, bbox_inches='tight')
   
    plt.close()


def backward_feature_selection(dataset, date, prediction_mode, nr_features=20):

    # Get model inputs
    X, y, groups    = prepare_inputs(dataset, date, prediction_mode)  

    # Get hyperparameter settings
    if (prediction_mode=='controlled'):
        params                 = params_controlled_init
        params['n_estimators'] = 5
        scoring                = 'roc_auc'
        
    elif (prediction_mode == 'assisted'):
        params                  = params_assisted_init
        params['n_estimators']  = 5
        scoring                 = 'average_precision'
        
    # Prepare XGBoost model    
    model       = xgb.XGBClassifier(**params) 
    
    # Get feature importance
    feature_importance_df = get_feature_importance(model, X, y, groups, dataset, prediction_mode, date)

    # Prepare results table
    features = feature_importance_df.index[0:nr_features].to_list()
    eliminated = []

    results = { 'p5 train':  [],
                'p25 train': [],
                'p50 train': [],
                'p75 train': [],
                'p95 train': [],
                'p5 test':   [],
                'p25 test':  [],
                'p50 test':  [],
                'p75 test':  [],
                'p95 test':  []}
      
    # Backward feature elimination based on importance
    
    while True:
        
        # Perform cross validation
        scores = cross_validate(model, X[features], y, groups=groups, cv=cv, scoring=scoring, return_train_score=True, error_score='raise')
        
        # Save results
        for score in ['train', 'test']:
            for percentile in [5,25,50,75,95]:
                results[f'p{percentile} {score}'].append(np.percentile(list(scores[f'{score}_score']), percentile))   
        
        # Break if only 1 feature is left
        if len(features) == 1:
            eliminated.append(features[0])
            break
        
        # Get feature importance
        feature_importance_df = get_feature_importance(model, X[features], y, groups, dataset, prediction_mode, date, save=False, figure=False)
           
        # Eliminate least important feature
        eliminated.append(feature_importance_df.index[-1])
        features = feature_importance_df.index[0:-1].to_list()
    
    results_df = pd.DataFrame(results, index=range(1,nr_features+1))
    results_df.to_csv(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/backward_feature_selection_performances_{date}.csv')
    
    # Plot feature curve
    x = range(1,nr_features+1)
    eliminated.reverse()
    
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(x, np.array(results['p50 train']), label='Train score', color=palette['red'])
    ax.fill_between(x, np.array(results['p5 train']), np.array(results['p95 train']), label="_none", color=palette['red'], alpha = 0.2)
    ax.fill_between(x, np.array(results['p25 train']), np.array(results['p75 train']), label="_none", color=palette['red'], alpha = 0.2)
                  
    ax.plot(x, np.array(results['p50 test']), label='Validation score', color=palette['blue'])
    ax.fill_between(x, np.array(results['p5 test']), np.array(results['p95 test']), label="_none", color=palette['blue'], alpha = 0.2)
    ax.fill_between(x, np.array(results['p25 test']), np.array(results['p75 test']), label="_none", color=palette['blue'], alpha = 0.2)

    ax.set_title('Feature curve')
    ax.set_xlabel('Nr of features')
    ax.set_xticks(range(nr_features,0,-1))
    ax.set_xticklabels(range(1,nr_features+1,1))
    
    # Add second x-axis with name of eliminated feature
    secax = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
    ticks = ax.get_xticks()
    secax.set_xticks(ticks)
    secax.set_xticklabels(eliminated)
    secax.set_xlabel('Eliminated feature')
    secax.spines['bottom'].set_position(('outward', 40))
    
    fig.canvas.draw()

    for label in secax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_va('top')
        label.set_rotation_mode('anchor')
    
    if scoring == 'roc_auc': 
        ax.set_ylim(0.4,1)
        ax.set_ylabel('AUROC')
        
    if scoring == 'average_precision': 
        ax.set_ylim(0,1)
        ax.set_ylabel('AUPRC')
    
    ax.legend()
    ax.grid(color='black', linewidth=0.5, linestyle=':')
    
    plt.savefig(f'P:/Emmelieve/output_{dataset}/{prediction_mode}/figures/feature_curve_bfe_{date}.png', dpi=300, bbox_inches='tight')
    plt.close()
  