# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:25:58 2025

@author: eldenbreejen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dcurves import dca
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix

palette = {'blue': '#2D69FF',
           'red': '#FF6E50',
           'green': '#4BAA7D',
           'navy': '#060453'}


def model_testing(model, test_set1, test_set2, mode):
    '''
    Function to make predictions with the trained model for 2 test datasets.

    Parameters
    ----------
    model : xgboost.sklearn.XGBClassifier
        Fitted model.
    test_set1 : str
        Name of test dataset 1.
    test_set2 : str
        Name of test dataset 2.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    X_test1 : DataFrame
        Feature input table for test dataset 1.
    y_test1 : Series
        Event labels for test dataset 1.
    y_pred1 : array
        Predicted probabilities for test dataset 1.
    X_test2 : DataFrame
        Feature input table for test dataset 2.
    y_test2 : Series
        Event labels for test dataset 2.
    y_pred2 : array
        Predicted probabilities for test dataset 2.
        
    '''
    
    # Read feature input tables
    selected_features = pd.read_csv(f'P:/Emmelieve/input_feature_tables_train3/selected_features_{mode}.csv', header=None)[0].to_list()
    test1             = pd.read_csv(f'P:/Emmelieve/input_feature_tables_{test_set1}/input_feature_values_{mode}.csv')
    test2             = pd.read_csv(f'P:/Emmelieve/input_feature_tables_{test_set2}/input_feature_values_{mode}.csv')
    
    # Prepare model input and event labels for test set 1
    X_test1 = test1[selected_features].copy()
    X_test1.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    y_test1 = test1.event
    y_test1 = y_test1.astype(int)
    
    print(test_set1)
    print('events:', y_test1.sum())
    print('control events:', len(y_test1)-y_test1.sum())
    print('')
    
    # Prepare model input and event labels for test set 2
    X_test2 = test2[selected_features].copy()
    X_test2.replace([np.inf, -np.inf], np.nan, inplace=True) 
    
    y_test2 = test2.event
    y_test2 = y_test2.astype(int)
    
    print(test_set2)
    print('events:', y_test2.sum())
    print('control events:', len(y_test2)-y_test2.sum())
    print('')
    
    # Make predictions
    y_pred1  = model.predict_proba(X_test1)
    y_pred1  = y_pred1[:,1]
    
    y_pred2  = model.predict_proba(X_test2)
    y_pred2  = y_pred2[:,1]
    
    # Save predictions
    test1['event_pred'] = y_pred1
    test2['event_pred'] = y_pred2
    
    test1.to_csv(f'P:/Emmelieve/output_test/{mode}/covid_test.csv', index=None)
    test2.to_csv(f'P:/Emmelieve/output_test/{mode}/noncovid_test.csv', index=None)
    
    return X_test1, y_test1, y_pred1, X_test2, y_test2, y_pred2


def auroc_auprc(y_test1, y_pred1, y_test2, y_pred2, target, metric, mode):
    '''
    Function to calculate AUROC and AUPRC and generate ROC and PR curves.

    Parameters
    ----------
    y_test1 : Series
        Event labels for test set 1.
    y_pred1 : array
        Predicted probabilitie for test set 1.
    y_test2 : Series
        Event labels for test set 2.
    y_pred2 : array
        Predicted probabilities for test set 2.
    target : float
        Target value for metric.
    metric : str
        Metric to set the target for: 'specificity', 'sensitivity' or 'PPV'.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    threshold1 : float
        Threshold probability for test set 1 for the set target.
    threshold2 : float
        Threshold probability for test set 2 for the set target.

    '''
    
    # Test set 1
    # Calculate the AUROC
    auroc1                  = roc_auc_score(y_test1, y_pred1)
    fpr1, tpr1, thresholds1 = roc_curve(y_test1, y_pred1)
    
    # Generate ROC curve
    plt.figure(figsize=(8,8))
    plt.plot([0,1], [0,1], label='Random classifier', linestyle='--', color='black')
    plt.plot(fpr1, tpr1, label=f'COVID test set (AUC = {round(auroc1,2)})', color=palette['blue'])
    
    # Calculate threshold probability for set target
    if metric   =='specificity':
        metric1 = fpr1
        target  = 1-target
    elif metric == 'sensitivity':
        metric1 = tpr1
    elif metric == 'PPV':
        prevalence  = y_test1.sum()/len(y_test1)
        denom       = tpr1 * prevalence + fpr1 * (1 - prevalence)
        metric1     = np.divide(tpr1 * prevalence, denom, out=np.zeros_like(denom), where=denom != 0)
        
    idx        = np.argmin(np.abs((metric1) - target))
    threshold1 = thresholds1[idx]
    
    # Test set 2
    # Calculate the AUROC
    auroc2                  = roc_auc_score(y_test2, y_pred2)
    fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_pred2)
    
    # Generate ROC curve
    plt.plot(fpr2, tpr2, label=f'Non-COVID test set (AUC = {round(auroc2,2)})', color=palette['red'])
    
    # Calculate threshold probability for set target
    if metric   =='specificity':
        metric2 = fpr2
    elif metric == 'sensitivity':
        metric2 = tpr2
    elif metric == 'PPV':
        prevalence  = y_test2.sum()/len(y_test2)
        denom       = tpr2 * prevalence + fpr2 * (1 - prevalence)
        metric2     = np.divide(tpr2 * prevalence, denom, out=np.zeros_like(denom), where=denom != 0)
        
    idx         = np.argmin(np.abs((metric2) - target))
    threshold2  = thresholds2[idx]
    
    # Finish figure lay-out and save
    plt.legend()
    plt.title('Receiver operating characteristic')
    plt.xlabel('1 - specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.grid(color='black', linewidth=0.5, linestyle=':')
    
    plt.savefig(f'P:/Emmelieve/output_test/{mode}/roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate AUPRC and generate precision recall curves
    plt.figure(figsize=(8,8))
    
    rand1   = (y_test1 == 1).sum() / len(y_test1)     # a priori precision test set 1
    plt.plot([0,1], [rand1, rand1], label='Random classifier COVID', linestyle='--', color=palette['blue'], alpha=0.5)
    
    rand2   = (y_test2 == 1).sum() / len(y_test2)     # a priori precision test set 2
    plt.plot([0,1], [rand2, rand2], label='Random classifier non-COVID', linestyle='--', color=palette['red'], alpha=0.5)
    
    precision, recall, thresholds = precision_recall_curve(y_test1, y_pred1)
    auprc1  = auc(recall, precision)       
    plt.plot(recall, precision, label=f'COVID test set (AUC = {round(auprc1,2)})', color=palette['blue'])
    
    precision, recall, thresholds = precision_recall_curve(y_test2, y_pred2)
    auprc2  = auc(recall, precision)       
    plt.plot(recall, precision, label=f'Non-COVID test set (AUC = {round(auprc2,2)})', color=palette['red'])
    
    plt.legend()
    plt.title('Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.grid(color='black', linewidth=0.5, linestyle=':')
    
    plt.savefig(f'P:/Emmelieve/output_test/{mode}/prc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # print scores
    print('AUROC COVID:', auroc1)
    print('AUPRC COVID:', auprc1) 
    print('')
    print('AUROC non-COVID:', auroc2)
    print('AUPRC non-COVID:', auprc2) 
    
    return threshold1, threshold2

def summary(y_true, y_prob, threshold, mode):
    '''
    Function to print classification report and sens, spec, ppv, npv at given threshold.

    Parameters
    ----------
    y_true : Series
        Event labels.
    y_prob : array
        Predicted probabilities.
    threshold : float
        Treshold probability.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
    
    # Get event prediction
    y_pred = y_prob >= threshold
    
    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    
    # Print sens, spec, ppv, npv
    print('sensitivity', tp/(tp+fn))
    print('specificty', tn/(tn+fp))
    print('ppv', tp/(tp+fp))
    print('npv', tn/(tn+fn))

    
def calibration_curve(y1, y1pred, y2, y2pred, y3, y3pred, mode, calibrated=False):
    '''
    Function to generate a calibration curve.

    Parameters
    ----------
    y1 : Series
        Event labels for dataset 1 (train set).
    y1pred : array
        Predicted probabilites for dataset 1 (train set).
    y2 : Series
        Event labels for dataset 2 (test set 1).
    y2pred : array
        Predicted probabilites for dataset 2 (test set 1).
    y3 : Series
        Event labels for dataset 3 (test set 2).
    y3pred : array
        Predicted probabilites for dataset 3 (test set 2).
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.
    calibrated : bool, optional
        Wheter the probabilities are calibrated. The default is False.

    Returns
    -------
    None.

    '''
    
    # Generate calibration plot
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_predictions(y1, y1pred, n_bins=10, strategy='quantile', name='Training set', ax=ax, color=palette['blue'])
    CalibrationDisplay.from_predictions(y2, y2pred, n_bins=10, strategy='quantile', name='COVID test set', ax=ax, color=palette['red'])
    CalibrationDisplay.from_predictions(y3, y3pred, n_bins=10, strategy='quantile', name='Non-COVID test set', ax=ax, color=palette['green'])
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.grid(color='black', linewidth=0.5, linestyle=':')
    
    # Set title and save figure
    if calibrated is True:
        plt.title('Calibration curves after model calibration')
        plt.savefig(f'P:/Emmelieve/output_test/{mode}/calibration_curve_calibrated.png', dpi=300, bbox_inches='tight')
    else:
        plt.title('Calibration curves')
        plt.savefig(f'P:/Emmelieve/output_test/{mode}/calibration_curve.png', dpi=300, bbox_inches='tight')
        
    plt.close()
    
def model_calibration(model, X, y):
    '''
    Function to calibrate the model.

    Parameters
    ----------
    model : xgboost.sklearn.XGBClassifier
        Fitted model.
    X : Dataframe
        Input features.
    y : Series
        Event labels.

    Returns
    -------
    y_pred : array
        Calibrated predicted probabilities.

    '''
    
    # Calibrate model (since cv='prefit' the model is not fitted again and all data is used to calibrate the model)
    calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X, y)
    
    # Generate new predictions
    y_pred = calibrated_model.predict_proba(X)[:,1]
    
    return y_pred

def decision_curve(y1, y_pred1, y2, y_pred2, mode):
    '''
    Function to generate a net-benefit (decision curve) analysis.

    Parameters
    ----------
    y1 : Series
        Event labels for test set 1.
    y_pred1 : array
        Predicted probabilities for test set 1.
    y2 : Series
        Event labels for test set 2.
    y_pred2 : array
        Predicted probabilities for test set 2.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
    
    # Calculate net benefit for test set 1
    df1             = pd.DataFrame({'y_true': y1, 'y_pred': y_pred1})
    dca1            = dca(data=df1, outcome='y_true', modelnames=['y_pred'], thresholds=np.arange(0,1,0.1)) 

    thresholds1     = dca1['threshold']
    net_benefit1    = dca1['net_benefit']
    
    # Calculate net benefit for tes set 2
    df2             = pd.DataFrame({'y_true': y2, 'y_pred': y_pred2})
    dca2            = dca(data=df2, outcome='y_true', modelnames=['y_pred'], thresholds=np.arange(0,1,0.1)) 

    thresholds2     = dca2['threshold']
    net_benefit2    = dca2['net_benefit']
     
    # Generate figure with net-benefit analysis
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(thresholds1[0:10], net_benefit1[0:10], color=palette['red'], label='Model')
    plt.plot(thresholds1[10:20], net_benefit1[10:20], color=palette['blue'], label='Treat all as unready')
    plt.plot(thresholds1[20:30], net_benefit1[20:30], color=palette['green'], label='Treat all as ready')
    
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.title('Net-benefit analysis COVID test set')
    plt.grid(color='black', linewidth=0.5, linestyle=':')
    plt.legend()
    plt.ylim(-0.05, 0.2)
    
    
    plt.subplot(1,2,2)
    plt.plot(thresholds2[0:10], net_benefit2[0:10], color=palette['red'], label='Model')
    plt.plot(thresholds2[10:20], net_benefit2[10:20], color=palette['blue'], label='Treat all as unready')
    plt.plot(thresholds2[20:30], net_benefit2[20:30], color=palette['green'], label='Treat all as ready')
    
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.title('Net-benefit analysis Non-COVID test set')
    plt.grid(color='black', linewidth=0.5, linestyle=':')
    plt.legend()
    plt.ylim(-0.05, 0.2)
    
    # Save figure
    plt.savefig(f'P:/Emmelieve/output_test/{mode}/dca.png', dpi=300, bbox_inches='tight')
    plt.close()
    

        
def outcome_distribution_plot(y1, y_pred1, y2, y_pred2, mode, name=''):
    '''
    Function to make an outcome distribution plot with violin, box and jitter plot for 2 datasets.

    Parameters
    ----------
    y1 : Series
        Event labels for test set 1.
    y_pred1 : array
        Predicted probabilities for test set 1.
    y2 : Series
        Event labels for test set 2.
    y_pred2 : array
        Predicted probabilities for test set 2.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
    
    # Prepare plot
    set1    = 'COVID test set'
    set2    = 'non-COVID test set'
    
    labels  = ['Event', 'Control']
    colors  = [palette['red'], palette['blue']]
    
    base_positions  = [1, 2]
    offsets         = {
                        'violin':  0.2,
                        'box':     0.0,
                        'jitter': -0.3
                    }    

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7))

    # Create a subplot for each dataset
    for ax, y, y_pred, dataset in [(ax1, y1, y_pred1, set1), (ax2, y2, y_pred2, set2)]:
        
        # Get event and control group
        groups = [y_pred[y == 1], y_pred[y == 0]]
        
        # Create violinplot
        vp = ax.violinplot(
            groups,
            positions=[p + offsets['violin'] for p in base_positions],
            vert=False,
            showextrema=False,
            widths=0.25
        )
        
        # Set colors
        for body, p, color in zip(vp['bodies'], base_positions, colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(1)
            
        
            # Keep half violin
            path            = body.get_paths()[0]
            verts           = path.vertices
            y_center        = p + offsets['violin']
            verts[:, 1]     = np.maximum(verts[:, 1], y_center)
            path.vertices   = verts
        
        # Create boxplot
        bp = ax.boxplot(
            groups,
            positions=[p + offsets['box'] for p in base_positions],
            vert=False,
            widths=0.15,
            patch_artist=True,
            showfliers=False
        )
        
        # Set colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
        
        for i, color in enumerate(colors):
            for element in ['boxes', 'medians']:
                bp[element][i].set_color(color)
        
            for element in ['whiskers', 'caps']:
                bp[element][2*i].set_color(color)
                bp[element][2*i + 1].set_color(color)
        
        # Create jitter plot
        for p, data, color in zip(base_positions, groups, colors):
            jitter_y = np.random.normal(p + offsets['jitter'], 0.03, size=len(data))
            ax.scatter(
                data,
                jitter_y,
                color=color,
                alpha=0.5,
                s=12,
                edgecolor='none'
            )
        
        # Ax settings
        ax.set_yticks(base_positions)
        ax.set_yticklabels(labels)
        ax.set_xlim([0,1])
        ax.set_xlabel('Predicted probability')
        ax.set_title(f'Probability distribution {dataset}')
        ax.grid(axis='x', linestyle=':', linewidth=0.5, color='black')
        ax.set_axisbelow(True)
        
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'P:/Emmelieve/output_test/{mode}/distribution_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()
        
def outcome_distribution_plot1(y, y_pred, mode):
    '''
    Function to make an outcome distribution plot with violin, box and jitter plot for 1 dataset.

    Parameters
    ----------
    y1 : Series
        Event labels.
    y_pred1 : array
        Predicted probabilities.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
    
    # Prepare plot    
    labels  = ['Event', 'Control']
    colors  = [palette['red'], palette['blue']]
    
    base_positions  = [1, 2]
    offsets         = {
                        'violin':  0.2,
                        'box':     0.0,
                        'jitter': -0.3
                    }    

    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Get event and control group
    groups  = [y_pred[y == 1], y_pred[y == 0]]
    
    # Create violinplot
    vp = ax.violinplot(
        groups,
        positions=[p + offsets['violin'] for p in base_positions],
        vert=False,
        showextrema=False,
        widths=0.25
    )
    
    # Set colors
    for body, p, color in zip(vp['bodies'], base_positions, colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(1)
        
    
        # Keep half violin
        path            = body.get_paths()[0]
        verts           = path.vertices
        y_center        = p + offsets['violin']
        verts[:, 1]     = np.maximum(verts[:, 1], y_center)
        path.vertices   = verts
    
    # Create boxplot
    bp = ax.boxplot(
        groups,
        positions=[p + offsets['box'] for p in base_positions],
        vert=False,
        widths=0.15,
        patch_artist=True,
        showfliers=False
    )
    
    # Set colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)
    
    for i, color in enumerate(colors):
        for element in ['boxes', 'medians']:
            bp[element][i].set_color(color)
    
        for element in ['whiskers', 'caps']:
            bp[element][2*i].set_color(color)
            bp[element][2*i + 1].set_color(color)
    
    # Create jitter plot
    for p, data, color in zip(base_positions, groups, colors):
        jitter_y = np.random.normal(p + offsets['jitter'], 0.03, size=len(data))
        ax.scatter(
            data,
            jitter_y,
            color=color,
            alpha=0.5,
            s=12,
            edgecolor='none'
        )
    
    # Ax settings
    ax.set_yticks(base_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim([0,1])
    ax.set_xlabel('Predicted probability')
    ax.set_title('Probability distribution')
    ax.grid(axis='x', linestyle=':', linewidth=0.5, color='black')
    ax.set_axisbelow(True)
        
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'P:/Emmelieve/output_test/{mode}/distribution1.png', dpi=300, bbox_inches='tight')
    plt.close() 
