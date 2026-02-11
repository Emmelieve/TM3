# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:25:44 2025

@author: eldenbreejen
"""
import numpy as np
import shap
import matplotlib.pyplot as plt
from dtreeviz import model
from pathlib import Path

palette = {'blue': '#2D69FF',
           'red': '#FF6E50',
           'green': '#4BAA7D',
           'navy': '#060453'}

def shap_summary(model, X, mode):
    '''
    Function to create a shap summary plot.

    Parameters
    ----------
    model : xgboost.sklearn.XGBClassifier
        Trained model (tree based).
    X : DataFrame
        Input feature values.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
   
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    shap_values = explanation.values

    shap.summary_plot(shap_values, X, max_display=20, plot_size=(18, 8), show=False)
    plt.savefig(f'P:/Emmelieve/output_explainability/{mode}/shap_summary.png', dpi=300)
    plt.close()
    
    
def local_shaps(model, X, y_true, y_pred, mode):
    '''
    Function to create shap waterfall plots for the 3 event samples with highest and lowest predicted probability.

    Parameters
    ----------
    model : xgboost.sklearn.XGBClassifier
        Trained model (tree based).
    X : DataFrame
        Input feature values.
    y_true : Series
        Event labels.
    y_pred : array
        Predicted probabilities.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
    explainer   = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    mask        = y_true == 1
    idx_candidates = np.where(mask)[0]

    top3_tp     = np.argsort(y_pred[mask])[-3:]
    idx         = idx_candidates[top3_tp]
    
    for i in idx:
        shap.plots.waterfall(shap_values[i], show=False)
        plt.title(f'Event sample with a high predicted probability (p={y_pred[i]:.2f})')
        plt.savefig(f'P:/Emmelieve/output_explainability/{mode}/local_shap_tp_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
    top3_fn = np.argsort(y_pred[mask])[:3]
    idx = idx_candidates[top3_fn]
    
    for i in idx:
        shap.plots.waterfall(shap_values[i], show=False)
        plt.title(f'Event sample with a low predicted probability (p={y_pred[i]:.2f})')
        plt.savefig(f'P:/Emmelieve/output_explainability/{mode}/local_shap_fn_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    mask        = y_true == 0
    idx_candidates = np.where(mask)[0]

    top3_fp     = np.argsort(y_pred[mask])[-3:]
    idx         = idx_candidates[top3_fp]
    
    for i in idx:
        shap.plots.waterfall(shap_values[i], show=False)
        plt.title(f'Control sample with a high predicted probability (p={y_pred[i]:.2f})')
        plt.savefig(f'P:/Emmelieve/output_explainability/{mode}/local_shap_fp_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
    top3_tn = np.argsort(y_pred[mask])[:3]
    idx = idx_candidates[top3_tn]
    
    for i in idx:
        shap.plots.waterfall(shap_values[i], show=False)
        plt.title(f'Control sample with a low predicted probability (p={y_pred[i]:.2f})')
        plt.savefig(f'P:/Emmelieve/output_explainability/{mode}/local_shap_tn_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    
def tree_visual(trained_model, X_train, y_train, trees, mode):
    '''
    Function to generate tree visualisations.

    Parameters
    ----------
    model : xgboost.sklearn.XGBClassifier
        Trained model (tree based).
    X_train : DataFrame
        Input feature values.
    y_train : Series
        Event labels.
    trees : list, optional
        List with tree numbers to be visualised.
    mode : str
        Ventilation mode during prediction, either 'controlled' or 'assisted'.

    Returns
    -------
    None.

    '''
    
    # Prepare X and y
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    
    # Create display
    for tree in trees:
        viz_model = model(
            trained_model.get_booster(),
            X_train         = X_train.values,
            y_train         = y_train.values,
            feature_names   = X_train.keys().to_list(),
            target_name     = 'Label',
            class_names     = ['Control', 'Event'],
            tree_index      = tree
        )
        
        viz = viz_model.view()
        viz.save(f'P:/Emmelieve/output_explainability/{mode}/tree_visual_{tree}.svg')
        
        # Change colors
        svg = Path(f"P:/Emmelieve/output_explainability/{mode}/tree_visual_{tree}.svg").read_text()
        
        svg = svg.replace("#fefebb", palette['blue'])     # Control
        svg = svg.replace("#a1dab4", palette['red'])      # Event
        
        Path(f"P:/Emmelieve/output_explainability/{mode}/tree_visual_{tree}.svg").write_text(svg)