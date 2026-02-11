# TM3
Python scripts used for my TM3 thesis project.

<h2>Scripts</h2>
<ul>
  <li>main_26.py - contains a main function that sequentially performs data labeling, input feature calculation and model training + script for all experiments and analyses performed for model development, validation, and interpretation</li>
  <li>event_detection_functions_15.py - contains functions for data preprocessing and data labeling</li>
  <li>feature_input_functions_12.py - contains functions to retrieve tabular features and calculate aggregate feature values</li>
  <li>model_training_functions_13.py - contains functions to cross-validate XGBoost and logistic regresion models, compare model performance, create learning curves and feature curves, and train and save the final model</li>
  <li>model_testing_functions_10.py - contains functions to evaluate classification/discriminative performance, calibration, net-benefit, and outcome distribution</li>
  <li>explainability_functions_10.py - contains functions for SHAP analyses and decision tree visualisation</li>
  <li>timeline_functions_10.py - contains a function to create a scrollable timelines of ICU admissions with event labels</li>
  <li>stepwise_hyperparameter_tuning_controlled_26.py, stepwise_hyperparameter_tuning_controlled_1B_26.py, stepwise_hyperparameter_tuning_assisted_26.py - scripts used for hyperparameter optimisation</li>
</ul>

<h2>Software</h2>
Spider IDE 5.5.1, Python 3.12.4<br>

Libraries:
<ul>
  <li>scikit-learn 1.4.2</li>
  <li>xgboost 3.1.1</li>
  <li>dcurves 1.1.7</li>
  <li>shap 0.50.0</li>
  <li>dtreeviz 2.2.2</li>
</ul>
