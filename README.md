# Logistic Regression from scratch

This repository contains **Logistic Regression** implementation from scratch.

## Features
The main class *LogReg* is a custom implementation of logistic regression using NumPy. It supports both L1 and L2 regularization, custom evaluation metrics, and Stochastic Gradient Descent  for optimization. It supports the following options:
- *n_iter* - number of training iterations (epochs).
- *learning_rate* - learning rate for gradient descent.
- *metric* - evaluation metric name.
- *reg* - regularization name.
- *l1_coef* - L1 regularization coefficient (default 0.0).
- *l2_coef*	- L2 regularization coefficient (default 0.0).
- *sgd_sample* - if set, use a random subset of data for each iteration. Can be an integer (count) or float (fraction).
- *random_state* - random seed for reproducibility.

Implemented regularizers:
- L1 (Lasso),
- L2 (Ridge),
- ElasticNet.

Implemented metrics:
- Accuracy,
- Precision,
- Recall,
- F1-score,
- ROC AUC. 

Custom learning rate (float or callable per iteration) can be set during model initialization.

Verbose training output can be set for every *n* iterations.

### Install dependencies
```console
pip install requirements.txt
```

### Run Logistic Regression example
To start the main script, execute the following command:
```console
python main.py [OPTIONS]
```

#### Available options
- **-e, --example** (required) - type of example to run. Available examples: linear, regularization.
- **-i, --iter** (required) - number of training iterations.
- **-l, --lr** (required) - learning rate for gradient descent.
- **-m, --metric** (required) - evaluation metric to track during training. Available metrics: accuracy, precision, recall, f1, roc_auc.
- **-r, --reg** (optional) - regularization type. Available regularizers: l1, l2, elasticnet.
- **--l1** (optional) - L1 regularization coefficient (float).
- **--l2** (optional) - L2 regularization coefficient (float).