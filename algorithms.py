from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression

regression_algorithms = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
        }
    },
    'ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']  # Added parameter
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']  # Added parameter
        }
    },
    'elastic_net': {
        'model': ElasticNet(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']  # Added parameter
        }
    },
    'sgd_regressor': {
        'model': SGDRegressor(),
        'params': {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'fit_intercept': [True, False],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'max_iter': [500, 1000, 1500]  # Added parameter
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']  # Added parameter
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'max_features': ['auto', 'sqrt', 'log2']  # Added parameter
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['auto', 'sqrt', 'log2']  # Added parameter
        }
    },
    'svm': {
        'model': SVR(),
        'params': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4, 5],  # Only used for 'poly' kernel
            'coef0': [0.0, 0.1, 0.5, 1.0, 2.0],  # Often useful for 'poly' and 'sigmoid' kernels
            'epsilon': [0.1, 0.2, 0.5, 1, 2]  # Added parameter for specifying the epsilon-tube within which no penalty is associated in the training loss function
        }  
    },   
    'pls': {
        'model': PLSRegression(),
        'params': {
            'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10],  
            'scale': [True, False],  
            'max_iter': [500, 1000, 1500],  
            'tol': [1e-06, 1e-05, 1e-04]  
        }
    }
}

                      
classification_algorithms = ['Logistic Regression', 'SVM', 'Random Forest']
neural_network_algorithms = ['CNN', 'RNN', 'LSTM']

algorithm_options = {
    'Regression': regression_algorithms,
    'Classification': classification_algorithms,
    'Neural Network': neural_network_algorithms
}

regression_dropdown_options = [
    {'label': 'OptiML (All)', 'value': 'optiML_all_regression'},
    {'label': 'Linear Regression', 'value': 'linear_regression'},
    {'label': 'Ridge Regression', 'value': 'ridge'},
    {'label': 'Lasso Regression', 'value': 'lasso'},
    {'label': 'Elastic Net Regression', 'value': 'elastic_net'},
    {'label': 'SGD Regressor', 'value': 'sgd_regressor'},
    {'label': 'Decision Tree Regressor', 'value': 'decision_tree'},
    {'label': 'Random Forest Regressor', 'value': 'random_forest'},
    {'label': 'Gradient Boosting Regressor', 'value': 'gradient_boosting'},
    {'label': 'Support Vector Machine', 'value': 'svm'},
    {'label': 'PLS Regression', 'value': 'pls'}
]
classification_dropdown_options = [
    {'label': 'OptiML (All)', 'value': 'optiML_all_classification'},
    {'label': 'Logistic Regression', 'value': 'logistic_regression'},
    {'label': 'K-Nearest Neighbors', 'value': 'k_nearest_neighbors'},
    {'label': 'Support Vector Machine', 'value': 'support_vector_machine'},
    {'label': 'Decision Trees', 'value': 'decision_trees'},
    {'label': 'Random Forest', 'value': 'random_forest'},
    {'label': 'Gradient Boosting', 'value': 'gradient_boosting'},
    {'label': 'Naive Bayes', 'value': 'naive_bayes'},
    {'label': 'XGBoost', 'value': 'xgboost'},
    {'label': 'LightGBM', 'value': 'lightgbm'},
    {'label': 'AdaBoost', 'value': 'adaboost'}
]
neural_network_dropdown_options = [
    {'label': 'OptiML (All)', 'value': 'optiML_all_neural_network'},
    {'label': 'Basic Neural Network', 'value': 'basic_neural_network'},
    {'label': 'Convolutional Neural Network', 'value': 'convolutional_neural_network'},
    {'label': 'Recurrent Neural Network', 'value': 'recurrent_neural_network'},
    {'label': 'Long Short-Term Memory', 'value': 'lstm'},
    {'label': 'Gated Recurrent Unit', 'value': 'gru'},
    {'label': 'Transformer', 'value': 'transformer'},
    {'label': 'Autoencoder', 'value': 'autoencoder'}
]
