# Importing required libraries
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Importing train set
df_train = pd.read_csv('train_final.csv')
print(df_train.head())

# Importing test set
df_test = pd.read_csv('test_final.csv')
print(df_test.head())

# Splitting into X_train, y_train, X_test
X_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']

X_test = df_test

# Creating a RandomizedSearchCV object to search for best parameters
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_train = linear_regressor.predict(X_train)

# Printing training regression metrics
print('Mean Absolute Error:', mean_absolute_error(y_train, y_pred_train))
print('Mean Squared Error:', mean_squared_error(y_train, y_pred_train))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('R2 Score:', r2_score(y_train, y_pred_train))

# Predicting on test data
y_pred_linear = linear_regressor.predict(X_test)

# Defining a dictionary of hyperparameters with values to tune over
params = {'n_estimators' : [100, 200, 500],
          'max_depth' : [int(x) for x in np.linspace(10, 100, 10)],
          'max_features' : ['auto', 'sqrt']
         }

# Creating a RandomizedSearchCV object to search for best parameters
random_forest_regressor = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params, cv=5, 
                                             n_iter=60, random_state=0, verbose=2, n_jobs=-1)
search = random_forest_regressor.fit(X_train, y_train)
print('Best Score:', search.best_score_)
print('Best Hyperparameters:', search.best_params_)

# Printing the cross validation results
results = pd.DataFrame(random_forest_regressor.cv_results_)
print(results)

# Predicting on test set
y_pred_random_forest = random_forest_regressor.predict(X_test)

# Defining a dictionary of hyperparameters with values to tune over
params = {'kernel' : ['linear', 'poly', 'rbf'],
          'C' : [1, 2, 3],
          'epsilon' : [0.1, 0.2, 0.3]
         }

# Creating a RandomizedSearchCV object to search for best parameters
support_vector_regressor = RandomizedSearchCV(estimator=SVR(), param_distributions=params, cv=5, 
                                              n_iter=100, random_state=0, verbose=2, n_jobs=-1)
search = support_vector_regressor.fit(X_train, y_train)
print('Best Score:', search.best_score_)
print('Best Hyperparameters:', search.best_params_)

# Printing the cross validation results
results = pd.DataFrame(support_vector_regressor.cv_results_)
print(results)

# Predicting on test data
y_pred_support_vector = support_vector_regressor.predict(X_test)

# First trying out a simple XGBoost model without any tuning
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
print('R2 Score:', r2_score(y_train, xgb.predict(X_train)))

# Defining a dictionary of hyperparameters with values to tune over
params = {'eta' : [0.001, 0.005, 0.01, 0.05],
          'max_depth' : [1, 2, 3, 4, 5, 6],
          'eval_metric' : ['rmse', 'mae']
         }

# Creating a RandomizedSearchCV object to search for best parameters
xgboost_regressor = RandomizedSearchCV(estimator=XGBRegressor(), param_distributions=params, cv=5, 
                                       n_iter=100, random_state=0, verbose=2, n_jobs=-1)
search = xgboost_regressor.fit(X_train, y_train)
print('Best Score:', search.best_score_)
print('Best Hyperparameters:', search.best_params_)

# Predicting on test set
y_pred_xgboost = xgboost_regressor.predict(X_test)

# Creating CSV file accoring to Kaggle requirements
df = pd.DataFrame(y_pred_xgboost)
df.columns = ['SalePrice']
df = pd.concat([pd.read_csv('test.csv')['Id'], df], axis=1)
print(df.head())

# Exporting the predictions to CSV file
df.to_csv('prediction.csv', index=False)

# Creating a pickle file and dumping all 4 models to it
pickle_file = open('models.pkl', 'wb')
pickle.dump([linear_regressor, random_forest_regressor, support_vector_regressor, xgboost_regressor], pickle_file)
pickle_file.close()