# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Reading the dataset
df = pd.read_csv('train_preprocessed.csv')
print(df.head())

# Printing shape of training dataset
print(df.shape)

# Extracting X_train (independent features) and y_train (dependent_features)
X_train = df.drop(['Id', 'SalePrice'], axis=1)
y_train = df['SalePrice']
print('X_train Dimensions:', X_train.shape)
print('y_train Dimensions:', y_train.shape)

# Extracting some continous features to plot correlation matrix heatmap
continous_features = [feature for feature in X_train.columns if X_train[feature].nunique() > 25]

# Plotting the heatmap of correlation matrix for independent features
plt.figure(figsize=(14, 9))
sns.heatmap(X_train[continous_features].corr(), annot=True)
plt.title('HeatMap of Correlation Matrix for Independent Features')
plt.show()

# Training ExtraTreeRegressor model on train data and plotting the feature importances
extra_tree_regressor = ExtraTreesRegressor()
extra_tree_regressor.fit(X_train, y_train)

feature_importances = pd.Series(extra_tree_regressor.feature_importances_, X_train.columns).sort_values()
plt.figure(figsize=(14, 14))
sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
plt.title('Feature Importances based on ExtraTreeRegressor')
plt.xlabel('Feature Importance')
plt.show()

# Extracting the top 20 independent features
features_selected_extra_tree = list(feature_importances.index[-20:])
print('Top 20 Most Important Features are\n\n', features_selected_extra_tree)

# Training a feature selection model using Lasso and selectFromModel
feature_selection_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_selection_model.fit(X_train, y_train)

# Printing which independent features were dropped and which were kept
print(feature_selection_model.get_support())

# Extracting the selected features and printing there count
features_selected = X_train.columns[feature_selection_model.get_support()]
print('The selected features are', features_selected, '\n')
print('Total Number of Features in Dataset:', X_train.shape[1])
print('Total Number of Selected Features:', len(features_selected))
print('Total Number of Features with Coefficients Shrunk to Zero:', X_train.shape[1] - len(features_selected))

# Printing the selected features from both the models
print('From Lasso and selectFromModel\n\n', sorted(features_selected))
print('\nFrom ExtraTreeRegressor\n\n', sorted(features_selected_extra_tree))

# Dropping the less important features and keep the selected features only
df = pd.concat([df[features_selected], df['SalePrice']], axis=1)
print(df.head())

# Exporting the dataframe to CSV file
df.to_csv('train_final.csv', index=False)

# Importing the test data
df_test = pd.read_csv('test_preprocessed.csv')
print(df_test.head())

# Dropping feature 'Id'
df_test = df_test.drop(['Id'], axis=1)
print('Dataframe Shape:', df_test.shape)

# Dropping the less important features and keep the selected features only
df_test = df_test[features_selected]
print('Dataframe Shape:', df_test.shape)

# Exporting the dataframe to CSV file
df_test.to_csv('test_final.csv', index=False)