# Importing necessary libraries
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Reading the dataset
df = pd.read_csv('train.csv', index_col='Id')
print(df.head())

# Printing shape of training dataset
print(df.shape)

# Calculating percentage of NaN values (in columns with NaN values present) in numerical features
features_with_nan = [feature for feature in df.columns if df[feature].isnull().sum() > 0 and df[feature].dtype != 'O']

for feature in features_with_nan:
    print(feature, ':', round(df[feature].isnull().mean() * 100, 2), '%')

# Replacing NaN values with the median of that feature
for feature in features_with_nan:
    if feature == 'LotFrontage' or feature == 'GarageYrBlt':
        df[feature+'_NaN'] = np.where(df[feature].isnull(), 1, 0)
    df[feature] = df[feature].fillna(df[feature].median())
    
# Printing number of NaN values in numerical features after handling them
print(df[features_with_nan].isnull().sum())

# Printing updated dataframe
print(df.head())

# Calculating percentage of NaN values (in columns with NaN values present) in categorical features
features_with_nan = [feature for feature in df.columns if df[feature].isnull().sum() > 0 and df[feature].dtype == 'O']

for feature in features_with_nan:
    print(feature, ':', round(df[feature].isnull().mean() * 100, 2), '%')

# Replacing NaN values with a new label called 'Missing'
df[features_with_nan] = df[features_with_nan].fillna('Missing')
print(df[features_with_nan].isnull().sum())

# Extracting features with years 
year_features = [feature for feature in df.columns if ('Yr' in feature or 'Year' in feature) and 'NaN' not in feature]
print(year_features)

# Converting year into age for first 3 features (in year_features) by taking difference between 'YrSold' and the feature
for feature in year_features[:-1]:
    df[feature] = df['YrSold'] - df[feature]
print(df.head())

# Printing specific columns from dataframe after updation
print(df[year_features].head())

# Extracting continous numerical features 
continous_numerical_features = [feature for feature in df.columns 
                                if df[feature].dtype != 'O' and feature not in year_features and df[feature].nunique() >= 25]

for feature in continous_numerical_features:
    if 0 not in df[feature].unique():
        df[feature] = np.log(df[feature])
        
print(df.head())

# Extracting categorical variables
# No need to account for year_features as they are a number now
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O'] 
print(categorical_features)

# Replacing rare categories in all categorical features by label 'Rare_Var'
for feature in categorical_features:
    temp = df.groupby(feature)['SalePrice'].count()/len(df)
    indices = temp[temp > 0.01].index
    df[feature] = np.where(df[feature].isin(indices), df[feature], 'Rare_Var')

print(df.head(10))

# Applying label encoder on each categorical feature to convert it from string to number
for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
print(df.head())

# Extracting X_train and y_train variables from the dataframe
X_train = df.drop(['SalePrice'], axis=1)
y_train = df[['SalePrice']]
print(y_train.head())

# Scaling features using MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=y_train.index)
print(X_train.head())

# Dumping the MinMaxScaler learnt to a pickle file
pickle_file = open('minmax_scaler.pkl', 'wb')
pickle.dump(scaler, pickle_file)
pickle_file.close()

# Concatenating X_train and y_train
df = pd.concat([X_train, y_train], axis=1)

# Exporting Train Data to CSV file
df.to_csv('train_preprocessed.csv')

# Importing test data
df_test = pd.read_csv('test.csv', index_col='Id')
print(df_test.head())

# Printing shape of test data
print(df_test.shape)

# Calculating percentage of NaN values (in columns with NaN values present) in numerical features
features_with_nan = [feature for feature in df_test.columns 
                     if df_test[feature].isnull().sum() > 0 and df_test[feature].dtype != 'O']

for feature in features_with_nan:
    print(feature, ':', round(df_test[feature].isnull().mean() * 100, 2), '%')

# Replacing NaN values with the median of that feature
for feature in features_with_nan:
    if feature == 'LotFrontage' or feature == 'GarageYrBlt':
        df_test[feature+'_NaN'] = np.where(df_test[feature].isnull(), 1, 0)
    df_test[feature] = df_test[feature].fillna(df_test[feature].median())
    
# Printing number of NaN values in numerical features after handling them
print(df_test[features_with_nan].isnull().sum())

# Calculating percentage of NaN values (in columns with NaN values present) in categorical features
features_with_nan = [feature for feature in df_test.columns 
                     if df_test[feature].isnull().sum() > 0 and df_test[feature].dtype == 'O']

for feature in features_with_nan:
    print(feature, ':', round(df_test[feature].isnull().mean() * 100, 2), '%')

# Replacing NaN values with a new label called 'Missing'
df_test[features_with_nan] = df_test[features_with_nan].fillna('Missing')
print(df_test[features_with_nan].isnull().sum())

# Extracting features with years 
year_features = [feature for feature in df_test.columns if ('Yr' in feature or 'Year' in feature) and 'NaN' not in feature]
print(year_features)

# Converting year into age for first 3 features (in year_features) by taking difference between 'YrSold' and the feature
for feature in year_features[:-1]:
    df_test[feature] = df_test['YrSold'] - df_test[feature]
print(df_test.head())

# Printing specific columns from dataframe after updation
print(df_test[year_features].head())

# Extracting continous numerical features 
continous_numerical_features = [feature for feature in df_test.columns 
                                if df_test[feature].dtype != 'O' and feature not in year_features and df_test[feature].nunique() >= 25]

for feature in continous_numerical_features:
    if 0 not in df_test[feature].unique():
        df_test[feature] = np.log(df_test[feature])
        
print(df_test.head())

# Extracting categorical variables
# No need to account for year_features as they are a number now
categorical_features = [feature for feature in df_test.columns if df_test[feature].dtype == 'O'] 
print(categorical_features)

# Replacing rare categories in all categorical features by label 'Rare_Var'
for feature in categorical_features:
    temp = df_test.groupby(feature)[feature].count()/len(df_test)
    indices = temp[temp > 0.01].index
    df_test[feature] = np.where(df_test[feature].isin(indices), df_test[feature], 'Rare_Var')

print(df_test.head(10))

# Applying label encoder on each categorical feature to convert it from string to number
for feature in categorical_features:
    encoder = LabelEncoder()
    df_test[feature] = encoder.fit_transform(df_test[feature])

# Scaling features using MinMaxScaler
df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
print(df_test.head())

# Exporting Test Data to CSV file
df_test.to_csv('test_preprocessed.csv')