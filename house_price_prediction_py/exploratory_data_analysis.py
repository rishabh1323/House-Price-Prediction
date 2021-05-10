# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# Reading the dataset
df = pd.read_csv('train.csv', index_col='Id')
print(df.head())

# Plotting a heatmap of density of NaN values present in the dataframe
plt.figure(figsize=(20, 6))
sns.heatmap(df.isnull(), yticklabels=False)
plt.title('Heat map showing density of NaN values in the dataframe')
plt.show()

# Calculating percentage of NaN values (in columns with NaN values present)
features_with_nan = []

for feature in df.columns:
    if(df[feature].isnull().sum() != 0):
        features_with_nan.append(feature)
        print(feature, ':', round(df[feature].isnull().mean() * 100, 2), '%')

# Checking if features with NaN values present has some relationship with the dependent feature or not
temp_df = df.copy()

for feature in features_with_nan:
    temp_df[feature] = np.where(temp_df[feature].isnull(), 1, 0)
    
    temp_df.groupby(feature)['SalePrice'].median().plot.bar(rot=0, color=['blue', 'orange'])
    plt.title('Barplot for SalePrice (Median) vs {}'.format(feature))
    plt.ylabel('Median Sale Price')
    plt.show()

# Extracting numerical features (feature datatype != object)
numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print(df[numerical_features].head())

# Extracting features with Year information (Date-Time variables)
year_features = [feature for feature in df.columns if 'Yr' in feature or 'Year' in feature]
print('Datetime type features:' ,year_features)

# Analyzing datetime features
df.groupby('YrSold')['SalePrice'].median().plot()
plt.title('Median Sale Price vs Year Sold')
plt.ylabel('Median Sale Price')
plt.show()

# Converting 'Year' features into 'Age' features by taking difference from the 'YrSold' feature and then analyzing them
temp_df = df.copy()
for feature in year_features:
    if feature != 'YrSold':
        temp_df[feature] = temp_df['YrSold'] - temp_df[feature]
        plt.scatter(temp_df[feature], temp_df['SalePrice'])
        plt.title(f'SalePrice vs (YrSold - {feature})')
        plt.xlabel(f'(YrSold - {feature})')
        plt.ylabel('SalePrice')
        plt.show()

# Extracting discrete numerical features (number of unique values < 25 and not datetime features)
discrete_numerical_features = [feature for feature in numerical_features 
                               if len(df[feature].unique()) < 25 and feature not in year_features]
print('Number of Discrete Numerical Features :', len(discrete_numerical_features), '\n')
print(discrete_numerical_features)

# Exploring relationship between discrete numerical features and output feature 'SalePrice'
for feature in discrete_numerical_features:
    df.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title('Median SalePrice vs {}'.format(feature))
    plt.ylabel('Median SalePrice')
    plt.show()

# Extracting continous numerical features (number of unique values >= 25 and not datetime features)
continous_numerical_features = [feature for feature in numerical_features 
                               if feature not in discrete_numerical_features and feature not in year_features]
print('Number of Continous Numerical Features :', len(continous_numerical_features), '\n')
print(continous_numerical_features)

# Exploring relationship between continous numerical features and output feature 'SalePrice' using histograms
for feature in continous_numerical_features:
    df[feature].hist(bins=50)
    plt.title('Histogram for {}'.format(feature))
    plt.ylabel('Count')
    plt.xlabel(feature)
    plt.show()

# Applying logarithm transformation on the continous numerical features
temp_df = df.copy()
for feature in continous_numerical_features:
    if 0 in temp_df[feature].unique():
        pass
    else:
        temp_df[feature] = np.log(temp_df[feature])
        plt.scatter(temp_df[feature], np.log(temp_df['SalePrice']))
        plt.xlabel('{} (log scale)'.format(feature))
        plt.ylabel('SalePrice (log scale)')
        plt.title('SalePrice vs {} (After Logarithmic transformation)'.format(feature))
        plt.show()

# Plotting heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df[continous_numerical_features].corr(), annot=True)
plt.title('HeatMap for Correlation Matrix of Continous Numerical Features')
plt.show()

# Plotting boxplots for continous numerical features to find outliers
for feature in continous_numerical_features:
    plt.figure(figsize=(5, 5))
    sns.boxplot(y=temp_df[feature])
    plt.xlabel(feature)
    if 0 in temp_df[feature].unique():
        plt.title(f'Box Plot for {feature}')
    else:
        plt.title(f'Box Plot for {feature} (Log Scale)')
    plt.show()

# Extracting numerical features (feature datatype == object)
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print(df[categorical_features].head())

# Finding cardinality of categorical features (number of unique categories in the feature)
for feature in categorical_features:
    print(f'{feature} : {df[feature].nunique()}')

# Plotting barplots to find relationship between categorical features and dependent feature 'SalePrice'
for feature in categorical_features:
    plt.figure(figsize=(7, 5))
    df.groupby(feature)['SalePrice'].median().plot.bar(rot=0)
    plt.title(f'SalePrice vs {feature}')
    plt.ylabel('SalePrice')
    plt.xlabel(feature)
    plt.show()