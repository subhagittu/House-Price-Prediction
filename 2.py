import tensorflow as tf

import numpy as np

import seaborn as sns

import pandas as pd

import warnings

import matplotlib.pyplot as plt

import scipy.stats as stats

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam, SGD

from tensorflow_addons.metrics import RSquare

from keras.utils import set_random_seed

warnings.filterwarnings("ignore")

import math

import os

for dirname, _, filenames in os.walk('/july23/bharatintern/bharatintern2/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns',100)

train = pd.read_csv('/july23/bharatintern1/bharatintern2/train.csv')

train.head()

train.shape

train = train.drop('Id', axis=1)

train['MSSubClass'] = train['MSSubClass'].astype(object)

X_train=train.drop(['Id', 'SalePrice'], axis=1).copy()

y_train=train['SalePrice'].copy()

def missed_inf(data):

    missing={'Feature':data.columns,'Total_NA':data.isnull().sum(),'Proportion_NA':data.isnull().sum() / len(data)}
    missing=pd.DataFrame(missing)

    missing = missing.sort_values('Total_NA', ascending=False)

    return missing[missing['Total_NA']>0].style.hide_index()



missed_inf(X_train.select_dtypes(include=[int,float]))


X_train['LotFrontage'] = X_train['LotFrontage'].fillna(0)

X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)


X_train['GarageYrBlt']= X_train['GarageYrBlt'].fillna(-9999)



X_train['GarageYrBlt']= X_train['GarageYrBlt'].fillna(-9999)


N_feature=['Alley','MasVnrType','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish', 'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']


for col in N_feature:

    X_train[col]=X_train[col].fillna('NA')

X_train['Electrical']=X_train['Electrical'].fillna('SBrkr')


missed_inf(X_train)


X_train['TotalFlArea'] = X_train['1stFlrSF']+X_train['2ndFlrSF']

X_train['2ndFlRatio'] = X_train['2ndFlrSF']/X_train['1stFlrSF']

X_train["LivLotRatio"] = X_train['GrLivArea'] / X_train['LotArea']

X_train['Home_QualCond'] = (X_train['OverallQual'] + X_train['OverallCond'])/2

X_train['MeanRoomArea'] =X_train['TotalFlArea']/X_train['TotRmsAbvGrd']

X_train['AgeSold']=X_train['YrSold']-X_train['YearBuilt']

X_train['GetRemodelAdd']=X_train['YearRemodAdd']-X_train['YearBuilt']

s_range = [0, 2, 5, 8, 11, 12]

s_label = ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter']

X_train['SeasonSold'] = pd.cut(X_train['MoSold'], bins=s_range, labels=s_label, right=False, ordered=False)


Q__Values={'Ex':5, 'Gd':4, 'TA':3,'Fa':2, 'Po':1,'NA':0}

Q_Feature = ['ExterQual','ExterCond','BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond','PoolQC', 'KitchenQual']

for col in Q_Feature:

    X_train[col]=X_train[col].replace(Q__Values)


B__Values={'Gd':3, 'Av':2,'Mn':1, 'No':0,'NA':-1}

X_train['BsmtExposure']=X_train['BsmtExposure'].replace(B__Values)


CentralAir_values={'Y':1,'N':0}

X_train['CentralAir']=X_train['CentralAir'].replace(CentralAir_values)

X_train = pd.get_dummies(X_train)

X_train.head()

test = pd.read_csv('/july23/bharatintern1/bharatintern2/train.csv')

test.head()

X_train.shape

train['MSSubClass'] = train['MSSubClass'].astype(object)

print(set(test['MSSubClass'])-set(train['MSSubClass']))

print(set(train['MSSubClass'])-set(test['MSSubClass']))

test['MSSubClass'] = test['MSSubClass'].replace({150:50})

X_test = test.drop('Id', axis=1).copy()

X_test['MSSubClass'] = X_test['MSSubClass'].astype(str).astype(object)

X_test['LotFrontage'] = X_test['LotFrontage'].fillna(0)

X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)

X_test['GarageYrBlt']= X_test['GarageYrBlt'].fillna(-9999)

for col in N_feature:

    X_test[col]=X_test[col].fillna('NA')

missed_inf(X_test.select_dtypes(include=[int,float]))

X_test['GarageCars']=X_test['GarageCars'].fillna(0)

X_test['GarageArea']=X_test['GarageArea'].fillna(0)

X_test['BsmtHalfBath']=X_test['BsmtHalfBath'].fillna(0)

X_test['BsmtFullBath']=X_test['BsmtFullBath'].fillna(0)

X_test['BsmtUnfSF']=X_test['BsmtUnfSF'].fillna(0)

X_test['BsmtFinSF1']=X_test['BsmtFinSF1'].fillna(0)

X_test['BsmtFinSF2']=X_test['BsmtFinSF2'].fillna(0)

X_test['TotalBsmtSF']=X_test['TotalBsmtSF'].fillna(0)

missed_inf(X_test.select_dtypes(include=[int,float]))

missed_inf(X_test.select_dtypes(exclude=[int,float]))

for col in X_test.select_dtypes(exclude=[int,float]):

    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])

missed_inf(X_test.select_dtypes(exclude=[int,float]))


X_test['TotalFlArea'] = X_test['1stFlrSF']+X_test['2ndFlrSF']

X_test['2ndFlRatio'] = X_test['2ndFlrSF']/X_test['1stFlrSF']

X_test["LivLotRatio"] = X_test['GrLivArea'] / X_test['LotArea']

X_test['Home_QualCond'] = (X_test['OverallQual'] + X_test['OverallCond'])/2

X_test['MeanRoomArea'] =X_test['TotalFlArea']/X_test['TotRmsAbvGrd']

X_test['AgeSold']=X_test['YrSold']-X_test['YearBuilt']

X_test['GetRemodelAdd']=X_test['YearRemodAdd']-X_test['YearBuilt']

s_range = [0, 2, 5, 8, 11, 12]

s_label = ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter']

X_test['SeasonSold'] = pd.cut(X_test['MoSold'], bins=s_range, labels=s_label, right=False, ordered=False)



Q__Values={'Ex':5, 'Gd':4, 'TA':3,'Fa':2, 'Po':1,'NA':0}

Q_Feature = ['ExterQual','ExterCond','BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond','PoolQC', 'KitchenQual']

for col in Q_Feature:

    X_test[col]=X_test[col].replace(Q__Values)


B__Values={'Gd':3, 'Av':2,'Mn':1, 'No':0,'NA':-1}

X_test['BsmtExposure']=X_test['BsmtExposure'].replace(B__Values)


CentralAir_values={'Y':1,'N':0}

X_test['CentralAir']=X_test['CentralAir'].replace(CentralAir_values)


X_test = pd.get_dummies(X_test)

print('X_train shape----------------------------------->', X_train.shape)

print('X_test shape------------------------------------->', X_test.shape)

set(X_train.columns)-set(X_test.columns)


set(X_test.columns)-set(X_train.columns)

missing_columns=['Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','Electrical_Mix','Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','Heating_Floor','Heating_OthW','HouseStyle_2.5Fin','MiscFeature_TenC','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Utilities_NoSeWa']

X_test = X_test.assign(**{col: 0 for col in missing_columns})

set(X_train.columns)-set(X_test.columns)

print('X_train shape----------->', X_train.shape)

print('X_test shape------------>', X_test.shape)

X_test = X_test[X_train.columns]

X_train.head()

X_test.head()

X_train.to_csv('X_train.csv', index=False)

X_test.to_csv('X_test.csv',index=False)

y_train.to_csv('y_train.csv', index=False)

num_data = train.select_dtypes(include=[int,float])

num_data.head()

num_data.describe().T

cat_data = train.select_dtypes(exclude=[int,float])

cat_data.head()

correlation_matrix = train.corr()

correlation_matrix

plt.figure(figsize=(10, 8))

sns.heatmap(correlation_matrix, cmap='coolwarm')

plt.title('Correlation Matrix')

plt.show()


correlation = train.corrwith(train['SalePrice'])

correlation = correlation.sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.heatmap(correlation.to_frame(), annot=True, cmap='coolwarm')

plt.title('Correlation with SalePrice')

plt.show()


num_features = num_data.shape[1]

num_rows = (num_features + 2) // 3

num_cols = 3

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 30))

for i, col in enumerate(num_data.columns):

    ax = axes.flat[i]

    feature_data = num_data[col]

    sns.boxplot(y=feature_data, ax=ax)

    ax.set_title(f'Feature {col}', fontsize=16)

for j in range(num_features, num_rows * num_cols):

    axes.flat[j].axis('off')

plt.tight_layout()

plt.show()

num_features = num_data.shape[1]

num_rows = (num_features + 2) // 3

num_cols = 3

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 30))

for i, col in enumerate(num_data.columns):

    ax = axes.flat[i]

    feature_data = num_data[col]

    sns.histplot(feature_data, kde=True, ax=ax)

    ax.set_title(f'Feature {col}', fontsize=16)


for j in range(num_features, num_rows * num_cols):

    axes.flat[j].axis('off')

plt.tight_layout()

plt.show()


num_predictor = num_data.drop('SalePrice', axis=1)

num_features = num_predictor.shape[1]

num_rows = (num_features + 2) // 3

num_cols = 3

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 30))

for i, col in enumerate(num_predictor.columns):

    ax = axes.flat[i]

    sns.scatterplot(num_data, x=col, y='SalePrice', ax=ax)

    ax.set_title(f'Feature {col}', fontsize=16)

for j in range(num_features, num_rows * num_cols):

    axes.flat[j].axis('off')

plt.tight_layout()

plt.show()



cat_features = cat_data.shape[1]

num_rows = (cat_features + 2) // 3

num_cols = 3

rotate_indices = ['Exterior1st', 'Exterior2nd', 'Neighborhood']

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15,40))

for i, col in enumerate(cat_data.columns):

    ax = axes.flat[i]

    sns.boxplot(train, x=col, y='SalePrice', ax=ax)

    ax.set_title(f'Feature {col}', fontsize=16)

    if col in rotate_indices:

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

for j in range(cat_features, num_rows * num_cols):

    axes.flat[j].axis('off')

plt.tight_layout()

plt.show()

cat_data.columns

def missed_inf(data):

    missing={'Feature':data.columns,'Total_NA':data.isnull().sum(),'Proportion_NA':data.isnull().sum() / len(data)}

    missing=pd.DataFrame(missing)

    missing = missing.sort_values('Total_NA', ascending=False)

    return missing[missing['Total_NA']>0].style.hide_index()

missed_inf(train)

test = pd.read_csv('/july23/bharatintern1/bharatintern2/train.csv')

submit = pd.read_csv('/july23/bharatintern1/bharatintern2/submit.csv')



X_train = pd.read_csv('/july23/bharatintern1/bharatintern2/X_train.csv')

X_test = pd.read_csv('/july23/bharatintern1/bharatintern2/X_test.csv')

y_train = np.ravel(pd.read_csv('/july23/bharatintern1/bharatintern2/y_train.csv'))


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV


filemodel3 = RandomForestRegressor(random_state=87, n_estimators=100, max_depth=12,min_impurity_decrease=0.01)

filemodel3.fit(X_train, y_train)

filemodel3.score(X_val, y_val)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_pred=filemodel3.predict(X_val)

print('r2 score---------------------->',r2_score(y_true=y_val, y_pred=y_pred))

print('mean absolude error----------->',mean_absolute_error(y_true=y_val, y_pred=y_pred))

print('mean square error------------->',mean_squared_error(y_true=y_val, y_pred=y_pred))

print('root mean square error-------->',mean_squared_error(y_true=y_val, y_pred=y_pred, squared=False))

submit['Id'] = test.reset_index()['Id']

submit.SalePrice = filemodel3.predict(X_test)

submit.set_index('Id').to_csv('file3.csv')

print(submit)

import xgboost as xgb

filemodel4 = xgb.XGBRegressor(random_state=42,booster='gbtree',learning_rate=0.1,max_depth=3,n_estimators=300,reg_lambda=0.1)

filemodel4.fit(X_train, y_train)

filemodel4.score(X_val,y_val)

y_pred=filemodel4.predict(X_val)

print('r2 score---------------------->',r2_score(y_true=y_val, y_pred=y_pred))

print('mean absolude error----------->',mean_absolute_error(y_true=y_val, y_pred=y_pred))

print('mean square error------------->',mean_squared_error(y_true=y_val, y_pred=y_pred))

print('root mean square error-------->',mean_squared_error(y_true=y_val, y_pred=y_pred, squared=False))


submit['Id'] = test.reset_index()['Id']

submit.SalePrice = filemodel4.predict(X_test)

submit.set_index('Id').to_csv('file4.csv')

print(submit)

import lightgbm as lgb

filemodel1 = lgb.LGBMRegressor(random_state=42,learning_rate=0.1,max_depth=4,n_estimators=300,num_leaves=20,reg_lambda=0.1)

filemodel1.fit(X_train, y_train)

y_pred=filemodel1.predict(X_val)

print('r2 score---------------------->',r2_score(y_true=y_val, y_pred=y_pred))

print('mean absolude error----------->',mean_absolute_error(y_true=y_val, y_pred=y_pred))

print('mean square error------------->',mean_squared_error(y_true=y_val, y_pred=y_pred))

print('root mean square error-------->',mean_squared_error(y_true=y_val, y_pred=y_pred, squared=False))

submit['Id'] = test.reset_index()['Id']

submit.SalePrice = filemodel1.predict(X_test)

submit.set_index('Id').to_csv('file1.csv')

print(submit)



set_random_seed(42)

filemodel2 =  Sequential()

filemodel2.add(Dense(64, input_shape=(X_train.shape[1],),activation='relu',bias_regularizer='l2'))

filemodel2.add(Dense(64, activation='relu', bias_regularizer='l2'))

filemodel2.add(Dense(1))

early_stop = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=25)

filemodel2.compile(optimizer='adam',loss='mean_squared_error',metrics=[RSquare()])
filemodel2.fit(X_train, y_train,epochs=500,batch_size=16,callbacks=[early_stop],validation_data=(X_val, y_val))

filemodel2.evaluate(X_val, y_val)

y_pred=filemodel2.predict(X_val)

print('r2 score---------------------->',r2_score(y_true=y_val, y_pred=y_pred))

print('mean absolude error----------->',mean_absolute_error(y_true=y_val, y_pred=y_pred))

print('mean square error------------->',mean_squared_error(y_true=y_val, y_pred=y_pred))

print('root mean square error-------->',mean_squared_error(y_true=y_val, y_pred=y_pred, squared=False))

submit['Id'] = test.reset_index()['Id']

submit.SalePrice = filemodel2.predict(X_test)

submit.set_index('Id').to_csv('file2.csv')

print(submit)





