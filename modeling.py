import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error

############################## Clean data ##############################
df = pd.read_csv('train.csv')

df['MSSubClass'] = df['MSSubClass'].astype('O')
df['LandSlope'].replace(['Sev', 'Mod', 'Gtl'], np.arange(3), inplace=True)
df['BsmtExposure'].replace([np.NaN, 'No', 'Mn', 'Av', 'Gd'], np.arange(5), inplace=True)
df['Functional'].replace(['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], np.arange(8), inplace=True)
df['GarageFinish'].replace([np.NaN, 'Unf', 'RFn', 'Fin'], np.arange(4), inplace=True)
df['PavedDrive'].replace(['N', 'P', 'Y'], np.arange(3), inplace=True)
po_ex = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
df[po_ex] = df[po_ex].replace([np.NaN, 'Po', 'Fa', 'TA', 'Gd', 'Ex'], np.arange(6))

df = pd.get_dummies(df)

df['LotFrontage'].fillna(0, inplace=True)
df['MasVnrArea'].fillna(0, inplace=True)
df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)


# Finish dealing with NAs
# LotFrontage 259 NAs
# MasVnrArea 8 NAs
# GarageYrBlt 81 NAs

############################## Model data ##############################
X = np.array(df.drop(columns=['Id', 'SalePrice']))
y = np.log(np.array(df['SalePrice']))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
lasso = LassoCV(random_state=42)
m = Pipeline([('scaler', scaler), ('m', lasso)])
m.fit(X_train, y_train)

print('Best alpha: {}'.format(m['m'].alpha_))
print('Train accuracy: {}'.format(m.score(X_train, y_train)))
print('Test accuracy: {}'.format(m.score(X_test, y_test)))

pred = m.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(y_test, pred))
print('Root mean squared log error: {}'.format(rmsle))
