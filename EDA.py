import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

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

df = pd.get_dummies(df, drop_first=True)

# Finish dealing with NAs
# LotFrontage 259 NAs
# MasVnrArea 8 NAs
# GarageYrBlt 81 NAs

X = np.array(df.drop(columns='SalePrice'))
y = np.array(df['SalePrice'])
