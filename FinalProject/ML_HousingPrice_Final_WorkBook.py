#!/usr/bin/env python
# coding: utf-8

# House price predictions:
# 
# 1. Data cleaning and preprocessing
# 
# Modeling:
# 2. Baseline regressions
# 3. Random Forest
# 3. Improvements- hyperparameter tuning
# 4. Bagging
# 5. Boosting
# 6. Neural network
# 7. SVM
# 

# In[ ]:





# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRFRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Get datasets


# In[108]:


df_sub = pd.read_csv('/sample_submission.csv')
df_train = pd.read_csv('/train.csv') 
df_test = pd.read_csv('/test.csv')


# STEP1: Data cleaning and visualizations

# a) Continuous features VS target label - Sales Price

# In[109]:


num_columns = [col for col in df_train.columns if (df_train[col].dtype == 'int64' or df_train[col].dtype == 'float64') and (col !='Id') ]
df_col = df_train[num_columns]
for i in range(0, len(num_columns), 5):
        sns.pairplot(data=df_col,
                    x_vars=df_col.columns[i:i+5],
                    y_vars=['SalePrice'])


# In[110]:


#Heatmap to compare each feature to every other feature
num_columns = [col for col in df_train.columns if (df_train[col].dtype == 'int64' or df_train[col].dtype == 'float64') and (col !='Id') ]
corr = df_train[num_columns].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.figure(figsize=(18,8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={"shrink": .8}, vmin=0, vmax=1)
plt.show()


# b) Visualizing missing data

# In[111]:


missing = df_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[112]:


df_train_nu = (df_train.isnull().sum() / len(df_train)) * 100
df_train_nu = df_train_nu.drop(df_train_nu[df_train_nu == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_train_nu})
missing_data.head(100)


# c) Handling missing data

# i) Removing data columns that is majorly missing

# In[113]:


print("Total records:",df_train.shape[0])
print("Large portion of data is missing for following features:")
print("PoolQC", df_train.PoolQC.isnull().sum())
print("MiscFeature",df_train['MiscFeature'].isnull().sum())
print("Alley",df_train['Alley'].isnull().sum())
print("Fence",df_train['Fence'].isnull().sum())
print("FireplaceQu",df_train['FireplaceQu'].isnull().sum())


# In[114]:


#Removing the above columns/features from both train and test
df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis = 1, inplace = True )

df_test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis = 1, inplace = True )


# ii) Data imputation

# With imputation need to ensure the data distribution is not hampered by filling in for those missing data.
# Below an example is shown for one feature where the distribution before and after imputation remain undisturbed.

# In[115]:


#Before impute
ax = sns.distplot(df_train['LotFrontage'])


# In[116]:


df_train['LotFrontage'].fillna(df_train.LotFrontage.median(), inplace = True) #Impute
ax = sns.distplot(df_train['LotFrontage'])


# Impute for other missing values

# In[117]:


#Training set imputation
df_train['GarageType'].fillna(df_train['GarageType'].mode()[0], inplace = True)
df_train['GarageYrBlt'].fillna(df_train.GarageYrBlt.median(), inplace = True)
df_train['GarageFinish'].fillna(df_train.GarageFinish.mode()[0], inplace = True)
df_train['GarageQual'].fillna(df_train.GarageQual.mode()[0], inplace = True)
df_train['GarageCond'].fillna(df_train.GarageCond.mode()[0], inplace = True)
df_train['BsmtExposure'].fillna(df_train.BsmtExposure.mode()[0], inplace = True)
df_train['BsmtFinType2'].fillna(df_train.BsmtFinType2.mode()[0], inplace = True)
df_train['BsmtQual'].fillna(df_train.BsmtQual.mode()[0], inplace = True)
df_train['BsmtCond'].fillna(df_train.BsmtCond.mode()[0], inplace = True)
df_train['BsmtFinType1'].fillna(df_train.BsmtFinType1.mode()[0], inplace = True)
df_train['MasVnrType'].fillna(df_train.BsmtFinType1.mode()[0], inplace = True)
df_train['MasVnrArea'].fillna(df_train.MasVnrArea.median(), inplace = True)
df_train['Electrical'].fillna(df_train.MasVnrArea.median(), inplace = True)


# In[ ]:


#Checking- no more missing data in training set


# In[118]:


df_train_nu = (df_train.isnull().sum() / len(df_train)) * 100
df_train_nu = df_train_nu.drop(df_train_nu[df_train_nu == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_train_nu})
missing_data.head(100)


# In[119]:


#Test set imputation

df_test['LotFrontage'].fillna(df_test.LotFrontage.median(), inplace = True)
df_test['GarageType'].fillna(df_test['GarageType'].mode()[0], inplace = True)
df_test['GarageYrBlt'].fillna(df_test.GarageYrBlt.median(), inplace = True)
df_test['GarageFinish'].fillna(df_test.GarageFinish.mode()[0], inplace = True)
df_test['GarageQual'].fillna(df_test.GarageQual.mode()[0], inplace = True)
df_test['GarageCond'].fillna(df_test.GarageCond.mode()[0], inplace = True)
df_test['BsmtExposure'].fillna(df_test.BsmtExposure.mode()[0], inplace = True)
df_test['BsmtFinType2'].fillna(df_test.BsmtFinType2.mode()[0], inplace = True)
df_test['BsmtQual'].fillna(df_test.BsmtQual.mode()[0], inplace = True)
df_test['BsmtCond'].fillna(df_test.BsmtCond.mode()[0], inplace = True)
df_test['BsmtFinType1'].fillna(df_test.BsmtFinType1.mode()[0], inplace = True)
df_test['MasVnrArea'].fillna(df_test.MasVnrArea.median(), inplace = True)
df_test['Electrical'].fillna(df_test.Electrical.mode()[0], inplace = True)
df_test['MasVnrType'].fillna(df_test.MasVnrType.mode()[0], inplace = True)
df_test['BsmtFullBath'].fillna(df_test.BsmtFullBath.mode()[0], inplace = True)
df_test['BsmtHalfBath'].fillna(df_test.BsmtHalfBath.mode()[0], inplace = True)
df_test['BsmtFinSF1'].fillna(df_test.BsmtFinSF1.median(), inplace = True)
df_test['GarageCars'].fillna(df_test.GarageCars.mode()[0], inplace = True)
df_test['GarageArea'].fillna(df_test.GarageArea.mean(), inplace = True)
df_test['TotalBsmtSF'].fillna(df_test.TotalBsmtSF.mode()[0], inplace = True)
df_test['BsmtFinSF2'].fillna(df_test.BsmtFinSF2.mode()[0], inplace = True)
df_test['BsmtUnfSF'].fillna(df_test.BsmtUnfSF.median(), inplace = True)
df_test['MSZoning'].fillna(df_test.MSZoning.mode()[0], inplace = True)
df_test['Functional'].fillna(df_test.Functional.mode()[0], inplace = True)
df_test['Utilities'].fillna(df_test.Utilities.mode()[0], inplace = True)
df_test['Exterior2nd'].fillna(df_test.Exterior2nd.mode()[0], inplace = True)
df_test['Exterior1st'].fillna(df_test.Exterior1st.mode()[0], inplace = True)
df_test['SaleType'].fillna(df_test.SaleType.mode()[0], inplace = True)
df_test['KitchenQual'].fillna(df_test.KitchenQual.mode()[0], inplace = True)


# In[120]:


df_test_nu = (df_test.isnull().sum() / len(df_test)) * 100
df_test_nu = df_test_nu.drop(df_test_nu[df_test_nu == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_test_nu})
missing_data.head(100)


# d) Handling categorical variables- Using Label encoder

# In[ ]:


#Train set


# In[121]:



df_train.select_dtypes(include=['object'])
df_train_object=df_train.select_dtypes(include=['object'])
df_train_object.columns


# In[122]:


object_cols=('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition')


# In[123]:


for i in object_cols:
    lb_encoder = LabelEncoder() 
    lb_encoder.fit(list(df_train[i].values)) 
    df_train[i] = lb_encoder.transform(list(df_train[i].values))


# In[124]:


#Checking for object type data - index is empty
df_train_object=df_train.select_dtypes(include=['object'])
df_train_object.columns


# In[ ]:


#Test Set


# In[125]:


df_test.select_dtypes(include=['object'])
df_test_object=df_test.select_dtypes(include=['object'])
df_test_object.columns


# In[126]:


object_cols=('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition')


# In[127]:


for i in object_cols:
    lb_encoder = LabelEncoder() 
    lb_encoder.fit(list(df_test[i].values)) 
    df_test[i] = lb_encoder.transform(list(df_test[i].values))


# In[128]:


df_test_object=df_test.select_dtypes(include=['object'])

df_test_object.columns


# In[ ]:





# **STEP 2:  Modeling **

# **REGRESSION MODELS**

# In[129]:


#Data Splitting

x_data=df_train.drop('SalePrice',axis=1)
y_data=df_train['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)


# In[130]:


#Linear regression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = LinearRegression()
model.fit(x_train , y_train)
predictions = model.predict(x_test)

print('Mean Absolute Error(MAE) test:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Absolute Error(MAE) train:', metrics.mean_absolute_error(y_train, model.predict(x_train)))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,predictions))
print('R2:',metrics.r2_score(y_test, predictions))


# In[131]:


#Decision Tree
model_DecisionTree = DecisionTreeRegressor()
model_DecisionTree.fit(x_train , y_train)
train_pred=model_DecisionTree.predict(x_train)

DecisionTree_predictions = model_DecisionTree.predict(x_test)

print('Mean Absolute Error(MAE) test:', metrics.mean_absolute_error(y_test, DecisionTree_predictions))
print('Mean Absolute Error(MAE) train:', metrics.mean_absolute_error(y_train, model_DecisionTree.predict(x_train)))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, DecisionTree_predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, DecisionTree_predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,DecisionTree_predictions))
print('R2:',metrics.r2_score(y_test, DecisionTree_predictions))


# Advanced Techniques are explored
# 
# - Bagging
# - Random Forest
# - Boosting
# 

# In[133]:


#Bagging
# define dataset
#x_train , y_train = make_regression( n_features=, n_informative=15, noise=0.1, random_state=5)


model_BR = BaggingRegressor()
model_BR.fit(x_train , y_train)
train_pred=model_BR.predict(x_train)
yhat = model_BR.predict(x_test)

print('Mean Absolute Error(MAE) Test:', metrics.mean_absolute_error(y_test, yhat))
print('Mean Absolute Error(MAE) train:', metrics.mean_absolute_error(y_train, model_BR.predict(x_train)))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, yhat))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, yhat)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,yhat))
print('R2:',metrics.r2_score(y_test, yhat))
print('R2 rounded:',(metrics.r2_score(y_test, yhat)).round(2))


# In[135]:


#Random Forest
model_RandomForest = RandomForestRegressor()
model_RandomForest.fit(x_train , y_train)
RandomForest_predictions = model_RandomForest.predict(x_test)

print('Mean Absolute Error(MAE) Test:', metrics.mean_absolute_error(y_test, RandomForest_predictions))
print('Mean Absolute Error(MAE) Train:', metrics.mean_absolute_error(y_train, model_RandomForest.predict(x_train)))


print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, RandomForest_predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, RandomForest_predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,RandomForest_predictions))
print('R2:',metrics.r2_score(y_test, RandomForest_predictions))
print('R2 rounded:',(metrics.r2_score(y_test, RandomForest_predictions)).round(2))


# Hyper parameter tuning for Random Forest Regressor

# In[137]:


#N_estimaotrs

n_estimator=[10,20,50,100,500]
# for loop to iterate for each leaf size
for n_est in n_estimator :
    print("N_est is:",n_est)
    model = RandomForestRegressor(n_estimators = n_est)
    model_RandomForest.fit(x_train , y_train)
    RandomForest_predictions = model_RandomForest.predict(x_test)
    print('Mean Absolute Error(MAE) Test:', metrics.mean_absolute_error(y_test, RandomForest_predictions))
    print('Mean Absolute Error(MAE) Train:', metrics.mean_absolute_error(y_train, model_RandomForest.predict(x_train)))

    print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, RandomForest_predictions))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, RandomForest_predictions)))
    print('Explained Variance Score (EVS):',explained_variance_score(y_test,RandomForest_predictions))
    print('R2:',metrics.r2_score(y_test, RandomForest_predictions))
    print('R2 rounded:',(metrics.r2_score(y_test, RandomForest_predictions)).round(2))



# In[138]:


#Max_Depth

max_depth=[2,5,10,12,15,20,25]
# for loop to iterate for each leaf size
for md in max_depth :
    print("md is:",md)
    model = RandomForestRegressor(n_estimators = 500, max_depth=md)
    model_RandomForest.fit(x_train , y_train)
    RandomForest_predictions = model_RandomForest.predict(x_test)
    print('Mean Absolute Error(MAE) Test:', metrics.mean_absolute_error(y_test, RandomForest_predictions))
    print('Mean Absolute Error(MAE) Train:', metrics.mean_absolute_error(y_train, model_RandomForest.predict(x_train)))

    print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, RandomForest_predictions))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, RandomForest_predictions)))
    print('Explained Variance Score (EVS):',explained_variance_score(y_test,RandomForest_predictions))
    print('R2:',metrics.r2_score(y_test, RandomForest_predictions))
    print('R2 rounded:',(metrics.r2_score(y_test, RandomForest_predictions)).round(2))


# In[ ]:





# For best value from above estimates- n_est=500 and md=15

# In[139]:


#Min_Sample_Leaf
min_samples_leaf=[1,2,5,10,12,15,20,25]
# for loop to iterate for each leaf size
for mn in min_samples_leaf :
    print("min_samples_leaf is:",mn)
    model = RandomForestRegressor(n_estimators = 500, max_depth=15, min_samples_leaf=mn)
    model_RandomForest.fit(x_train , y_train)
    RandomForest_predictions = model_RandomForest.predict(x_test)
    print('Mean Absolute Error(MAE) Test:', metrics.mean_absolute_error(y_test, RandomForest_predictions))
    print('Mean Absolute Error(MAE) Train:', metrics.mean_absolute_error(y_train, model_RandomForest.predict(x_train)))

    print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, RandomForest_predictions))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, RandomForest_predictions)))
    print('Explained Variance Score (EVS):',explained_variance_score(y_test,RandomForest_predictions))
    print('R2:',metrics.r2_score(y_test, RandomForest_predictions))
    print('R2 rounded:',(metrics.r2_score(y_test, RandomForest_predictions)).round(2))


    


# Best MAE=15222 is found at n_est=500, max_depth=15, min_leaf=10
# 

# In[ ]:



#XGboost regressor
model = XGBRFRegressor(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x_train , y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# In[176]:


#GradientBoostingRegressor
gbr_params = {'n_estimators': 1000,
          'max_depth': 3,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
gbr = GradientBoostingRegressor(**gbr_params)
gbr.fit(x_train, y_train)
ygbr=gbr.predict(x_test)

print("Model Accuracy Test: %.3f" % gbr.score(x_test, y_test))
print("Model Accuracy Train: %.3f" % gbr.score(x_train, y_train))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, ygbr))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, ygbr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, ygbr)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,ygbr))
print('R2:',metrics.r2_score(y_test, ygbr))
print('R2 rounded:',(metrics.r2_score(y_test, ygbr)).round(2))


# The MAE is greatly reduced to 14454 from previous regression models.
# Also the accuracy scores for train Vs test is less determining there is no overfit issues.
# 

# **NEURAL NETWORK**

# In[142]:


col_list1 = ['SalePrice','MSSubClass','MSZoning','LotFrontage','LotArea','Street','YearBuilt','LotShape','1stFlrSF','2ndFlrSF']
col_list2 = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','YearBuilt','LotShape','1stFlrSF','2ndFlrSF']
pricing_data = pd.read_csv('/train.csv',usecols=col_list1).dropna()
pricing_test_data = pd.read_csv('/test.csv',usecols=col_list2).dropna()


# In[143]:


import datetime
pricing_data['Total Years'] = datetime.datetime.now().year - pricing_data['YearBuilt']
pricing_data.drop('YearBuilt',axis=1,inplace=True)


# In[144]:


cat_features = ['MSSubClass','MSZoning','Street','LotShape']
label_encoder ={}

for feature in cat_features:
    label_encoder[feature]=LabelEncoder()
    pricing_data[feature]= label_encoder[feature].fit_transform(pricing_data[feature])

pricing_data.head()


# In[145]:


cat_features = np.stack([pricing_data['MSSubClass'],pricing_data['MSZoning'],pricing_data['Street'],pricing_data['LotShape']],1)
cat_features


# In[146]:


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as f


# In[147]:


## Converting numpy to tensors

cat_features = torch.tensor(cat_features,dtype=torch.int64)
cat_features


# In[148]:


## Creating Continous Variables
cont_features =[]
for i in pricing_data.columns:
    if i in ['MSSubClass','MSZoning','Street','LotShape','SalePrice']:
        pass
    else:
        cont_features.append(i)

cont_values = np.stack([pricing_data[i].values for i in cont_features],axis = 1)
cont_values = torch.tensor(cont_values,dtype=torch.float)
len(cont_features)


# In[149]:


## Dependent Feature
y = torch.tensor(pricing_data['SalePrice'].values,dtype=torch.float).reshape(-1,1)
y


# In[150]:


cat_features.shape,cont_values.shape,y.shape


# Embedding Categorical Features into the Neural Network
# While embedding the categorical features, we need to know the exact
# input and output dimensions of the features
# Here 'MSSubClass','MSZoning','Street','LotShape' has output dimensions as [15,5,2,4]
# respectively

# In[151]:


cat_dims = [len(pricing_data[col].unique()) for col in ['MSSubClass','MSZoning','Street','LotShape']]
## output dimensions should be set based on the input dimensions (min)
embedding_dims = [(x, min(50,(x+1)//2)) for x in cat_dims]
embedding_dims


# In[152]:


embed_representation = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dims])
embed_representation


# In[153]:


embedding_val=[]
for i,e in enumerate(embed_representation):
    embedding_val.append(e(cat_features[:,i]))
embedding_val


# In[154]:


z = torch.cat(embedding_val,1)
z


# In[155]:


# Dropout layer -> Regularization method
dropout=nn.Dropout(0.4)
final_embed = dropout(z)
final_embed


# In[156]:


## Creating the neural network
class FeedForwardNN(nn.Module):
    def __init__(self,embedding_dim,n_cont,out_sz,layers,p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((out for inp,out in embedding_dim))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in =i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self,x_cat,x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings,1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x,x_cont],1)
        x = self.layers(x)
        return x


# In[157]:


len(cont_features)


# In[158]:


torch.manual_seed(100)
price_model = FeedForwardNN(embedding_dims,len(cont_features),1,[100,50],p=0.4)
price_model


# In[159]:


loss_ftn = nn.MSELoss()
optimizer = torch.optim.Adam(price_model.parameters(),lr = 0.01)


# In[160]:


pricing_data.shape


# In[161]:


batch_size = 1200
test_size = int(batch_size*0.15)
train_categorical = cat_features[:batch_size - test_size]
test_categorical = cat_features[batch_size - test_size: batch_size]
train_cont = cont_values[:batch_size - test_size]
test_cont = cont_values[batch_size - test_size: batch_size]

y_train = y[:batch_size - test_size]
y_test = y[batch_size - test_size:batch_size]


# In[162]:


len(train_categorical),len(test_categorical),len(train_cont),len(test_cont),len(y_train),len(y_test)


# In[163]:


epochs = 5000
final_losses = []
for i in range(epochs):
    i = i+1
    y_pred = price_model(train_categorical,train_cont)
    loss = torch.sqrt(loss_ftn(y_pred,y_train)) ## RMSE
    final_losses.append(loss)
    if i%10 == 1:
        print("Epoch number: {} and the loss: {}".format(i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# In[164]:


mylosses=torch.Tensor(final_losses)
loss1 = mylosses.detach().numpy()
plt.plot(range(epochs),list(loss1))
plt.ylabel('RMSE Loss')
plt.xlabel('epoch')


# In[165]:


# Validation of the test data
y_pred = ""
with torch.no_grad():
    y_pred = price_model(test_categorical,test_cont)
    loss = torch.sqrt(loss_ftn(y_pred,y_test))
print('RMSE: {}'.format(loss))


# In[166]:


data_verify = pd.DataFrame(y_test.tolist(),columns=['Test'])
data_predicted = pd.DataFrame(y_pred.tolist(),columns=['Prediction'])
data_predicted


# In[ ]:


torch.save(price_model,'houseprice.pt')


# **SVM**

# In[167]:


from sklearn import svm


# In[172]:


#Data Splitting

x_data=df_train.drop('SalePrice',axis=1)
y_data=df_train['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)


# In[173]:


model_svm = svm.SVC(kernel='linear')
model_svm.fit(x_train , y_train)
predictions = model_svm.predict(x_test)

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, predictions))
print('Mean Absolute Error(MAE): on training', metrics.mean_absolute_error(y_train, model_svm.predict(x_train)))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,predictions))
print('R2:',metrics.r2_score(y_test, predictions))


# In[174]:


model = svm.SVR()
model.fit(x_train , y_train)
predictions = model.predict(x_test)

#confusion_matrix(y_test, predictions)

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,predictions))
print('R2:',metrics.r2_score(y_test, predictions))


# In[175]:


model = svm.SVC(kernel='rbf')
model.fit(x_train , y_train)
predictions = model.predict(x_test)

#confusion_matrix(y_test, predictions)

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,predictions))
print('R2:',metrics.r2_score(y_test, predictions))

