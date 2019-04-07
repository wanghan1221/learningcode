# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:48:40 2018

@author: lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.stats import norm, skew
import numpy as np
import seaborn as sns
import warnings
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
#读取数据
wd = 'C:/Users/lenovo/Desktop/codes/machine-learning/kaggle/houseprice/'
df_train = pd.read_csv(str(wd)+'train.csv')
df_test = pd.read_csv(str(wd)+'test.csv')
#查看变量名
df_train.columns
var=pd.DataFrame({'name':df_train.columns,'type':df_train.dtypes})
#var.to_csv(str(wd)+'variable.csv')
#----------------------------------------------part1 描述性分析----------------------------------------------------
#房价的描述性分析
df_train['SalePrice'].describe()
sns.distplot(np.log(df_train['SalePrice']+1))
df_train['LogSalePrice'] = np.log1p(df_train['SalePrice'])
plt.subplot(1,2,1)
sns.distplot(df_train['SalePrice'])#直方图看出正偏态
plt.subplot(1,2,2)
sns.distplot(df_train['LogSalePrice'])#对sales进行log变换后服从正态分布
#----------------------------连续变量
#SalePrice相关系数热力图
corrmat = df_train.corr()
k=10
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
sns.heatmap(df_train[cols].corr(),annot=True)
#各变量之间相关系数热力图
sns.heatmap(corrmat)
#salprice与相关性较强的变量的散点图
for i in range(1,len(cols)):
   plt.figure(figsize=(15,8))
   plt.scatter(df_train[cols[i]],df_train['SalePrice'])

#GrLivArea剔除异常值
df_train =df_train[df_train['GrLivArea']<4500]
df_train.drop(['Id'],axis=1,inplace=True)        
df_test.drop(['Id'],axis=1,inplace=True)     

y_train = df_train.LogSalePrice
features_train = df_train.drop(['SalePrice','LogSalePrice'],axis=1)
features_test = df_test
features = pd.concat([features_train,features_test]).reset_index(drop=True)

#-----------------------------------------------缺失值处理-------------------------------------------------------------
nulls = np.sum(features.isnull(),axis=0)
nullcols = nulls.loc[nulls!=0]    
types = features.dtypes
nulltypes = types.loc[nulls!=0]
info = pd.concat([nullcols,nulltypes],axis=1).sort_values(by = 0,ascending = False)
#无相关变量的变量缺失值处理
features['Functional'].value_counts()#Typ频数最高
features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'].value_counts()#SBrkr频数最高
features['Electrical'] = features['Electrical'].fillna('SBrkr')

features['KitchenQual'].value_counts()#TA频数最高
features['KitchenQual'] = features['KitchenQual'].fillna('TA')

features['Exterior2nd'].value_counts()#VinylSd频数最高
features['Exterior2nd'] = features['Exterior2nd'].fillna('VinylSd')

features['Exterior1st'].value_counts()#VinylSd频数最高
features['Exterior1st'] = features['Exterior1st'].fillna('VinylSd')

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features['Utilities'].value_counts()
features['Utilities'] = features['Utilities'].fillna('AllPub')

#相关变量的变量缺失值处理
pd.set_option('max_columns',None)
features.loc[(features['PoolArea']>0) & (features['PoolQC'].isna())]
features.loc[2418, 'PoolQC'] = 'Fa'
features.loc[2501, 'PoolQC'] = 'Gd'
features.loc[2597, 'PoolQC'] = 'Fa'

garage_col = ['GarageFinish','GarageCond','GarageYrBlt','GarageType','GarageQual']
features_garage = features[garage_col]
features_garage[np.sum(features_garage.isnull(),axis=1)==5]#157行5个变量都为空，则不填补
features[(features['GarageType'] == 'Detchd') & (features['GarageYrBlt'].isna())]
sns.boxplot(features_garage['GarageType'],features_garage['GarageYrBlt'])
garage_Detch = features[features['GarageType'] == 'Detchd']
features.loc[2124, 'GarageYrBlt'] = garage_Detch['GarageYrBlt'].median()
features.loc[2574, 'GarageYrBlt'] = garage_Detch['GarageYrBlt'].median()
features.loc[2124, 'GarageFinish'] = garage_Detch['GarageFinish'].mode()[0]
features.loc[2574, 'GarageFinish'] = garage_Detch['GarageFinish'].mode()[0]
features.loc[2574, 'GarageCars'] = garage_Detch['GarageCars'].median()
features.loc[2124, 'GarageArea'] = garage_Detch['GarageArea'].median()
features.loc[2574, 'GarageArea'] = garage_Detch['GarageArea'].median()
features.loc[2124, 'GarageQual'] = garage_Detch['GarageQual'].mode()[0]
features.loc[2574, 'GarageQual'] = garage_Detch['GarageQual'].mode()[0]
features.loc[2124, 'GarageCond'] = garage_Detch['GarageCond'].mode()[0]
features.loc[2574, 'GarageCond'] = garage_Detch['GarageCond'].mode()[0]

basement_col = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF']
feature_basement = features[basement_col]
feature_basement[(np.sum(feature_basement.isnull(),axis=1)<5)&(np.sum(feature_basement.isnull(),axis=1)>0)]
plt.subplot(1,2,1)
sns.boxplot(feature_basement['BsmtFinType1'],feature_basement['BsmtFinSF1'])
plt.subplot(1,2,2)
sns.boxplot(feature_basement['BsmtFinType2'],feature_basement['BsmtFinSF2'])
features.loc[332, 'BsmtFinType2'] = 'ALQ'
features.loc[947, 'BsmtExposure'] = 'No' 
features.loc[1485, 'BsmtExposure'] = 'No'
feature_basement['BsmtCond'].value_counts()
features.loc[2215, 'BsmtQual'] = 'Po' #v small basement so let's do Poor.
features.loc[2216, 'BsmtQual'] = 'Fa' #similar but a bit bigger.
features.loc[2346, 'BsmtExposure'] = 'No' #unfinished bsmt so prob not.
features.loc[2522, 'BsmtCond'] = 'Gd' #cause ALQ for bsmtfintype1

features.groupby('MSSubClass')['MSZoning'].agg(lambda x : x.mode()[0])#apply也可以
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

features[(features['MasVnrType'].isnull())&(features['MasVnrArea'].notnull())]
sns.boxplot(features['MasVnrType'],features['MasVnrArea'])
features.groupby('MasVnrArea')['MasVnrType'].agg(lambda x: x.mode()[0])
features.loc[2608,'MasVnrType'] = 'Stone'

#分类变量的填充，包括前面未补全的变量
objects = []
for i in features.columns:
    if features[i].dtype == 'object':
        objects.append(i)
features.update(features[objects].fillna('none'))        

#未补全的数值型变量
nulls = np.sum(features.isnull(),axis = 0)
nullcols = nulls.loc[nulls != 0]
types = features.dtypes.loc[nulls != 0]
info = pd.concat([nullcols,types],axis = 1).sort_values(by = 0,ascending = False)
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

features[(features['GarageYrBlt'].isnull())&(features['GarageArea']>0)]

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes: 
        numerics.append(i)
features.update(features[numerics].fillna(0))

features.describe()
#GarageYrBlt最大值为2207，异常值
features.loc[features['GarageYrBlt'] == 2207]#第2590行
features.loc[2590,'GarageYrBlt'] = 2007

features.update(features['MSSubClass'].astype('str'))

#查看数值变量的偏度
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_cols =[]
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numeric_cols.append(i)
skew = features[numeric_cols].apply(lambda x:x.skew()).sort_values(ascending=False)
#对skew超过0.5进行box_cox变换,这里需要验证是否考虑左偏分布的变量，是进行boxcox变换还是log变换
high_skew = skew[skew > 0.5]
high_skew_index = high_skew.index
for i in high_skew_index:
    features[i] = boxcox1p(features[i],boxcox_normmax(features[i]+1))

#查看分类变量类别数
objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
object_classes = features[objects].apply(lambda x: len(x.unique())).sort_values(ascending=False)
features[object_classes[object_classes<4].index].apply(lambda x:x.value_counts())
#Utilities&Street基本上只有一个值，故删掉这俩变量
features.drop(['Utilities','Street'],inplace=True,axis=1)

#变量整合
features['Total_Squarefoot'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])
features['Total_Bathrooms'] = (features['FullBath'] + (0.5*features['HalfBath']) + 
                               features['BsmtFullBath'] + (0.5*features['BsmtHalfBath']))
features['Total_porch'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'])
#新建分类变量
features['haspool'] = features['PoolArea'].apply(lambda x:1 if x>0 else 0)#是否有泳池
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x:1 if x>0 else 0)#是否有二楼
features['hasgarage'] = features['GarageArea'].apply(lambda x:1 if x>0 else 0)#是否有停车场
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x:1 if x>0 else 0)#是否有地下室
features['hasfireplace'] = features['Fireplaces'].apply(lambda x:1 if x>0 else 0)#是否有壁橱

#哑变量化
final_features = pd.get_dummies(features).reset_index(drop=True)
x_train = final_features[:len(y_train)]
x_test = final_features[len(y_train):]

#将某一属性值占比超过99.95%的变量剔除（可以看做某变量所有值都相同）
one_class = []
for var in x_train.columns:
    if (x_train[var].value_counts().iloc[0])/len(x_train[var])>0.9994:
        one_class.append(var)
x_train.drop(one_class,axis = 1,inplace = True)
x_test.drop(one_class,axis = 1,inplace = True)
#对于连续变量可以用VarianceThreshold函数，可以自动删除方差过小的变量
from sklearn.feature_selection import VarianceThreshold
continous_col = []
cols = x_train.columns
for col in cols:
    if x_train[col].dtype != 'object':
        continous_col.append(col)
continous_data = x_train[continous_col]
vt = VarianceThreshold()
vt_continous = vt.fit_transform(continous_data)
print(vt.variances_)

#----------------------------------------------建立模型--------------------------------------------------------------
kfolds = KFold(n_splits = 10, shuffle = True, random_state = 23)
def cv_rmse(model):
    rse = np.sqrt(-cross_val_score(model,x_train,y_train,cv=kfolds,scoring="neg_mean_squared_error"))
    return rse

benchmark_model = make_pipeline(RobustScaler(),LinearRegression()).fit(x_train,y_train)
cv_rmse(benchmark_model).mean()
coeffs = pd.DataFrame(list(zip(x_train.columns,benchmark_model.steps[1][1].coef_)),columns = ['predictors','coefficients'])
coeffs = coeffs.sort_values(by = 'coefficients',ascending=False)
#岭回归
def ridge_selector(k):
    ridge_model = make_pipeline(RobustScaler(),RidgeCV(alphas = [k],cv = kfolds)).fit(x_train,y_train)
    ridge_rmse = cv_rmse(ridge_model).mean()
    return(ridge_rmse)
ridge_scores = []
ridge_alpha = [.0001, .0003, .0005, .0007, .0009, .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]
for alpha in ridge_alpha:
    score = ridge_selector(alpha)
    ridge_scores.append(score)
plt.plot(ridge_alpha,ridge_scores,label = 'ridge')
ridge_alpha[ridge_scores.index(min(ridge_scores))]
alphas_alt = [19.5, 19.6, 19.7, 19.8, 19.9, 20, 20.1, 20.2, 20.3, 20.4, 20.5]
ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas = alphas_alt,cv=kfolds)).fit(x_train, y_train)
cv_rmse(ridge_model).mean()
ridge_model.steps[1][1].alpha_#19.5

#lasso
alphas = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,0.0006, 0.0007, 0.0008]
lasso_model=make_pipeline(RobustScaler(),LassoCV(alphas=alphas,cv=kfolds)).fit(x_train,y_train)
scores = lasso_model.steps[1][1].mse_path_
plt.plot(alphas,scores)
lasso_model.steps[1][1].alpha_#0.005
coeffs = pd.DataFrame(list(zip(x_train.columns,lasso_model.steps[1][1].coef_)),columns = ['predictor','coeffiecient'])
used_coeffs = coeffs[coeffs['coeffiecient'] != 0].sort_values(by = 'coeffiecient',ascending=False)
#由lasso选出的变量，要求每个变量同一属性值不能超过99.5%
used_coeffs_values = x_train[used_coeffs['predictor']]
overfit_cols = []
for i in used_coeffs_values.columns:
    if used_coeffs_values[i].value_counts().iloc[0]/len(used_coeffs_values[i])>0.995:
        overfit_cols.append(i)

#弹性网
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
elastic_model = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio = e_l1ratio,alphas =e_alphas,max_iter =1e7,cv=kfolds)).fit(x_train,y_train)
cv_rmse(elastic_model).mean()#0.10708
elastic_model.steps[1]
#------------------xgb
rcParams['figure.figsize'] = 12, 4
import xgboost as xgb
from xgboost import XGBRegressor
xgb = XGBRegressor(learning_rate =0.01, n_estimators=3460, max_depth=3,
                     min_child_weight=0 ,gamma=0, subsample=0.7,
                     colsample_bytree=0.7,objective= 'reg:linear',
                     nthread=4,scale_pos_weight=1,seed=27, reg_alpha=0.00006)
xgb_fit = xgb.fit(x_train, y_train)

#----------------------svm
from sklearn import svm
svr_opt = svm.SVR(C = 100000, gamma = 1e-08)
svr_fit = svr_opt.fit(x_train, y_train)
cv_rmse(svr_fit).mean

#-----------------LGBMRegressor
from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgbm_fit = lgbm_model.fit(x_train, y_train)
cv_rmse(lgbm_fit).mean

#-----------------stacking model
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline

#setup models
ridge = make_pipeline(RobustScaler(), 
                      RidgeCV(alphas = alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas2,
                              random_state = 42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio))

lightgbm = make_pipeline(RobustScaler(),
                        LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11))

xgboost = make_pipeline(RobustScaler(),
                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 
                                     max_depth=3,min_child_weight=0 ,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective= 'reg:linear',nthread=4,
                                     scale_pos_weight=1,seed=27, 
                                     reg_alpha=0.00006))


#stack
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, 
                                            xgboost, lightgbm), 
                               meta_regressor=xgboost,
                               use_features_in_secondary=True)

#prepare dataframes
stackX = np.array(x_train)
stacky = np.array(y_train)
stack_gen_model = stack_gen.fit(stackX, stacky)
em_preds = elastic_model3.predict(testing_features)
lasso_preds = lasso_model2.predict(testing_features)
ridge_preds = ridge_model2.predict(testing_features)
stack_gen_preds = stack_gen_model.predict(testing_features)
xgb_preds = xgb_fit.predict(testing_features)
svr_preds = svr_fit.predict(testing_features)
lgbm_preds = lgbm_fit.predict(testing_features)