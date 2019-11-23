import pandas as pd 
import numpy as np 
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics


#Create train-test splits
feat_df = pd.read_pickle('../data/features_df.pkl')
meta_df = pd.read_csv('../data/speechdetails.csv')
#Removes all the text that was in before
feat_df = feat_df.select_dtypes(exclude=['object'])
X = feat_df.values
y = meta_df['IC'].values

#scaler = MinMaxScaler()
#scaler.fit_transform(X)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2,
                                                    random_state=11)


#XGBOOST MODEL 1
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
print('Model 1 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

#XGBOOST MODEL 2
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 20)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
print('Model 2 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

#XGBOOST MODEL 3
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 30)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
print('Model 3 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

#XGBOOST MODEL 4
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 50)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
print('Model 4 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

#K-FOLD CROSS VALIDATION XGBOOST
data_dmatrix = xgb.DMatrix(data=X,label=y)


#XGBOOST CV MODEL 5
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10,'n_estimators':30}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print('CV Model 1 RMSE:', cv_results["test-rmse-mean"].tail(1).values[0])

#XGBOOST CV MODEL 6
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print('CV Model 2 RMSE:', cv_results["test-rmse-mean"].tail(1).values[0])

#XGBOOST CV MODEL 7
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=7,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print('CV Model 3 RMSE:', cv_results["test-rmse-mean"].tail(1).values[0])

#XGBOOST CV MODEL 8
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print('CV Model 4 RMSE:', cv_results["test-rmse-mean"].tail(1).values[0])

#RANDOM FOREST TRAIN
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

#RF 1
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('RF Model 1 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#RF 2
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('RF Model 2 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#RF 3
regressor = RandomForestRegressor(n_estimators=30, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('RF Model 3 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#RF 4
regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('RF Model 4 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
