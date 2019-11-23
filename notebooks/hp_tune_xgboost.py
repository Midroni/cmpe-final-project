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


#Set up HP Search
params = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "alpha"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "grow_policy"      : ['depthwise','lossguide'] }

grid_search_dict =  {"learning_rate"    : list() ,
 "max_depth"        : list(),
 "min_child_weight" : list(),
 "gamma"            : list(),
 "alpha"            : list(),
 "colsample_bytree" : list(),
 "grow_policy"      : list(),
 "rmse" : list()}

#Perform Search
for lr in params['learning_rate']:
    for md in params['max_depth']:
        for cw in params['min_child_weight']:
            for g in params['gamma']:
                for a in params['alpha']:
                    for cbt in params['colsample_bytree']:
                        for gp in params['grow_policy']:
                            grid_search_dict['learning_rate'].append(lr)
                            grid_search_dict['max_depth'].append(md)
                            grid_search_dict['min_child_weight'].append(cw)
                            grid_search_dict['gamma'].append(g)
                            grid_search_dict['alpha'].append(a)
                            grid_search_dict['colsample_bytree'].append(cbt)
                            grid_search_dict['grow_policy'].append(gp)
                            
                            xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = cbt, 
                                                    learning_rate = lr, gamma = g,
                                                    max_depth = md, alpha = a, 
                                                    min_child_weight = cw,
                                                    grow_policy = gp,
                                                    n_estimators = 30)

                            xg_reg.fit(X_train, y_train)

                            preds = xg_reg.predict(X_test)
                            grid_search_dict['rmse'].append(np.sqrt(metrics.mean_squared_error(y_test, preds)))

#Pickle
df = pd.DataFrame().from_dict(grid_search_dict)
df.to_pickle('../data/hp_tune_xgboost.pkl')