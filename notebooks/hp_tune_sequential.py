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


#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2,
                                                    random_state=11)

#Set up HP Search
params = {
     'layer1' : [16, 8, 4],
     'layer2' : [16, 8, 4],
     'layer3' : [8, 4],
     'layer4' : [8, 4],
     'epochs': [150],
     'optimizer': ['adam', 'adagrad','RMSprop'],
     'losses': ['binary_crossentropy','mse']}

grid_search_dict =  {
     'layer1' : list(),
     'layer2' : list(),
     'layer3' : list(),
     'layer4' : list(),
     'epochs': list(),
     'optimizer': list(),
     'losses': list(),
     'rmse': list()}

count = 0
#Perform Search
for l1 in params['layer1']:
    for l2 in params['layer2']:
        for l3 in params['layer3']:
            for l4 in params['layer4']:
                for ep in params['epochs']:
                    for op in params['optimizer']:
                        for los in params['losses']:
                            grid_search_dict['layer1'].append(l1)
                            grid_search_dict['layer2'].append(l2)
                            grid_search_dict['layer3'].append(l3)
                            grid_search_dict['layer4'].append(l4)
                            grid_search_dict['epochs'].append(ep)
                            grid_search_dict['optimizer'].append(op)
                            grid_search_dict['losses'].append(los)

                            import keras
                            from keras.models import Sequential
                            from keras.layers import Dense

                            model = Sequential()
                            model.add(Dense(16, input_dim=len(X[0]), activation='relu'))
                            model.add(Dense(l1, activation='relu'))
                            model.add(Dense(l2, activation='relu'))
                            model.add(Dense(l3, activation='relu'))
                            model.add(Dense(l4, activation='relu'))
                            model.add(Dense(1, activation='linear'))
                            model.compile(loss=los, optimizer=op)
                            model.fit(X_train, y_train, epochs=ep, verbose=0)

                            preds = model.predict(X_test)
                            rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
                            count+=1
                            print("Model",count)
                            grid_search_dict['rmse'].append(rmse)

#Pickle
df = pd.DataFrame().from_dict(grid_search_dict)
df.to_pickle('../data/hp_tune_sequential.pkl')