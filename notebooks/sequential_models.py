import pandas as pd 
import numpy as np 
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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

def plot_loss_and_metric(histories, epochs_after=0):
    loss_values = histories.history['loss'][epochs_after:]
    val_loss_values = histories.history['val_loss'][epochs_after:]
    
    epochs = range(1, len(loss_values)+1)
    
    plt.plot(epochs,loss_values,'b',val_loss_values,'b:')
    #fig = plt.figure(figsize=(12, 5))
    #fig.plot(epochs, loss_values, 'b', label='Training loss')
    #fig.plot(epochs, val_loss_values, 'b:', label='Validation loss')
    #fig.set_title('Training and validation loss')
    #fig.set_xlabel('Epochs')
    #fig.set_ylabel('Loss')
    #fig.legend()
    plt.show()

#Set the random seed to ensure reproducible results
from tensorflow import set_random_seed
set_random_seed(11)

#SEQUENTIAL MODEL 1
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=len(X[0]), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, 
                    epochs=100,
                    validation_data=(X_train,y_train),
                    verbose=0)

#for key in history.history.keys():
#    print(history.history[key])
#plot_loss_and_metric(history, 0)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Model 1 RMSE: %f" % (rmse))

#SEQUENTIAL MODEL 2
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=len(X[0]), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X_train, y_train, epochs=100, verbose=0)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Model 2 RMSE: %f" % (rmse))

#SEQUENTIAL MODEL 3
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=len(X[0]), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, verbose=0)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Model 3 RMSE: %f" % (rmse))

#SEQUENTIAL MODEL 4
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=len(X[0]), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, verbose=0)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Model 4 RMSE: %f" % (rmse))