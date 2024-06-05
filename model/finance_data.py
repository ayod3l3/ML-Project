import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sklearn
import pandas as pd


#Import Training data
training_data=pd.read_csv('/Users/ayodeleogundele/Desktop/data_science/ANN_pjt/data/as1-bank.csv')
print(training_data)

#Check for missing values
missing_values = training_data.isnull().sum()
print(missing_values) 

#Separate targe variable from dataset and store in "training_y" 
training_y = training_data.pop('y')

#Convert target variable to binary  
training_y.replace("yes", 1, inplace=True)
training_y.replace("no", 0, inplace=True)

#consider this as well
#training_y = training_y.replace("yes", 1).replace("no", 0)

#Preview changes 
training_y

#Assign training data to variable training_x
training_x = training_data
print(training_x.columns)

#data transformation 
#Id columns that need to convert categorical strings to binary 
cat_columns = ['default', 'housing', 'loan']

training_x[cat_columns] = training_x[cat_columns].replace({'no': 0, 'yes': 1})

#verify transformation 
training_x

#Split data into training and test data for classification model development 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(training_x, training_y, test_size=0.3, random_state=33, shuffle=True)

print(X_train.shape)
print(X_test.shape)


#Convert Pandas to Numpy array for ML
arr_train_x=training_x.to_numpy()
arr_train_y=training_y.to_numpy()

print(arr_train_x.shape)
print(arr_train_y.shape)


print(arr_train_x)
print(arr_train_y) 

#Define model 
model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(13,))) #There 13 columns 
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='SGD',
    metrics=['accuracy']
)


#train model
model_training_history=model.fit(
    arr_train_x,
    arr_train_y,
    epochs=50
)

#Evaluate model
print(model_training_history)
model_training_history

test_loss,test_acc= model.evaluate(X_test, y_test)

import matplotlib.pyplot as plt

#plot vis. accuracy vs. loss
fig, (ax1, ax2)=plt.subplots(2, figsize=(8,6))
acc=model_training_history['accuracy']
loss=model_training_history['loss']

ax1.plot(acc)
ax2.plot(loss)

ax1.set_ylabel('training accuracy')
ax2.set_xlabel('training loss')

#both share same axis 
ax2.xlabel('epochs')

# Get accuracy and loss values from model_training_history
acc = model_training_history.history['accuracy']
loss = model_training_history.history['loss']

# Plot accuracy and loss
fig, (ax1, ax2)=plt.subplots(2, figsize=(8,6))
acc=model_training_history.history['accuracy']
loss=model_training_history.history['loss']

ax1.plot(acc)
ax2.plot(loss)

ax1.set_ylabel('training accuracy')
ax2.set_ylabel('trining loss')

#they both share one axis definded at the bottom
ax2.set_xlabel('epochs')

