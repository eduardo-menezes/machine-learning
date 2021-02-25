# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Loading the data
data = pd.read_csv("iris.csv")

# Defining inputs and outputs
inputs = data.iloc[:,0:4].values
outputs = data.iloc[:,-1].values

# AS we can see, we need to map the categorical data into a numerical data
from sklearn.preprocessing import LabelEncoder

# Let's create an object from this class
label_encoder = LabelEncoder()
outputs = label_encoder.fit_transform(outputs)

# As we can see, "outputs" must be as a dammy variable to satisfy compatibility
from keras.utils import np_utils
outputs_dummy = np_utils.to_categorical(outputs)
print(outputs_dummy)

# Defining train and test data
input_train,input_test,output_train,output_test = train_test_split(inputs,outputs_dummy,test_size=0.25)

# Building the model
input_size = 4
output_size=3
hidden_layers=5
model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layers,activation="relu",input_dim=input_size),
                             tf.keras.layers.Dense(hidden_layers,activation="relu"),
                             tf.keras.layers.Dense(output_size,activation="softmax")])

# Compiling the model
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["categorical_accuracy"])

#Training the model
model.fit(input_train,output_train,batch_size=10,epochs=1000)

# Testing the model
results = model.evaluate(input_test,output_test)

# Making prediction
predictions = model.predict(input_test)
print(predictions)

# Modifing the data to a one-dimensional array that contains the indice os the predicted highest value (0|1)
output_test_new = [np.argmax(i) for i in output_test]
predictions_new = [np.argmax(i) for i in predictions]

# Analising the confusion matrix to get insights
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(output_test_new,predictions_new)
print(matrix)