# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Loading the data
data = pd.read_csv("iris.csv")

# Defining inputs and outputs
inputs = data.iloc[:,0:4].values
outputs = data.iloc[:,-1].values

# AS we can see, we need to map the categorical data into a numerical data
# Let's create an object from this class
label_encoder = LabelEncoder()
outputs = label_encoder.fit_transform(outputs)

# As we can see, "outputs" must be as a dammy variable to satisfy compatibility
outputs_dummy = np_utils.to_categorical(outputs)
print(outputs_dummy)

def buildNN():
    input_size = 4
    output_size = 3
    hidden_layers = 5
    model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layers, activation="relu", input_dim=input_size),
                                 tf.keras.layers.Dense(hidden_layers, activation="relu"),
                                 tf.keras.layers.Dense(output_size, activation="softmax")])

    # Compiling the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model

model = KerasClassifier(buildNN, epochs = 1000, batch_size=10)
results = cross_val_score(estimator=model, X=inputs, y=outputs, scoring="accuracy", cv=10)
mean_accurancy = results.mean()
std_accurancy = results.std()

print(mean_accurancy,std_accurancy)


