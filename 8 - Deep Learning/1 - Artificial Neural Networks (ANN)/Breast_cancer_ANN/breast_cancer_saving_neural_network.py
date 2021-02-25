# importing libraries
import pandas as pd
import tensorflow as tf
import numpy as np

#Loading the data
inputs = pd.read_csv("entradas_breast.csv")
outputs = pd.read_csv("saidas_breast.csv")

input_size = 30
outputs_size = 1
hidden_layers = 8

#model building according to the best parameters found in tuning process
model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layers, activation="relu", input_dim=input_size,
                             kernel_initializer="normal"),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(hidden_layers, activation="relu", kernel_initializer="normal"),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(outputs_size, activation="sigmoid")])

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["binary_accuracy"])

# Training the model
model.fit(inputs,outputs,batch_size=10,epochs=100)

#saving the structure
compile_json = model.to_json()

# writing the neural networlk structure into a file
with open("structure_neural_network.json","w") as json_file:
    json_file.write(compile_json)

model.save_weights("weights_breast_cancer.h5")