# Importing the libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

#Loading the data
inputs = pd.read_csv("entradas_breast.csv")
outputs = pd.read_csv("saidas_breast.csv")

# Splitting the data into train and test data automatically
inputs_train, inputs_test, outputs_train,outputs_test = train_test_split(inputs,outputs, test_size=0.25)

# building the model
input_size = 30
outputs_size = 1
hidden_layers = 16

model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layers,activation="relu", input_dim=input_size, kernel_initializer = "random_uniform"),
                             tf.keras.layers.Dense(hidden_layers, activation="relu"),
                             tf.keras.layers.Dense(outputs_size, activation="sigmoid")
                             ])
# Defining the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001,clipvalue=0.5)

#Compiling the model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics="binary_accuracy")

# Fitting the model
batch_size = 10
epochs = 100
model.fit(inputs_train,outputs_train,batch_size=batch_size,epochs=epochs)

# Using the method to get the weights
weights = model.layers[2].get_weights()
print(weights)

''' # Testing the model
predictions = model.predict(inputs_test)
predictions = [1 if item > 0.5 else 0 for item in predictions]
accuracy = accuracy_score(outputs_test,predictions)
matrix = confusion_matrix(outputs_test,predictions)
results = model.evaluate(inputs_test,outputs_test)
'''







