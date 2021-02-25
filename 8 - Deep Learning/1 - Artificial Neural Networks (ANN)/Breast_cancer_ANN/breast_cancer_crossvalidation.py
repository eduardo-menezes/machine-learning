import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#Loading the data
inputs = pd.read_csv("entradas_breast.csv")
outputs = pd.read_csv("saidas_breast.csv")

# We need to build a function to pass as argument for KerasClassifier
def buildNN():
    input_size = 30
    outputs_size = 1
    hidden_layers = 16
    model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layers, activation="relu", input_dim=input_size,
                                                       kernel_initializer="random_uniform"),
                                 tf.keras.layers.Dense(hidden_layers, activation="relu"),
                                 tf.keras.layers.Dense(outputs_size, activation="sigmoid")
                                 ])
    # Defining the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)

    # Compiling the model
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics="binary_accuracy")
    return model

# The classifier itself

classifier = KerasClassifier(build_fn=buildNN,epochs = 100,batch_size=10)
results = cross_val_score(estimator=classifier,X=inputs,y=outputs, cv=10,scoring="accuracy")
print(results)
mean_results = results.mean()
std_results = results.std() #Higher values of std indicates overfitting
print(std_results)