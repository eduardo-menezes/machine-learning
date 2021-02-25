import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

#Loading the data
inputs = pd.read_csv("entradas_breast.csv")
outputs = pd.read_csv("saidas_breast.csv")

# We need to build a function to pass as argument for KerasClassifier
def buildNN(optimizer,loss,kernel_initializer,activation,hidden_layers):
    input_size = 30
    outputs_size = 1
    model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layers, activation=activation, input_dim=input_size,
                                                       kernel_initializer=kernel_initializer),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(hidden_layers, activation=activation,kernel_initializer=kernel_initializer),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(outputs_size, activation=activation)
                                 ])
    # Compiling the model
    model.compile(optimizer=optimizer, loss=loss, metrics="binary_accuracy")
    return model

# The classifier itself

classifier = KerasClassifier(build_fn=buildNN)
parameters = {"batch_size":[10,30], "epochs":[50,100],"optmizer":["adam", "rgb"],
              "loss":["binary_crossentropy", "hinge"], "kernel_initializer":["random_uniform", "normal"],
              "activation":["relu", "tanh"], "hidden_layers":[8,16]}

grid_search = GridSearchCV(estimator=classifier, parameters=parameters, scoring="accuracy", cv=5)
grid_search = grid_search.fit(inputs,outputs)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_