# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
To build a complete model we need
DATA, MODEL, OBJECTIVE FUNCTION and OPTIMIZATION ALGORITHM
'''

'''DATA'''
# Generating random samples
n = 1000
x1 = np.random.uniform(low=-10, high=10,size=(n,1))
x2 = np.random.uniform(-10,10,(n,1))

# Creating column vectors
generated_inputs = np.column_stack((x1,x2))

#Generating noise
noise = np.random.uniform(-1,1,(n,1))

#Here, we define the target
generated_target = 2*x1 - 3*x2 + 5 + noise

# Saving the data in a compatible tf file
np.savez("tf_intro",input=generated_inputs,output=generated_target)

# Now we load the tf compatible file saved in "savez" command
training_data = np.load("tf_intro.npz")

# Now we create two variables that measure the input and output length
input_size = 2 # two variables, x1 and x2
output_size = 1 # only one y

'''MODEL'''
# Building the model for the respect output and standard tf parameters
#model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)])

# we can define our initializer parameters
model = tf.keras.Sequential([tf.keras.layers.Dense(output_size,
                                                   kernel_initializer= tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
                                                   bias_initializer= tf.random_uniform_initializer(minval=-0.1,maxval=0.1))])

''' OTIMIZER AND OBJECTIVE FUNCTION '''

#using standard tf paramenters
#model.compile(optimizer="sgd", loss="mean_squared_error")
#model.fit(training_data["inputs"],training_data["output"],epochs=100,verbose=1)

# we can define our learning rate by modifying "model.compile optimizer" argument
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer=custom_optimizer,loss="mean_squared_error")
model.fit(training_data["input"],training_data["output"],epochs=100,verbose=2)
# The model has just been fitted and now we can extract the weights and bias
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print(weights,bias)

# Now, with bias and weights, we can make predictions about possible
# outcomes
model.predict_on_batch(training_data["input"]).round(1)

#comparing with target
training_data["output"].round(1)

##Plotting output x target

plt.plot(np.squeeze(model.predict_on_batch(training_data["input"])),np.squeeze(training_data["output"]))
plt.xlabel("outputs")
plt.ylabel("targets")
plt.show()



