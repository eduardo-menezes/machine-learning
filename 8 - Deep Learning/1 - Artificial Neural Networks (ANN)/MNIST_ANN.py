''' Let's define our steps to analyze MNIST
    1 - Prepare the data and Pre-process them
    2 - Create training, validation and testing data
    3 - Define the model
    4 - Choose a activation function
    5 - Define the optimizer
    6 - Define the loss function
    7 - Make it LEARN
    8 - Test model accuracy '''

# Importing the libraries
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

'''LOADING AND PRE PROCESSING THE DATA'''

# Loading the data
mnist_dataset,mnist_info = tfds.load(name="mnist", with_info=True, as_supervised=True)

# Extracting training and test data
mnist_train,mnist_test = mnist_dataset["train"], mnist_dataset["test"]

# We need to define the validation data based in the biggest sample,
# the train sample.
num_validation_samles = 0.1 * mnist_info.splits["train"].num_examples

# As we can see, the num_validation_samles might be a float, so we have to
# convert it to a integer
num_validation_samles = tf.cast(num_validation_samles, tf.int64)

# Now, let's store the number of  test samples in a proper variable
num_test_samples = mnist_info.splits["test"].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)

# It's good to standardize the data to avoid numerical errors.
def scale(image,label):
    image = tf.cast(image,tf.float32)
    image = image/255.
    return image,label

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# Let's shuffle the samples to maintain informations
buffer_size = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(buffer_size)

# Extracting validation data
validation_data = shuffled_train_and_validation_data.take(num_validation_samles)
train_data = shuffled_train_and_validation_data.skip(num_validation_samles)

#Definig the batch
batch_size = 100
train_data = train_data.batch(batch_size)
validation_data = validation_data.batch(num_validation_samles)
test_data = test_data.batch(num_test_samples)
validation_inputs,validation_targets = next(iter(validation_data))

''' DEFINING THE MODEL AND ACTIVATION FUNCTION'''

#Let's outline the model
input_size = 784 #input layer
output_size = 10 #output layer
hidden_layer_size = 100

#The model itself
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28,1)),
                             tf.keras.layers.Dense(hidden_layer_size,activation="relu"),
                             tf.keras.layers.Dense(hidden_layer_size,activation="relu"),
                             tf.keras.layers.Dense(output_size,activation="softmax")
                            ])
''' DEFINING THE OPTIMIZER AND LOSS FUNCTION '''
# Let's define the opmizer with the compile command
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

''' TRAINING THE DATA '''
num_epochs = 6
model.fit(train_data,epochs=num_epochs,validation_data=(validation_inputs,validation_targets),verbose=2)

''' TESTING THE MODEL '''
test_loss,test_accuracy = model.evaluate(test_data)
print(test_loss,test_accuracy)