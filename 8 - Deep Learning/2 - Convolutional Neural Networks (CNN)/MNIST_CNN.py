# Importing the libraries
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D,Dense
from keras.layers.normalization import BatchNormalization

# this dataset already contains train and test data
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# Now we need to preprocess the data
# Let's put the data in a compatible form with TensorFlow (Tensor form)
x_train_tf = x_train.reshape(x_train.shape[0],28,28,1)
x_test_tf = x_test.reshape(x_test.shape[0],28,28,1)

# we need to convert da input data into a float so we can normalize it for
# a RGB scale
x_train_tf = x_train_tf.astype("float32")
x_test_tf = x_test_tf.astype("float32")

# normalizing the data
x_train_tf_scaled = x_train_tf/255
x_test_tf_scaled = x_test_tf/255

# to predict, we need the dummy variables
y_train_dummy = np_utils.to_categorical(y_train,10) #y train is the records and 10 is the number of classes
y_test_dummy = np_utils.to_categorical(y_test,10)

# Building the model

# 1 - Convolutional Operator
model = tf.keras.Sequential()
model.add(Conv2D(32,kernel_size=(3,3), input_shape=(28,28,1), activation="relu"))

# 1.1 here we can normalize na features map to reduce the training time
model.add(BatchNormalization())

# 2 - Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Adding another conv layers
model.add(Conv2D(32,kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# 3 - Flatenning
model.add(tf.keras.layers.Flatten())

# Building the Dense layers
model.add(Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))

model.add(Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))

model.add(Dense(units=10,activation="softmax"))

# Compiling the model

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(x_train_tf_scaled,y_train_dummy,batch_size = 128, epochs = 5, validation_data=(x_test_tf_scaled, y_test_dummy))

# Results
results = model.evaluate(x_test_tf_scaled,y_test_dummy)
print(results)