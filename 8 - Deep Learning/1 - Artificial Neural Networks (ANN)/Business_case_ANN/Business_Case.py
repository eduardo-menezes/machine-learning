## Importing the libraries
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

#Extracting the inputs and targets
raw_csv_data = np.loadtxt("audiobooks_data.csv",delimiter=",")
unscaled_inputs_all = raw_csv_data[:,1:-1] # data withou the first and last column
targets_all = raw_csv_data[:,-1]

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

# For loop to count the number of zeros AND BALANCE THE DATA
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter+=1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

# Balancing the data
unscaled_inputs_equal = np.delete(unscaled_inputs_all,indices_to_remove,axis=0)
targets_equal = np.delete(targets_all,indices_to_remove, axis=0)

# Standardizing the inputs
scaled_inputs = preprocessing.scale(unscaled_inputs_equal)

# Now, shuffling the data to provide equality information
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal[shuffled_indices]

# Now, we need to split the data into train, validation and test data
# First, let's define the size of each sample
samples_count = shuffled_inputs.shape[0]
train_sample_count = int(0.8*samples_count)
validation_sample_count = int(0.1*samples_count)
test_sample_count = samples_count - train_sample_count - validation_sample_count
print(train_sample_count,validation_sample_count,test_sample_count)

# Now, we split the data according to each size
train_inputs = shuffled_inputs[:train_sample_count]
train_targets = shuffled_targets[:train_sample_count]

validation_inputs = shuffled_inputs[train_sample_count:train_sample_count+validation_sample_count]
validation_targets = shuffled_targets[train_sample_count:train_sample_count+validation_sample_count]

test_inputs = shuffled_inputs[train_sample_count+validation_sample_count:]
test_targets = shuffled_targets[train_sample_count+validation_sample_count:]

# Verifying if the data is balanced
print(np.sum(train_targets/train_sample_count), np.sum(validation_targets)/validation_sample_count,
      np.sum(test_targets/test_sample_count))

# Now, we save the dataset in compatibility with TensorFlow
np.savez("audiobook_data_train", inputs = train_inputs, targets = train_targets)
np.savez("audiobook_data_validation",inputs = validation_inputs, targets = validation_targets)
np.savez("audiobook_data_test", inputs = test_inputs, targets = test_targets)

#Loading the preprocessed data
npz = np.load("audiobook_data_train.npz")
train_inputs = npz["inputs"].astype(np.float)
test_targets = npz["targets"].astype(np.int)

npz = np.load("audiobook_data_validation.npz")
validation_inputs,validation_targets = npz["inputs"].astype(np.float), npz["targets"].astype(np.int)

npz = np.load("audiobook_data_test.npz")
test_inputs,test_targets = npz["inputs"].astype(np.float), npz["targets"].astype(np.int)

# Building the model
input_size = 10
output_size = 2
hidden_layer = 100

model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layer,activation="relu"),
                             tf.keras.layers.Dense(hidden_layer,activation="relu"),
                            tf.keras.layers.Dense(output_size,activation="softmax")])

# Defining the optmizer and loss function

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")

#Defining the hyperparameters
batch_size = 100
num_epochs = 100

# Training the model
#model.fit(train_inputs,train_targets, batch_size=batch_size, epochs=num_epochs,
#          validation_data=(validation_inputs,validation_targets), verbose=2)

''' In training, val_loss oscillates and it indicates overfitting, so we need to stop in first oscillation or
    tolarate a small number of oscilations '''

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,train_targets, batch_size=batch_size, callbacks=[early_stopping], epochs=num_epochs,
          validation_data=(validation_inputs,validation_targets), verbose=2)

# Testing the model
test_loss,test_accuracy = model.evaluate(test_inputs,test_targets)