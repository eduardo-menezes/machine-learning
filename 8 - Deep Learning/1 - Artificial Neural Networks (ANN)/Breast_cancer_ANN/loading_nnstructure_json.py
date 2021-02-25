import numpy as np
from keras.models import model_from_json

#opening json file
file = open("structure_neural_network.json", "r")

# getting the neural structure
network_structure = file.read()
file.close()

# getting the compiler
model = model_from_json(network_structure)
model.load_weights("weights_breast_cancer.h5")

new_input = np.array([np.random.random_sample(30,)])
predictions = model.predict(new_input)

# Comprehension to return binary prediction

binary_prediction = [1 if item > 0.7 else 0 for item in predictions]
print(binary_prediction)
