import numpy as np 
import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c


## NN FRAMEWORK
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        norm_values = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values/norm_values 


## TESTING
nnfs.init() #helper functions + random seed to match book/videos
#create input data
X, y = spiral_data(100, 3)   
#initialize NN classes
dense1 = Layer_Dense(2,3) #HL1
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3) #HL2
activation2 = Activation_Softmax()
#Run NN Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(dense1.output)
activation2.forward(dense2.output)
#print the first 5 outputs
print(activation2.output[:5])
