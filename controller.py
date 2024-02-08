import random
import jax.numpy as jnp
import numpy as np
import jax

    

class PIDController():
    def __init__(self):
        self.e_prev = 0
        self.e_sum = 0
        self.e_diff = 0
        self.U = 0
    
    def reset(self):
        self.e_prev = 0
        self.e_sum = 0
        self.e_diff = 0
        self.U = 0
    
    def predict(self, e, params):
        self.e_sum += e
        self.e_diff = e - self.e_prev
        self.U = params[0] * e + params[1] * self.e_sum + params[2] * self.e_diff
        self.e_prev = e
        return self.U
    
class NeuralController():
    def __init__(self, activations):
        self.activations = activations
        self.e_prev = 0
        self.e_sum = 0
        self.e_diff = 0
        self.U = 0
        
    def gen_params(self, hidden_layers, initRange):
        parameters = []
        key = jax.random.PRNGKey(44)  # You can choose any seed value
        layers = [3] + hidden_layers + [1]
        print (layers)
        for i in range(1, len(layers)):
            key, subkey = jax.random.split(key)
            input_size = layers[i - 1]
            output_size = layers[i]

            weights = jax.random.uniform(subkey, (output_size, input_size), minval=initRange[0], maxval=initRange[1])
            biases = jax.random.uniform(subkey, (output_size,), minval=initRange[0], maxval=initRange[1])
            weights = jnp.array(weights)
            biases = jnp.array(biases)

            parameters.append((weights, biases))
        for i, (weights, biases) in enumerate(parameters):
            print(f"Layer {i + 1} - Weights shape: {np.array(weights)}, Biases shape: {np.array(biases)}")
        return parameters
    
    def predict(self, E: float, params: jnp.array):
        
        def relu(x): return jnp.maximum(0, x + 1e-7)
        def tanh(x): return jnp.tanh(x)
        def sigmoid(x): return 1/(1+jnp.exp(-x))
        
        self.e_sum += E
        self.e_diff = E - self.e_prev
        self.e_prev = E
        
        features = jnp.array([E, self.e_sum, self.e_diff])
        for (weights, biases), type in zip(params, self.activations):
            #jax.debug.print("features: {features}", features=features)
            output = jnp.dot(weights, features) + biases
            #jax.debug.print("output: {features}", features=output)
            #print(type)
            if type == 'relu':
                features = relu(output)
            elif type == 'tanh':
                features = tanh(output)
            elif type == 'sigmoid':
                features = sigmoid(output)
        #jax.debug.print("Signal: {features[0]}", features=features)
        return features[0]
    
    def reset(self):
        self.e_prev = 0
        self.e_sum = 0
        self.e_diff = 0
        self.U = 0
    
 