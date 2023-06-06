import numpy as np

class Neuron:
    def __init__(self, n_inputs, ntype):
        self.V = 0  # Membrane potential
        self.n_inputs = n_inputs
        self.weights = np.random.uniform(-1, 1, n_inputs)  # Randomly initialized weights
        self.ntype = ntype  # Neuron type: 'lif' or 'sigmoid'
        
    @classmethod
    def create(cls, n_inputs, ntype, threshold=None, resetPot=None, memTimeConst=None):
        # Factory method to create different types of neurons based on the ntype parameter
        if ntype == 'lif':
            return LifNeuron(n_inputs, threshold, resetPot, memTimeConst)
        elif ntype == 'sigmoid':
            return SigmoidNeuron(n_inputs)
        else:
            raise ValueError("Invalid neuron type: must be 'lif' or 'sigmoid'")

class SigmoidNeuron(Neuron):
    def __init__(self, n_inputs):
        Neuron.__init__(self, n_inputs, ntype='sigmoid')
    
    # Setters
    def clearActivity(self):
        self.V = 0
    
    # Getters
    def getOutput(self):
        return self.V
        
    # Methods
    def integration(self, inputs):
        inputSum = np.sum(self.weights * inputs)
        self.V = 1 / (1 + np.exp(-inputSum))  # Sigmoid activation function
    
class LifNeuron(Neuron):
    def __init__(self, n_inputs, threshold, resetPot, memTimeConst):
        Neuron.__init__(self, n_inputs, ntype='lif')
        
        self.threshold = threshold  # Threshold for firing
        self.resetPot = resetPot  # Membrane potential reset value
        self.memTimeConst = memTimeConst  # Membrane time constant
        self.out = 0  # Output of the LIF neuron
    
    # Setters
    def clearActivity(self):
        self.V = 0  # Reset membrane potential
        self.out = 0  # Reset output
    
    # Getters
    def getOutput(self):
        return self.out
        
    # Methods
    def integration(self, inputs):
        inputSum = np.sum(self.weights * inputs)
        self.V += inputSum  # Accumulate input
        self.V *= self.memTimeConst  # Apply membrane time constant
        
        if self.V >= self.threshold:
            self.out = 1  # Spike output
            self.V = self.resetPot  # Reset membrane potential
    
class Layer:
    def __init__(self, n_neurons, n_inputs, ntype, threshold=None, resetPot=None, memTimeConst=None):
        if type(ntype) == str:
            self.ntype = [ntype for _ in range(n_neurons)]  # Set the same ntype for all neurons
        else:
            self.ntype = ntype  # Use provided list of ntypes
            
        self.n_neurons = n_neurons
        self.neurons = [Neuron.create(n_inputs, self.ntype[i], threshold, resetPot, memTimeConst) for i in range(n_neurons)]
        
    # Setters
    def clearActivity(self):
        for neuron in self.neurons:
            neuron.clearActivity()
    
    # Getters
    def getOutput(self):
        return [neuron.getOutput() for neuron in self.neurons]
      
    # Methods
    def update(self, inputs):
        for neuron in self.neurons:
            neuron.integration(inputs)
            
    def checkExit(self, exitThreshold): 
        # check if the exit condition has been met
        if np.mean(self.getOutput()) >= exitThreshold:
            stop = True
        else:
            stop = False
        return stop
                
class Network:
    def __init__(self, n_layers, n_neurons, n_inputs, n_outputs, ntype, threshold=None, resetPot=None, memTimeConst=None):
        self.n_layers = n_layers
        self.layers = [0] * n_layers
        
        # Error exception
        if type(n_neurons) == int:
            self.n_neurons = [n_neurons for i in range(n_layers)]  # Set the same number of neurons for all layers
        elif type(n_neurons) == list:
            self.n_neurons = n_neurons  # Use provided list of neuron counts
        else:
            raise ValueError('n_neurons type must be integer or list of integers')
            
        if type(ntype) == str:
            self.ntype = [ntype for i in range(n_layers)]  # Set the same ntype for all layers
        elif type(ntype) == list:
            self.ntype = ntype  # Use provided list of ntypes
        else:
            raise ValueError('ntype must be string or list of strings')
            
            
        # First layer   
        self.layers[0] = Layer(n_neurons=self.n_neurons[0], 
                               n_inputs=n_inputs, ntype=self.ntype[0],
                               threshold=threshold, resetPot=resetPot, memTimeConst=memTimeConst)

        # Middle layers
        for i in range(1, n_layers - 1):
            self.layers[i] = Layer(n_neurons=self.n_neurons[i], 
                                   n_inputs=self.n_neurons[i-1], 
                                   ntype=self.ntype[i],
                                   threshold=threshold, resetPot=resetPot, memTimeConst=memTimeConst)
            
        # Last layer
        self.layers[n_layers - 1] = Layer(n_neurons=n_outputs, 
                                          n_inputs=self.n_neurons[n_layers - 2], 
                                          ntype=self.ntype[n_layers - 1], 
                                          threshold=threshold, resetPot=resetPot, memTimeConst=memTimeConst)
    
    # Setters
    def clearActivity(self):
        for layer in self.layers:
            layer.clearActivity()
    
    # Methods
    def run(self, inputs, exitThreshold):
        for i in range(self.n_layers):
            # If not the first layer, use the outputs of the previous layer as inputs
            if i != 0:
                inputs = [neuron.getOutput() for neuron in self.layers[i-1].neurons]

            # Update the current layer with the inputs
            self.layers[i].update(inputs)

            # Check if the exit threshold is reached in the current layer
            if self.layers[i].checkExit(exitThreshold):
                output = np.array(self.layers[i].getOutput())
                return (i / self.n_layers), output

        # If the exit threshold is not reached in any layer, return a message indicating it
        return (np.inf, 'Threshold not reached')

