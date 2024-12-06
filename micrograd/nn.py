import random
from engine import Value

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0
            
    
class Neuron(Module):
    """
    a single neuron that takes some inputs and has one output
    out = tanh(w*x) + b
    """
    def __init__(self, nin, nonlin=True):
        # important to set the bias to 0. Setting it to random.uniform(-1, 1) leads to loss not converging
        # TODO: investigate this
        self.b = Value(random.uniform(-1,1))
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.nonlin = nonlin
        
    def __call__(self, x):
        act = sum((xi*wi) for xi,wi in zip(self.w, x)) + self.b
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return [self.b] + self.w
        
class Layer(Module):
    """
    A layer is a list of neurons that all take the same input
    """
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP(Module):
    """
    a MLP is a list of layers
    The last layer does not have a non-linearity
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    
    
