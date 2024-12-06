
import math

class Value(object):
    """
    Store a scalar and its gradient
    """
    def __init__(self, data, _prev=(), _op='', label=''):
        self.data = data
        self._prev = set(_prev)
        self.grad = 0
        self._op = _op
        self.label = label
        self._backward = lambda: None
    
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _prev=(self, other), _op='+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _prev=(self, other), _op='*')
        
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, _prev=(self), _op=f'**{{other}}')
        
        def _backward():
            self.grad += out.grad*other*(self.data**(other-1))
            
        out._backward = _backward
        
    def __truediv__(self, other):
        return self*(other**-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self*-1

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self*other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) +1)
        out = Value(t, _prev=(self,), _op='tanh')
        
        def _backward():
            self.grad += out.grad*(1-t**2)
        out._backward = _backward
        return out
    
    def relu(self):
        t = self.data if self.data > 0 else 0
        out = Value(t, _prev=(self,), _op='relu')
        
        def _backward():
            self.grad += out.grad if t > 0 else 0
        out._backward = _backward
        return out
    
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def backward(self):
        """
        backprop to all nodes in topological sort order
        """
        visited = set()
        postorder = []
        def topo(node):
            if node not in visited:
                visited.add(node)
                for prev_node in node._prev:
                    topo(prev_node)
                postorder.append(node)
            
        topo(self)
        self.grad = 1
        
        for node in reversed(postorder):
            node._backward()
            