# Activation Function
# sigmoid / tanh / relu
# backpropagation 을 optimizer에서 구현

import numpy as np

class ActivationFunction:
    def __init__(self,x):
        self.x = np.array(x)

    def sigmoid(self):
        result = 1.0 / (1.0 + np.exp(-self.x))
        self.back = result*(1-result)
        return result

    def tanh(self):
        result = (np.exp(self.x)-np.exp(-self.x))/(np.exp(self.x)+np.exp(-self.x))
        self.back = 1-result^2
        return result

    def relu(self):
        result = np.maximum(self.x,0)
        self.back = (self.x > 0) * 1
        return result

    def softmax(self):
        self.O = np.exp(self.x) / np.sum(np.exp(self.x),axis=1).reshape(-1,1)
        self.back = 1-self.O #얘는 loss를 따로 안 곱해도 된다.
        result = np.argmax(self.O,axis=1)
        return self.O

A=[[1,2,3,4],[2,5,-1,2]]
z = ActivationFunction(A)
z.relu()
z.back
np.array(A).T