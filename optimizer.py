# optimizer function

import sys
sys.path
sys.path.append('D:/paperstudy/start/layers.py')
sys.path.append('D:/paperstudy/start/loss_function.py')

import numpy as np
from loss_function import LossFunction
from layers import make_w
from layers import layers

class Optimizer:
    def __init__(self,x,y,w):
        self.x = np.array(x)
        self.y = np.array(y)  # onehot으로 들어와야 된다.
        self.w = w
        self.loss = []

    def Gradient_relu(self,epochs,lr):
        for i in range(epochs):
            self.result = layers(self.x,self.y,self.w)
            pp = self.result.layer_relu()
            chain_rule = self.y * self.result.diff[-1]
            for j in range(len(self.w)-1):
                self.w[-(j+1)] = self.w[-(j+1)] - lr * self.result.H[-(j+2)].T.dot(chain_rule)
                chain_rule = chain_rule.dot(self.w[-(j+1)].T) * self.result.diff[-(j+2)]
            self.w[0] = self.w[0] - lr * self.x.T.dot(chain_rule)
            current_loss = LossFunction(self.y,pp)
            self.loss.append(current_loss.cross_entropy())
            if i % 10 == 0:
                print(W,self.result.pred_y)
        return [W,self.result.pred_y]

    def Gradient_sigmoid(self,epochs,lr):
        for i in range(epochs):
            self.result = layers(self.x,self.y,self.w)
            pp = self.result.layer_sigmoid()
            chain_rule = self.y * self.result.diff[-1]
            for j in range(len(self.w)-1):
                self.w[-(j+1)] = self.w[-(j+1)] - lr * self.result.H[-(j+2)].T.dot(chain_rule)
                chain_rule = chain_rule.dot(self.w[-(j+1)].T) * self.result.diff[-(j+2)]
            self.w[0] = self.w[0] - lr * self.x.T.dot(chain_rule)
            current_loss = LossFunction(self.y,pp)
            self.loss.append(current_loss.cross_entropy())
            if i % 10 == 0:
                print(W,self.result.pred_y)
        return [W,self.result.pred_y]


