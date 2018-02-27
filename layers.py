# layer 일반화 #


import sys
sys.path
sys.path.append('D:/paperstudy/start/activation_function2.py')

import numpy as np
from activation_function2 import ActivationFunction

def make_w(x,y,w):
    form = [x.shape[1]]
    for i in w:
        form.append(i)
    form.append(y.shape[1])
    W = []
    for i in range(len(form)-1):
        W.append([form[i], form[i + 1]])
    return W

class layers:
    def __init__(self,x,y,w=[15,10]):
        # w에는 #_hidden1_node,#_hidden2_node.... 차례대로 입력해주면 된다.
        # default 는 #_hidden1_node = 15,#_hidden2_node = 10
        self.x = np.array(x)
        self.y = np.array(y) #onehot으로 들어와야 된다.
        self.W = make_w(self.x,self.y,w)

    def layer(self,method='relu'):








x = [[1,2,3,4,5],[2,3,4,5,6],[2,6,1,2,3]]
w = [15,10]
y=[[0,1],[1,0],[0,1]]
k = layers(x,y,w=w)
l =k.W
l[0]