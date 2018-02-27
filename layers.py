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

    def layer_relu(self):
        H = self.x
        self.result_w = []
        self.diff = []
        # softmax 전까지
        # if else 로 바꾸기 relu하다가 softmax하는거
        for i in range(len(self.W)-1):
            W = np.random.randn(self.W[i][0],self.W[i][1]) + 1 # W ~ N(1,1)
            J = np.dot(H,W)
            K = ActivationFunction(J)
            K.relu()
            diff = np.dot(H.T,K.back)
            H = K.relu()
            self.result_w.append(W)
            self.diff.append(diff)
        # softmax 항
        W = np.random.randn(self.W[-1][0],self.W[-1][1])
        J = np.dot(H,W)
        K = ActivationFunction(J)
        O = K.softmax()
        diff = np.dot(H.T,K.back)
        self.result_w.append(W)
        self.diff.append(diff)
        pred_y = np.argmax(O,axis=1)
        return [self.result_w, self.diff,pred_y]








x = [[1,2,3,4,5],[2,3,4,5,6],[2,6,1,2,3]]
w = [15,10]
y=[[0,1],[1,0],[0,1]]
k = layers(x,y,w=w)
k.layer_relu()
k.result_w
k.diff
# okay 이제 diff 를 곱해서 빼주는 작업을 optimizer에서 해주면 된다