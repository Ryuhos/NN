# Loss Function
# sse / cross entropy

import numpy as np

class LossFunction:
    def __init__(self,y,pred_y):
        self.y = y
        self.pred_y = pred_y

    def sse(self):
        result = np.sum((self.y-self.pred_y)^2)/2
        return result

    def cross_entropy(self):
        result = -np.sum(self.y*np.log(self.pred_y))
        return result

