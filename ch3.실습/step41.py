if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F


"""
행렬 곱
"""

# class Matmul(Function):
#     def forward(self, x, W):
#         y = x.dot(W)
#         return y

#     def backward(self, gy):
#         x, W = self.inputs
#         gx = matmul(gy, W.T)
#         gW = matmul(x.T, gy)
#         return gx, gW

# def matmul(x, W):
#     return Matmul()(x, W)

x = Variable(np.random.randn(2, 3))
w = Variable(np.random.randn(3, 4))
y = F.matmul(x, w)
y.backward()

print(x.grad.shape)
print(w.grad.shape)

