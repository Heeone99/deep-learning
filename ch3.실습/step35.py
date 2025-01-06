if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


"""
하이퍼볼릭 탄젠트 함수 고차 미분
"""

# 하이퍼볼릭 탄젠트 함수
# class Tanh(Function):
#     def forward(self, x):
#         xp = cuda.get_array_module(x)
#         y = xp.tanh(x)
#         return y

#     def backward(self, gy):
#         y = self.outputs[0]()  
#         gx = gy * (1 - y * y)
#         return gx


# def tanh(x):
#     return Tanh()(x)


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 1

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')