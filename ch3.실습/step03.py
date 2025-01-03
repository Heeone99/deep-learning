import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output


# Square 함수
class Square(Function):
    def forward(self, x):
        return x**2

# Exp 함수
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
        

A = Square()
B = Exp()
C = Square()

# 합성함수 구현
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)