import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        # 미분값 저장
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 입력 변수를 보관해 역전파에서 활용
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()



class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    # 역전파
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    # 역전파
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)


# 역전파는 dy/dy = 1 에서 시작
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

# x의 역전파
print(x.grad)
