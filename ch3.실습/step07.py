import numpy as np

"""
재귀를 활용한 역전파 구현
"""


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    # 변수 설정
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator    # 1. 함수를 가져옴
        if f is not None:
            x = f.input     # 2. 함수의 입력을 가져옴
            x.grad = f.backward(self.grad)  # 3. 함수의 역전파 호출
            x.backward()    # 4. 하나 앞 변수의 역전파 호출(재귀)
                            # x의 creator가 이전 함수(Function)를 참조하고 있기 때문
                            # 링크드 리스트


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)    # 출력 변수에 함수 설정(역전파 계산을 위해 함수를 알아야 한다.)
        self.input = input
        self.output = output    # 출력 저장
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

# backward
y.grad = np.array(1.0)
y.backward()
print(x.grad)