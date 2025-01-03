import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input      # 객체의 인스턴스 변수를 동적으로 추가
        self.output = output    # 객체의 인스턴스 변수를 동적으로 추가
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

# 중앙차분을 활용한 수치미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

# 합성 함수
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

# 자릿수 누락으로 인해 오차가 포함되기 쉽다.
# 효율적 계산 + 오차도 더 적은 역전파의 등장
x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)