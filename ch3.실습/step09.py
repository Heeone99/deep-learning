import numpy as np

"""
직관성을 위한 코드 개선과 데이터 타입 처리
"""

# 인스턴스 처리를 위해 예외 코드 추가
class Variable:
    def __init__(self, data):
        if data is None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.', format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func


    # 변수의 grad가 None이면 자동으로 미분값 생성
    # ones_like()를 통해 모든 요소가 1로 초기화된 배열 생성
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# ndarray 인스턴스를 활용해 계산시 데이터 타입이 float으로 변경 됨
# as_array()를 활용해 ndarray로 타입 변경
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))  # ndarray 인스턴스로 변환
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 직관성을 위해 개선
def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)


# 직관성을 위해 개선
def exp(x):
    # f = Exp()
    # return f(x)
    return Exp()(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

# 인스턴스만 처리
x = Variable(np.array(1.0)) # np.array는 ndarray 객체를 반환하므로 사용 가능
x = Variable(None)  # 불가능
x = Variable(1.0)   # 불가능