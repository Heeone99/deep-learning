import unittest
import numpy as np


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


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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


def square(x):
    return Square()(x)


# 수치 미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        result = np.array(6.0)      # unittest시 미분값이 4.0인데 6.0으로 설정해서 비교시 오류 발생
        self.assertEqual(y.data, result)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        result = np.array(6.0)
        self.assertEqual(x.grad, result)

    # 기울기 확인
    # 수치 미분과 역전파의 결과를 비교
    def test_gradient(self):
        x = Variable(np.random.rand(1)) # 무작위 입력값 생성
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad) # 두 값이 얼마나 가까운지 판정
        self.assertTrue(flg)