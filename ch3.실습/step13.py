import numpy as np


"""
가변 길이 입출력 - 역전파
"""

class Variable:
    def __init__(self, data):
        if data is not None:
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
            gy_list = [output.grad for output in f.outputs] # 미분값들을 리스트에 저장
            gx_list = f.backward(*gy_list)      # 위의 리스트들을 바탕으로 함수 f의 역전파 호출 
            if not isinstance(gx_list, tuple):  # 튜플 변환
                gx_list = (gx_list,)

            for x, gx in zip(f.inputs, gx_list):    # 모든 Variable 인스턴스에 각각 알맞은 미분값 설정
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)



def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        x_list = [x.data for x in inputs]
        y_list = self.forward(*x_list)
        if not isinstance(y_list, tuple):
            y_list = (y_list,)
        outputs = [Variable(as_array(y)) for y in y_list]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x_list):
        raise NotImplementedError()

    def backward(self, gy_list):
        raise NotImplementedError()



class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data     # 입력 변수 x 호출
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


# 덧셈이라 입력 2개 출력 1개
# 역전파에서는 입력 1개 출력 2개
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add (x0, x1):
    return Add()(x0, x1)



x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()

print(z.data)
print(x.grad)
print(y.grad)



