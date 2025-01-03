import numpy as np


"""
가변 길이의 입출력 처리 개선
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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):            # 임의 개수의 인수를 받아 함수 호출 가능(동적 할당)
        x_list = [x.data for x in inputs]
        y_list = self.forward(*x_list)      # *를 통해 언팩(리스트의 원소를 낱개로 풀어 전달)
        if not isinstance(y_list, tuple):   # 튜플로 변환
            y_list = (y_list,)
        outputs = [Variable(as_array(y)) for y in y_list]

        for output in outputs:
            output.set_creator(self)
        self.input = input
        self.output = output

        return outputs if len(outputs) > 1 else outputs[0]  # 리스트의 원소가 하나라면 첫 번째 원소 반환

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def add (x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)