import numpy as np

"""
가변 길이의 입출력 처리
"""

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


# inputs 리스트에서 원소를 꺼내 새로운 리스트에 넣음
class Function:
    def __call__(self, inputs):
        x_list = [x.data for x in inputs]
        y_list = self.forward(x_list)
        outputs = [Variable(as_array(y)) for y in y_list]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs

    def forward(self, x_list):
        raise NotImplementedError()

    def backward(self, gy_list):
        raise NotImplementedError()


# 덧셈을 위한 Add 클래스
# 여러 출력값을 처리할 수 있도록 일반화하기 위해 
# 튜플 형태로 반환
class Add(Function):
    def forward(self, x_list):
        x0, x1 = x_list
        y = x0 + x1
        return (y,) # 길이가 1인 튜플을 생성


# 리스트가 아닌 튜플을 반환하는 이유
# 튜플은 출력값의 개수와 구조가 고정
# 리스트는 출력값의 개수가 가변적

# Add 클래스와 같은 함수의 출력값은 개수가 고정적이라 튜플로 반환


x_list = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
y_list = f(x_list)
y = y_list[0]
print(y.data)