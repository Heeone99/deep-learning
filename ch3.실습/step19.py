import weakref
import numpy as np
import contextlib

"""
변수 사용성 개선
"""

# 설정 데이터
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('역전파 불가능', False)

class Variable:
    def __init__(self, data, name = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.', format(type(data)))


        self.data = data
        self.name = name    # 변수 구분을 위한 이름 지정
        self.grad = None
        self.creator = None
        self.generation = 0

    # 다차원 배열의 형상
    @property
    def shape(self):
        return self.data.shape
    
    # 차원 수
    @property
    def ndim(self):
        return self.data.ndim

    # 원소 수
    @property
    def size(self):
        return self.data.size

    # 데이터 타입
    @property
    def dtype(self):
        return self.data.dtype

    # 객체 내부의 원소 수 반환
    def __len__(self):
        return len(self.data)

    # 출력 결과 수정
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'


    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None


    # retain_grad가 True일 경우 모든 변수가 미분 결과(기울기) 유지
    # False일 경우 중간 변수의 미분값을 모두 None으로 설정
    # 역전파로 구하고 싶은 말단값의 미분변수만 구하기 위한 장치
    # 학습이 아닌 추론의 경우 활용됨
    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # 약한 참조를 활용해 메모리 관리
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y는 약한 참조

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # 설정 데이터를 활용한 역전파 활성화 / 비활성화
        # 인스턴스가 아닌 클래스 상태로 사용
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs) > 1 else outputs[0]


    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
x.name = 'x'

print(x.name)
print(x.shape)
print(x)