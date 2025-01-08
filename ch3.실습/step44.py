if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Parameter
import dezero.functions as F
import dezero.layers as L


"""
Parameter와 Layer 클래스를 활용한 신경망 구현
"""


# Parameter 클래스
# Variable 클래스르 상속한게 전부
# Variable과 기능은 같지만 구분이 가능하다.

# class Parameter(Variable):
#     pass


x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))

y = x * p

print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))



# Layer 클래스
# Function 클래스를 상속받고 매개변수를 유지한다.

# class Layer:
#     def __init__(self):
#         self._params = set()

#     def __setattr__(self, nmae, value):
#         if isinstance(value, Parameter):
#             self._params.add(name)
#         super().__setattr__(name, value)

#     def __call__(self, *inputs):
#         outputs = self.forward(*inputs)
#         if not isinstance(outputs, tuple):
#             outputs = (outputs,)
#         self.inputs = [weakref.ref(x) for x in inputs]
#         self.outputs = [weakref.ref(y) for y in outputs]
#         return outputs if len(outputs) > 1 else outputs[0]

#     def forward(self, inputs):
#         raise NotImplementedError()

#     def params(self):
#         for name in self._params:
#             obj = self.__dict__[name]

#             if isinstance(obj, Layer):
#                 yield from obj.params()
#             else:
#                 yield obj

#     def cleargrads(self):
#         for param in self.params():
#             param.cleargrad()



# Linear 클래스
# 함수로서의 Linear가 아닌 계층으로서의 Linear 구현

# class Linear(Layer):
#     def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
#         super().__init__()
#         self.in_size = in_size
#         self.out_size = out_size
#         self.dtype = dtype

#         # Parameter 인스턴스 변수에 가중치 설정
#         self.W = Parameter(None, name='W')
#         # in_size 미지정시 나중으로 연기
#         if self.in_size is not None:
#             self._init_W()

#         if nobias:
#             self.b = None
#         else:
#             # Parameter 인스턴스 변수에 편향 설정
#             self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

#     def _init_W(self, xp=np):
#         I, O = self.in_size, self.out_size
#         W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
#         self.W.data = W_data

#     def forward(self, x):
#         # 데이터를 흘려보내는 시점에 가중치 초기화
#         if self.W.data is None:
#             self.in_size = x.shape[1]
#             xp = cuda.get_array_module(x)
#             self._init_W(xp)

#         y = F.linear(x, self.W, self.b)
#         return y


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)   # 출력 크기 지정
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)