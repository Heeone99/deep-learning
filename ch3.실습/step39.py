if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F


"""
합계 함수와 축
"""


x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

# axis(축, 방향)
# axis=1일 경우 가로방향 합(같은 행 끼리 합) : [[6], [15]]의 결과
# axis=0인 경우 세로방향 합(같은 열 끼리 합)
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

#keepdims는 입력과 출력의 차원 수(축 수)를 똑같게 유지할지 정하는 플래그
x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)
