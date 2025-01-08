if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
np.random.seed(0)
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP


"""
Softmax와 Cross Entropy
"""


# 소프트맥스 함수
def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

model = MLP((10, 3))    

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = softmax1d(y)
print(y)
print(p)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
p = F.softmax_simple(y)
print(y)
print(p)



# 교차 엔트로피 오차
def softmax_cross_entropy_simple(x, t):
    # x는 소프트 맥스 적용 전 출력
    # t는 정답 데이터
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)   # p의 값은 0~1 사이값
    # log(0)을 방지 1e-15이하면 1e-15 / 1이상이면 1으로 변환
    p = clip(p, 1e-15, 1.0) 
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


loss = F.softmax_cross_entropy_simple(y, t)
loss.backward()
print("softmax_cross_entropy: ", loss)
