if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F



"""
선형 회귀
"""


# 선형 회귀
# x로 부터 실숫값 y를 예측하는 것을 회귀라 한다.
# 회귀 모델 중 예측값이 선형(직선)을 이루는 것을 선형 회귀라 한다.

# 임의의 데이터셋 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)  # y에 무작위 노이즈 추가
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))



def predict(x):
    y = F.matmul(x, W) + b
    return y


# MSE(평균 제곱 오차)를 손실 함수로 사용
# 이를 줄이는 방향으로 학습

# class MeanSquaredError(Function):
#     def forward(self, x0, x1):
#         diff = x0 - x1
#         y = (diff ** 2).sum() / len(diff)
#         return y

#     def backward(self, gy):
#         x0, x1 = self.inputs
#         diff = x0 - x1
#         gx0 = gy * diff * (2. / len(diff))
#         gx1 = -gx0
#         return gx0, gx1


# def mean_squared_error(x0, x1):
#     return MeanSquaredError()(x0, x1)

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100


for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()


    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)



plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.show()