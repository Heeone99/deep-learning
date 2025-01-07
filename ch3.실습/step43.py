if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F



"""
신경망 구현
"""


# 임의의 데이터셋 생성
# 복잡한 데이터를 위해 생성시 sin 함수 활용
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 입력, 은닉, 출력층 설정
# 가중치 초기화
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


# matmul 이후 sum 수행 -> linear 함수 구현
# def linear_simple(x, W, b=None):
#     t = matmul(x, W)
#     if b is None:
#         return t

#     y = t + b
#     t.data = None 
#     return y



# 활성화 함수인 sigmoid
# 비선형성
# def sigmoid_simple(x):
#     x = as_variable(x)
#     y = 1 / (1 + exp(-x))
#     return y


#신경망 추론
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

# 신경망 학습
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)  # Loss 함수를 MSE 사용

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:   # 1000회마다 loss 출력
        print(loss)



plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()