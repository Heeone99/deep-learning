import numpy as np
import math


"""
경사하강법
"""

# 로젠브록 함수
# f(x0, x1) = b(x1 - x0 ** 2) ** 2 + (a - x0) ** 2
# 최적화 문제의 벤치마크 함수로 자주 사용
# a = 1, b = 100으로 설정하여 벤치마크 하는것이 일반적
def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001  # 학습률
iters = 1000 # 반복 횟수

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)
    
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

