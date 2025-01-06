if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero
if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable
    setup_variable()

    

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

