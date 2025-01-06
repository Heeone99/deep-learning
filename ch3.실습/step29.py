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
뉴턴 방법
"""

# 경사하강법 : 
# 손실 함수의 기울기를 따라 최소값으로 이동 / 수렴속도 느림 / 계산 복잡도 낮음 / 메모리 요구량 적음 / 딥러닝, 대규모 데이터 최적화

# 뉴턴 방법 : 
# 손실 함수의 2차 근사(테일러 전개)를 이용해 최소값을 계산 / 수렴 속도 빠름 / 계산 복잡도 높음 / 메모리 요구량 많음 / 물리, 경제학, 최적화 이론 적합


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y= f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)