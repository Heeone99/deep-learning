import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


# 데이터 생성
data = np.array(1.0)

# 객체 생성
x = Variable(data)
print(x.data)

# 데이터 변경
x.data = np.array(2.0)
print(x.data)

"""
넘파이의 다차원 배열
"""

# array
# 배열을 간편하게 생성하기 위한 고수준 함수로 일반적인 배열 작업에 적합

# ndarray
# 배열을 직접 제어하거나 특수한 경우에 사용하며 배열의 모양, 데이터 타입 등을 명시적으로 지정해야 함
# 주로 주로 성능 최적화, 메모리 관리, 그리고 데이터 구조의 세부 제어가 필요한 경우 사용


y = np.array(1)
print(y.ndim)
# 내부 데이터 타입 확인
print(type(y))
# 배열 객체의 타입 확인
print(y.dtype)

y = np.array([1,2,3])
print(y.ndim)

y= np.array([[1,2],[3,4]])
print(y.ndim)