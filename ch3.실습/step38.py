if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F



"""
행렬의 형상 변환
"""

# reshape를 통해 텐서의 형상 변경
# np.reshape(x, shape)형태
# 단순히 형상만 변경하고 구체적인 계산은 하지 않는다 
x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
y.backward(retain_grad=True)
print(x.grad)

# transpose를 통해 텐서의 행렬을 변경(전치)
# np.transpose()
x.Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.transpose(X) # y = x.T
y.backward()
print(x.grad)