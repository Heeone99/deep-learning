if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as f


"""
텐서의 활용
"""


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))

y = F.sin(x)    # sin 함수가 원소별로 적용
print(y)    

t = x + c   # 원소별 덧셈
print(t)

y = F.sum(t)


y.backward(retain_grad=True)

print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)