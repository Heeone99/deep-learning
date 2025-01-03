import numpy as np

"""
함수 구현
"""

class Variable:
    def __init__(self, data):
        self.data = data

# class Function:
#     def __call__(self, input):
#         x = input.data
#         y = x ** 2
#         output = Variable(y)
#         return output


class Function:
    # __call__은 f = Function() 형태임
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, in_data):
        raise NotImplementedError()

# Function 클래스 상속으로 __call__메소드 계승
class Square(Function):
    def forward(self, x):
        return x ** 2



x = Variable(np.array(10))
f = Square()
y = f(x)

# y의 객체 클래스와 실제 데이터 출력
print(type(y))
print(y.data)