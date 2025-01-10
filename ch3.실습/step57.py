if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F


"""
Convolution과 Pooling 구현
"""


# 함수 형태의 Conv
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W  # Width의 W와 구분하기 위함
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)   # pair(x) -> 인수가 x라면 (x,x) 형태의 튜플로 변환해 반환
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # im2col을 통해 3차원을 2차원 텐서로 변환
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose() # (10, 3, 5, 5)배열을 -> reshape(10, -1) -> (10,75)
    t = liner(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


# # 계층 형태의 Conv
# class Conv2d(layer):
#     def __init__(self, out_channels, kernel_size, stride=1,
#                  pad = 0, nobias=False, dtype=np.float32, in_channels=None):
#         super().__init__()
#         self.in_channels = in_channels      # 입력 데이터의 채널 수
#         self.out_channels = out_channels    # 출력 데이터의 채널 수
#         self.kernel_size = kernel_size      # 커널 크기
#         self.stride = stride            # 스트라이드
#         self.pad = pad          # 패딩
#         self.dtype = dtype

#         self.W = Parameter(None, name='W')
#         if in_channels is not None:
#             self._init_W()

#         if nobias:
#             self.b = None
#         else:
#             self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

#     def _init_W(self, xp=np):
#         C, OC = self.in_channels, self.out_channels
#         KH, KW = pair(self.kernel_size)
#         scale = np.sqrt(1 / (C * KH * KW))
#         W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
#         self.W.data = W_data

#     def forward(self, x):
#         if self.W.data is None:
#             self.in_channels = x.shape[1]
#             xp = cuda.get_array_module(x)
#             self._init_W(xp)

#         y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
#         return y



# # Pooling
# def pooling_simple(x, kernel_size, stride=1, pad=0):
#     x = as_variable(x)

#     N, C, H, W = x.shape
#     KH, KW = pair(kernel_size)
#     PH, PW = pair(pad)
#     SH, SW = pair(stride)
#     OH = get_conv_outsize(H, KH, SH, PH)
#     OW = get_conv_outsize(W, KW, SW, PW)

#     col = im2col(x, kernel_size, stride, pad, to_matrix=True)   # 입력 데이터 전개(2차원 변환)
#     col = col.reshape(-1, KH * KW)
#     y = col.max(axis=1) # 각 행의 최댓값
#     y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)   # 적절한 크기로 출력의 형상을 변환
#     return y


# im2col
x1 = np.random.rand(1, 3, 7, 7)
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)   # (9, 75)

x2 = np.random.rand(10, 3, 7, 7)
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)   # (90, 75)


# conv2d
N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)
x = Variable(np.random.randn(N ,C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()
print(y.shape)  # (1, 8, 15, 15)
print(x.grad.shape) # (1, 5, 15, 15)