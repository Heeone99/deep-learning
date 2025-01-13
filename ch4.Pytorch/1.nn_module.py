import torch
import torch.nn as nn
from torch import optim

in_dim, out_dim = 256, 10   # 입력 256차원, 출력 10차원
vec = torch.randn(256)  # 차원이 256인 벡터 / 입력 데이터
# FC layer 정의
layer = nn.Linear(in_dim, out_dim, bias=True)   # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
out = layer(vec)    # 임의의 값 W와 b를 통해 자동 계산
# print('auto output: ', out)


# 수동으로 값을 지정해서 계산
W = torch.rand(10, 256)
b = torch.zeros(10, 1)  # 0으로 구성된 10,1 행렬
# print('b: ', b)
out = torch.matmul(W, vec) + b

# print('manual output: ', out)


# 계층 추가
in_dim, feature_dim, out_dim = 784, 256, 10
vec = torch.randn(784)
layer1 = nn.Linear(in_dim, feature_dim, bias=True)
layer2 = nn.Linear(feature_dim, out_dim, bias=True)
out = layer2(layer1(vec))
# print('multi_layer: ',out)


# 비선형 함수 추가
relu = nn.ReLU()
out = layer2(relu(layer1(vec)))
# print('relu: ', out)



# 모든 신경망을 서브 클래싱하는 기본 클래스인 nn.Module
class BaseClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.layer1 = nn.Linear(in_dim, feature_dim, bias=True)
        self.layer2 = nn.Linear(feature_dim, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        y = self.layer2(x)
        return y


data = 10
in_dim, feature_dim, out_dim = 784, 256, 10
x = torch.randn((data, in_dim)) # 10개의 입력 데이터
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
out = classifier(x) # forward() 메서드가 암시적으로 호출
# print("result: ", out)


# 역전파
loss = nn.CrossEntropyLoss()
target = torch.tensor([0, 3, 2, 8, 2, 9, 3, 7, 1, 6])
computed_loss = loss(out, target)
computed_loss.backward()

# optimizer
lr = 1e-3 # 러닝 레이트
optimizer = optim.SGD(classifier.parameters(), lr=lr)

optimizer.step()
optimizer.zero_grad()
