import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim


"""
최소점과 극소점
"""



class Net(nn.Module):
  def __init__(self, in_dim, feature_dim, out_dim):
    super(Net, self).__init__()
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, out_dim),
        nn.ReLU()
    )

  def forward(self, inputs):
    return self.classifier(inputs)




torch.manual_seed(0)

# Pytorch에서 MNIST 데이터셋을 읽어들인다.
trainset = MNIST('.', train=True, download=True, transform=ToTensor())
testset = MNIST('.', train=False, download=True, transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)



IN_DIM, FEATURE_DIM, OUT_DIM = 784, 256, 10
lr = 1e-4
loss_fn = nn.CrossEntropyLoss()
num_epochs = 40
classifier = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
optimizer = optim.SGD(classifier.parameters(), lr=lr)
classifier


# 새로운 신경망 모델 학습 
# for epochs in range(num_epochs):
#   running_loss = 0.0
#   for inputs, labels in trainloader:
#     optimizer.zero_grad()
#     outputs = classifier(inputs)
#     loss = loss_fn(outputs, labels)
#     loss.backward()
#     running_loss += loss.item()
#     optimizer.step()
#   print(running_loss/len(trainloader))


# model = classifier
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", 
#           model.state_dict()[param_tensor].size())


          
# torch.save(model.state_dict(), './MNIST/mnist2.pt')        


model = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
# 학습된 모델을 다시 인스턴스화
model.load_state_dict(torch.load('./MNIST/mnist2.pt'), strict=False)


# state_dict를 활용해 파라미터에 접근
opt_state_dict = copy.deepcopy(model.state_dict())

for param_tensor in opt_state_dict:
    print(param_tensor, "\t", opt_state_dict[param_tensor].size())





# 무작위로 초기화된 신경망 생성
model_rand = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
rand_state_dict = copy.deepcopy(model_rand.state_dict())

# 선형 보간된 파라미터를 위해 신규 state_dict를 생성
test_model = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
test_state_dict = copy.deepcopy(test_model.state_dict())

alpha = 0.2
beta = 1.0 - alpha
for p in opt_state_dict:
    test_state_dict[p] = (opt_state_dict[p] * beta + rand_state_dict[p] * alpha)


# 추론 함수
def inference(testloader, model, loss_fn):
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        running_loss += loss.item()
    running_loss /= len(testloader)
    return running_loss


results = []
for alpha in torch.arange(-2, 2, 0.05, dtype=torch.float32):
    beta = 1.0 - alpha
    # 선형 보간된 파라미터를 계산
    for p in opt_state_dict:
        test_state_dict[p] = (opt_state_dict[p] * beta + rand_state_dict[p] * alpha)

    # 선형 보간된 파라미터를 테스트 모델로 로드
    model.load_state_dict(test_state_dict)
    # 선형 보간된 파라미터를 사용해서 손실을 계산
    loss = inference(trainloader, model, loss_fn)
    results.append(loss)
    print(f"Alpha: {alpha:.2f}, Loss: {loss:.4f}")


# 시각화
plt.plot(np.arange(-2, 2, 0.05), results, 'ro')
plt.ylabel('Incurred Error')
plt.xlabel('Alpha')
plt.title('Loss Interpolation between Random and Trained Model')
plt.show()