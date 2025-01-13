import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


"""
MNIST 분류기 구축
"""


class BaseClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


# Pytorch에서 MNIST 데이터셋을 읽어들인다.
train_data = MNIST(".", train=True, download=True, transform=ToTensor())
test_data = MNIST(".", train=False, download=True, transform=ToTensor())
train_load = DataLoader(train_data, batch_size=64, shuffle=True)
test_load = DataLoader(test_data, batch_size=64, shuffle=False)


# 모델, 옵티마이저, 하이퍼 파라미터 인스턴스화
in_dim, feature_dim, out_dim = 784, 256, 10
lr = 1e-3
loss = nn.CrossEntropyLoss()
epochs = 40
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)


def train(
    classifier=classifier,
    optimizer=optimizer,
    epochs=epochs,
    loss_fn=loss):
    classifier.train()
    loss_lt = []
    for epoch in range(epochs):
        running_loss = 0.0
        for minibatch in train_load:
            data, target = minibatch
            data = data.flatten(start_dim=1)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 각 미니배치 손실 합을 기록
            running_loss += computed_loss.item()
        loss_lt.append(running_loss/len(train_load))
        print("Epoch: {} train loss: {}".format(epoch+1, running_loss/len(train_load)))
        
        plt.plot([i for i in range(1,epoch+1)], loss_lt)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("MNIST Training Loss: Optimizer {}, lr {}".format("SGD", lr))
        plt.show()

        # 상태를 체크포인트로 파일에 저장
        torch.save(classifier.state_dict(), 'mnist.pt')

def test(classifier=classifier, loss_fn=loss):
    classifier.eval()
    accuracy = 0.0
    computed_loss = 0.0

    with torch.no_grad():
        for data, target in test_load:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)

            # 손실과 정확도
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)

        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item()/(len(test_load)*64), 
            accuracy*100.0/(len(test_load)*64)
        ))



# 학습 실행
train(
    classifier=classifier,
    optimizer=optimizer,
    epochs=epochs,
    loss_fn=loss
)

# 테스트 실행
test(classifier=classifier, loss_fn=loss)
