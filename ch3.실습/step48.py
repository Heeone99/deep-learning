if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP


"""
다중 클래스 분류
"""


# 하이퍼 파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 데이터 불러오기 / 모델, 옵티마이저 생성
train_set = dezero.datasets.Spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 데이터셋의 인덱스 섞기
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 미니배치 생성
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[idx] for idx in batch_index]
        batch_x = np.array([b[0] for b in batch])  # Features
        batch_t = np.array([b[1] for b in batch])  # Labels


        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # 학습 경과 출력
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))