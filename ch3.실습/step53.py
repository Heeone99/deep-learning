if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP


"""
모델의 매개변수 저장 및 읽어오기
"""

# def save_weights(self, path):
#     self.to_cpu()

#     params_dict = {}
#     self._flatten_params(params_dict)
#     array_dict = {key: param.data for key, param in params_dict.items()
#                     if param is not None}
#     try:
#         np.savez_compressed(path, **array_dict)
#     except (Exception, KeyboardInterrupt) as e:
#         if os.path.exists(path):
#             os.remove(path)
#         raise

# def load_weights(self, path):
#     npz = np.load(path)
#     params_dict = {}
#     self._flatten_params(params_dict)
#     for key, param in params_dict.items():
#         param.data = npz[key]


max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# 매개변수 읽기
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss / len(train_set)))

# 매개변수 저장
model.save_weights('my_mlp.npz')