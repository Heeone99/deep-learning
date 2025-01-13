import os
from PIL import Image
from torchvision import transforms



"""
Pytorch 데어테셋과 데이터 로더
사용자 편의에 맞춰 Dataset의 메서드 재정의
"""



class ImageDataset(Dataset):
    def __init__(self, img_dir, label_file):
        super(ImageDataset, self).__init__()
        self.img_dir = img_dir
        self.labels = torch.tensor(np.load(label_file, allow_pickle=True))
        self.transforms = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
        img = Image.open(img_path)
        img = self.transforms(img).flatten()    # 이미지를 텐서로 변환 후 1차원으로 평탄화
        label = self.labels[idx]
        return {"data":img, "label":label}

    def __len__(self):
        return len(self.labels)


# 임의로 생성
# 아래의 경로를 변경해 주세요
img_dir = "path_to_image_directory"  # 이미지 디렉토리 경로
label_file = "path_to_label_file.npy"  # 라벨 파일 경로


# Custom Dataset 객체 생성
custom_dataset = ImageDataset(img_dir=img_dir, label_file=label_file)


# Custom DataLoader
# 하이퍼 파라미터로 변경 가능합니다.
batch_size = 32 # 배치 크기
shuffle = True  # 데이터 셔플 여부
num_workers = 2 # 데이터 로드에 사용할 워커 수(데이터 로드에 사용할 프로세스의 개수)


data_loader = DataLoader(
    dataset = custom_dataset,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers)


# 사용 예시
for minibatch in data_loader:
    data, labels = minibatch['data'], minibatch['label']
    print(data)
    print(labels)