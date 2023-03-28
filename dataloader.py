import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch


class MyMnist(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.labels = df["label"]
        imgs = df.drop(["label"], axis=1)
        self.length = len(df.index)
        imgs = np.array(imgs).reshape(self.length, 28, 28).astype(np.float32)
        imgs = imgs / 255
        self.imgs = torch.from_numpy(imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        label_vector = [0.0 for i in range(26)]
        label_vector[label] = 1.0
        return torch.Tensor(label_vector), label, self.imgs[index].view(1, 28, 28)


if __name__ == '__main__':
    dataset = MyMnist('datas/sign_mnist_test.csv')
    print(len(dataset))
    print(dataset[0])