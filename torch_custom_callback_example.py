from torch.utils.data import Dataset
import os
import gzip
import numpy as np
from torchvision import transforms

class Normalize:
    def __call__(self, image):
        image = image.astype("float32") / 255.0
        return image

class ToTensor:
    def __call__(self, image):
        image = np.expand_dims(image, axis = -1)
        image = image.transpose((2, 0, 1))

        return image


def read_gzip(data_path):
    with open(data_path, "rb") as f:
        content = f.read()

    return content


class MnistDataset(Dataset):
    def __init__(self, dir_path , validation_type = False):
        if validation_type:
            self._init_validation(dir_path = dir_path)
        else:
            self._init_training(dir_path = dir_path)

        print(self.x.shape)
        print(self.y.shape)

        self.transforms = transforms.Compose([Normalize(), ToTensor()])



    def _init_validation(self , dir_path):
        with gzip.open(os.path.join(dir_path, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
            self.y = np.frombuffer(f.read(), np.uint8, offset = 8)

        with gzip.open(os.path.join(dir_path, "t10k-images-idx3-ubyte.gz"), "rb") as f:
            self.x = np.frombuffer(f.read(), np.uint8, offset = 16).reshape(len(self.y), 28, 28)


    def _init_training(self , dir_path):
        with gzip.open(os.path.join(dir_path, "train-labels-idx1-ubyte.gz"), "rb") as f:
            self.y = np.frombuffer(f.read(), np.uint8, offset = 8)

        with gzip.open(os.path.join(dir_path, "train-images-idx3-ubyte.gz"), "rb") as f:
            self.x = np.frombuffer(f.read(), np.uint8, offset = 16).reshape(len(self.y), 28, 28)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = { "image" : self.transforms(self.x[idx]), "label" : self.y[idx] }
        return sample

if __name__ == "__main__":
    dataset = MnistDataset(dir_path = "./datasets/mnist", validation_type = True)

    for index in range(len(dataset)):
        sample = dataset[index]
        #print(sample["image"].shape, sample["label"].shape)
        print(type(sample["label"]))
