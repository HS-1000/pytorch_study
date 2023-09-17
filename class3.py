import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from  tqdm import tqdm

class baseDataset(Dataset):
    def __init__(self, root="./datasets/MNIST", istrain=True):
        super().__init__()

        dir = os.path.join(root, 'train' if istrain else 'val')
        print(len(os.listdir(root)))
        file_list = []
        for i in range(10):
            c_dir = os.path.join(dir, str(i))
            for p in os.listdir(c_dir):
                p = os.path.join(c_dir, p)
                file_list.append((i, p))
        self.file_list = file_list
        self.transform = transforms.Compose([transforms, ToTensor()])

    def __len__(self):
        return len(self.file_list)

    def __gettime(self, idx:int):
        pass

def get_dataloader(root, batch_size):
    train_dataset = baseDataset(root, True)
    val_dataset = baseDataset(root, False)
    train_dataloader = DataLoader(train_dataset, batch_size)

if __name__ == "__main__":
    test_traindataset = baseDataset("./datasets/MNIST", True)
    test_valdataset = baseDataset(is_train=False)
    print(len(test_traindataset))
    
