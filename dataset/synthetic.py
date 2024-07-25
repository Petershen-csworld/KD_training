import torch 
from torch.utils.data import Dataset, DataLoader 
import torchvision 
from typing import Dict, List, Tuple 
import numpy as np
import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder
import os 
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2 
from tqdm import tqdm 

CHANNEL_NUM = 3
# we are using colored image 
def cal_dir_stat(root):
    cls_dirs = [d for d in listdir(root) if isdir(join(root, d))]
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for idx, d in tqdm(enumerate(cls_dirs)):
        # print("#{} class".format(idx))
        im_pths = glob(join(root, d, "*.jpg"))

        for path in im_pths:
            im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im/255.0
            pixel_num += (im.size/CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std


def get_synset_dataloader_fulltrain(root:str = None,
                          batch_size: int = 64,
                          num_workers: int = 8,
                          transform:transforms = None):
    synset = ImageFolder(
        root = root,
        transform = transform,
    ) 
    class_to_index = synset.class_to_idx
    synset_loader = DataLoader(synset,
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers = num_workers)
    return synset_loader, class_to_index

"""From https://github.com/pytorch/vision/blob/e92b5155d7d78363826a822ccacf7c046d19245a/torchvision/datasets/folder.py#L95"""

class CustomCIFAR10Dataset(ImageFolder):
    def find_classes(self, dir):
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship",	"truck"] 
        class_to_idx = {cls:i for i,cls in enumerate(classes)}
        return classes, class_to_idx

def syn_to_cifar_transform() -> transforms:
    transform = [transforms.Resize((32, 32))]
    transform += [transforms.RandomHorizontalFlip()]
    transform += [transforms.ToTensor()]
    transform += [transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])]
    return transforms.Compose(transform) 
    




class CustomDataset(Dataset):
    def __init__(self, root:str, transform:transforms, subsample:int):
        """
        root: str, the root path for the synthetic dataset
        subsample: int , the number of subsampled image in each category
        """
        # assume that each category contains equal number of image
        self.root = root 
        categories = os.listdir(root)
        num_images = len(os.listdir(os.path.join(root, categories[0])))
        self.categories = categories 
        self.num_images = num_images 
        self.subsample = min(subsample, num_images)
        self.transform = transform 
    def __len__(self):
        return len(self.categories) * self.subsample
    def __getitem__(self, idx):
        # the naming convention for each image in each subfolder of class should xxx.png(eg. 0.jpg, 1.jpg), 
        class_idx = idx // self.subsample 
        sub_idx = idx % self.subsample 
        image_path = os.path.join(self.root, self.categories[class_idx], str(sub_idx) + ".jpg")
        img = cv2.imread(image_path)
        if self.transform:
            img = self.transform(img)
        img = torch.tensor(img, dtype = torch.uint8)
        return img, class_idx 

if __name__ == "__main__":
    ds = CustomDataset(root = "/home/shenhaoyu/github_projects/KD_training/generate_results_basic_single_objectCIFAR100",
                       transform = None,
                       subsample = 10)
    dl = DataLoader(ds, batch_size = 64, shuffle= True, num_workers = 4)
    for img, label in dl:
        print(img)
        print(label)
        break 