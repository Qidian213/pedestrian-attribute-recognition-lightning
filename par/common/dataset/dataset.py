import json
import os.path as osp

import torch
from torch.utils import data
from torchvision.datasets.folder import default_loader


class Dataset(data.Dataset):

    def __init__(self, root, split, transform):
        self.root = root
        self.transform = transform

        self.dataset = []
        dataset = open(osp.join(self.root, "dataset", f"{split}.txt"), 'r')
        dataset = [x.strip() for x in dataset.readlines()]

        for d in dataset:
            d = json.loads(d)
            self.dataset += [(d['path'], d['attributes'])]

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = default_loader(img)
        label = torch.tensor(label).float()

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)
