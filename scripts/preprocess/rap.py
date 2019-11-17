import json
import os
import os.path as osp
import sys

import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    root = sys.argv[1]

    path_dataset = osp.join(root, "dataset")
    path_image = osp.join(root, "images")

    os.makedirs(path_dataset, exist_ok=True)

    data = loadmat(osp.join(root, "RAP_annotation.mat"))

    images, attributes = [], []
    for idx in range(41585):
        images += [osp.join(
            path_image, data['RAP_annotation'][0][0][5][idx][0][0])]
        attributes += [data['RAP_annotation'][0][0][1][idx, :51]]
    images = np.array(images)
    attributes = np.array(attributes)

    splits = {}
    splits['train'] = (data['RAP_annotation'][0][0][0][0]
                       [0][0][0][0][0, :] - 1).tolist()
    splits['test'] = (data['RAP_annotation'][0][0][0][0]
                      [0][0][0][1][0, :] - 1).tolist()
    del data

    for split in ['train', 'test']:
        with open(osp.join(root, "dataset", f"{split}.txt"), 'w') as f:
            paths = images[splits[split]]
            labels = attributes[splits[split]]

            if split == 'train':
                ratios = np.mean(labels == 1, axis=0)
                np.save(osp.join(root, "dataset", "positive_ratios"), ratios)

            for path, label in zip(paths, labels):
                img = {}
                img['path'] = path
                img['attributes'] = label.tolist()
                img = json.dumps(img)
                f.write(img + '\n')
