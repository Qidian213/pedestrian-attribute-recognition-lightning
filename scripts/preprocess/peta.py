import json
import os
import os.path as osp
import sys

import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    root = sys.argv[1]

    os.makedirs(osp.join(root, "dataset"), exist_ok=True)

    dataset = {'att': [], 'image': []}
    data = loadmat(osp.join(root, "PETA.mat"))

    with open(osp.join(root, "dataset", "attributes.txt"), 'w') as f:
        for idx in range(35):
            f.write(data['peta'][0][0][1][idx, 0][0] + "\n")

    images = []
    attributes = []
    for idx in range(19000):
        images += [osp.join(root, "images", f"{(idx + 1):05d}.png")]
        attributes += [data['peta'][0][0][0][idx, 4:4 + 35].tolist()]
    images = np.array(images)
    attributes = np.array(attributes)

    splits = {}
    splits['train'] = (data['peta'][0][0][3][0][0][0][0][0][:, 0] - 1).tolist()
    splits['test'] = (data['peta'][0][0][3][0][0][0][0][2][:, 0] - 1).tolist()
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
