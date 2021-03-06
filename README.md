# Pedestrian Attribute Recognition

## Description   
A baseline for pedestrian attribute recognition aimed at reproducibility using [PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning). The experiments in this repo can be easily reproduced (seeded) and extended.

Supports:
- torchvision backbones
- Apex multi-precision training
- Multi-GPU training

The experiments in this repo is carried out on PyTorch 1.3.1 and torchvision 0.4.2. The results may differ across different PyTorch version even though training is seeded.

## Requirements
- Python >= 3.7.5
- PyTorch >= 1.2.0
- torchvision >= 0.4.2

## How to run
First, install dependencies   
```bash
# Clone project   
git clone https://github.com/xingzhaolee/pedestrian-attribute-recognition

# Install project   
cd pedestrian-attribute-recognition
pip install -e .   
pip install -r requirements.txt
 ```   

Next, generate the training and testing list.
```bash
# PETA
python scripts/preprocess/peta.py PATH_TO_PETA

# RAP
python scripts/preprocess/rap.py PATH_TO_RAP
```

Next, navigate to the chosen implementation in `par/implementations/` and run it. Refer to the `README.md` in the respective folder for more detailed explanations.
```bash
python par/implementations/CHOSEN_IMPLEMENTATION/trainer.py

# Tensorboard visualization
tensorboard --logdir PATH_TO_OUTPUT_DIR
```

## Implementations      
- [Baseline](https://github.com/xingzhaolee/pedestrian-attribute-recognition-lightning/tree/master/par/implementations/baseline)


## Folder Structure    
- `par/` contains the research codes

    - `implementations/` contains various implementations of different algorithm

    - `common/` contains common modules such as backbone architecture, layers and etc. which is reused across all implementations

- `scripts/` contains various useful scripts

    - `preprocess/` contains preprocessing scripts for different dataset

## Datasets
Please refer to [this](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch) repo by dangweili on how to download the PETA and RAP dataset.
