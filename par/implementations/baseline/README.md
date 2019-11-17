## Baseline
Implementation of baseline following [Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios](http://dangweili.github.io/misc/pdfs/acpr15-att.pdf).

### Key ideas
- Sigmoid cross entropy to train a single model for pedestrian attribute recognition
- Using weighted cross entropy loss to tackle class imbalance

### Differences in this implementation
- Mixed precision training with Apex `opt_level='O1'` to save time on the cost of possibility of slight performance drop
- Image size is `h:256, w:228` to better represent the human body instead of square images

### Supported backbones
- ResNet variants: `resnet18, resnet34, resnet50, resnet101,
resnet152, resnext50_32x4d, resnext101_32x8d,
wide_resnet50_2, wide_resnet101_2`

### Results

#### RAP
| Backbone | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 (`dropout=0.5`) | 78.62 | 65.84 | 78.33| 78.52 | 78.42 |
| ResNet34 | 77.45 | 63.69 | 76.77 | 76.96 | 76.87 |
| ResNet18 | 76.28 | 63.50 | 76.72 | 76.89 | 76.80 |


#### PETA
| Backbone | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 (`dropout=0.5`) | 83.55 | 77.79 | 86.64 | 84.60 | 85.61 |
| ResNet34 | 83.35 | 77.49 | 86.37 | 84.52 | 85.44 |
| ResNet18 (`dropout=0.5`) | 81.45 | 76.38 | 85.93 | 83.24 | 84.56 |

### How to run
Below shows the example on how to run a ResNet50 baseline model on single and multiple GPU.

```bash
# Single GPU
python par/implementations/baseline/trainer.py --use_16bit \
    -data_dir PATH_TO_PETA \
    -num_classes 35 \
    --weighted_loss \
    -backbone resnet50 \
    -output_dir PATH_TO_OUTPUT_DIR

python par/implementations/baseline/trainer.py --use_16bit \
    -data_dir PATH_TO_RAP \
    -num_classes 51 \
    --weighted_loss \
    -backbone resnet50 \
    -output_dir PATH_TO_OUTPUT_DIR

# Use 4 GPUs
python par/implementations/baseline/trainer.py --use_16bit -gpus 4 \
    -data_dir PATH_TO_PETA \
    -num_classes 35 \
    --weighted_loss \
    -backbone resnet50 \
    -output_dir PATH_TO_OUTPUT_DIR

python par/implementations/baseline/trainer.py --use_16bit -gpus 4 \
    -data_dir PATH_TO_RAP \
    -num_classes 51 \
    --weighted_loss \
    -backbone resnet50 \
    -output_dir PATH_TO_OUTPUT_DIR
```   
