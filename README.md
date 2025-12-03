# Deep Edge Filter: Return of the Human-Crafted Layer in Deep Learning (NeurIPS 2025)

This code is the official implementation of the following paper: [Deep Edge Filter: Return of the Human-Crafted Layer in Deep Learning](https://arxiv.org/abs/2510.13865). Our work relies on the key insight that deep networks store task-relevant semantic information in high-frequency components of deep features, while domain-specific biases hide in low-frequency components. By applying classical edge filtering concepts directly to deep neural network features (not just to input data!), we can isolate the generalizable representations that matter. Unlike previous approaches that applied edge detection only to input data, our Deep Edge Filter works on intermediate feature representations across ANY architecture and modality: Vision (CNNs & ViT), Text (Transformers), 3D (NeRF), and Audio(CNN).

Here, we only cover the test-time adaptation experiment code conducted for the Vision modality. In the TTA experiments performed on Vision data, applying the Deep Edge Filter yielded performance improvements, such as a +10.2% gain on the CIFAR100-C dataset, demonstrating enhanced performance through the simple addition of a filter layer.

Ours code includes the source code of [robustbench](https://github.com/RobustBench/robustbench) and [TENT](https://github.com/DequanWang/tent). Thanks to their contributors.

## Installation
```
pip install -r requirements.txt
```

## Dataset Preperation
Prepare the clean datasets CIFAR-10, CIFAR-100, and ImageNet, along with their corresponding corrupted datasets CIFAR-10C, CIFAR-100C, and ImageNet-C. Even if you wish to experiment with only some datasets, the clean and corrupted datasets must be prepared in pairs. You can specify the root folder where the datasets are stored using the ‘--root’ argument. The experiment proceeds in the following order: the model is first trained on the clean dataset, and then the trained model is used to measure TTA accuracy on the corrupted dataset.

## Run the Code

```
# WRN backbone
python main.py --dataset cifar100 --arch WRN-28-10 --mode average --pos 1 --config config/average/v11.yaml

# ViT backbone
python main.py --dataset ImageNet --arch ViT-B32 --mode average --pos 11 --config config/average/vit_v7.yaml
```
You can also run WRN and ViT experiments through `wrn.sh` and `vit.sh`.

For the **arch** argument, which refers to the model architecture, WRN-28-10, WRN-40-2 and ViT-B32 options are available.

The **mode** argument refers to the the filter type. you can choose between none, average, median, gaussian, and average_LOW. Average here means mean filter.

The **pos** argument referes to the filter position. The filter will be attached right after the layer corresponding to the number you select. You can choose between 0 to 3 for the WRN backbone, and between 0 to 11 for the ViT backbone.

The **config** argument links to a YAML file containing various parameters required for the experiment. It primarily handles the filter kernel size and learning rate. The number in the config file indicates the filter kernel size.

## Citation
```
@inproceedings{lee2025deep,
title={Deep Edge Filter: Return of the Human-Crafted Layer in Deep Learning},
author={Dongkwan Lee and Junhoo Lee and Nojun Kwak},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/pdf?id=QcItn1s1jO}
}
```
