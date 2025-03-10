# Dynamic Graph Convolution and Spatio-Temporal Self-Attention Network for Traffic Flow Prediction

This is a PyTorch implementation of Dynamic Graph Convolution and Spatio-Temporal Self-Attention Network for Traffic Flow Prediction.

## Requirements

Our code is based on Python version 3.9.7 and PyTorch version 1.10.1. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:

```shell
pip install -r requirements.txt
```

## Data

The dataset link is [Google Drive](https://drive.google.com/drive/folders/1MMQAaL8G31CjXdK1RQXNk8p4XhoBm7IW?usp=drive_link). You can download the datasets, unzip them and place them in the `raw_data` directory.

## Train

You can train **DGSTA** through the following commands. Parameter configuration (**--config_file**) reads the JSON file in the root directory. If you need to modify the parameter configuration of the model, please modify the corresponding **JSON** file.

```shell
python run_model.py --task traffic_state_pred --model DGSTA --dataset PeMS04 --config_file PeMS04
```

## Acknowledgement

Our code is modified based on [Libcity](https://github.com/LibCity/Bigscity-LibCity), thanks for their contribution.