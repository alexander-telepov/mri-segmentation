# MRI segmentation 

This is repository for Neuroimaging and Machine Learning for Biomedicine course project.
The goal of the project was to investigate behaviour of segmentation network trained on healthy humans on patients with brain diseases such as epilepsy, schizophrenia etc.
![demo](demo/metrics.png?raw=true)

For more info look on prepared configuration files in `configs`. 

Analysis and visualization of results in `notebooks`.  

Train logs avaliable on [CometML](https://www.comet.ml/alexander-telepov/neuro-project/)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation

- Clone this repo:
```bash
git clone git@github.com:alexander-telepov/mri-segmentation.git
cd mri-segmentation
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchio, surface_distance).
  - Use command `pip install -r requirements.txt`.
  - Or `./<path_to_repo>/install.sh`.

### Example of train/predict for base model on HCP dataset

- Train a model:
```bash
python ./configs/hcp/train.py
```
To see more intermediate results, check out corresponding experiment pannels on CometML.
- Test the model:
```bash
python ./configs/hcp/predict.py
```
