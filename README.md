# README

This repository contains source code for machine learning models used to predict thermal conductivity based on phonon density of states. The models implemented include:

- A simple CNN model.
- A double attention model with spatial and channel attention blocks.
- A single spatial attention model with masking.

See `work_flow.ipynb`
## Requirements

- Move the `ntwk` folder to your `/env/lib/python3.XX/site-packages/` directory. 
  Note: The `ntwk` module is not available on `PyPI`, so it cannot be installed using `pip`.

- Keep other custom-defined Python files in the same location as `model.py` and `train.py` for proper functionality.

## Data

Raw data is not provided in this repository. Experiment thermal conductivity values form Springer Material is collectd in `new_set_tcd.csv`. However, we outline the data preprocessing steps:
- The dataset consists of 299 phonon density of states plots paired with experimental thermal conductivity values.
- The files prefixed with "train" are mapped to specific models and demonstrate how the data is processed.

## Pre-trained Models

There are some pre-trained models that you can find in the folders which end with the respective model names.

## How to Run

Use the following command to run the models:

```bash
python3 train_xx.py -c config.yaml
```
## Configuration

The `config.yaml` file contains the configuration details needed for training. Refer to the original `train.py` file for explanations of the configuration options.

