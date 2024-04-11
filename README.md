## Overview

Reference de base de ce travail [FedAvg](https://github.com/naderAsadi/FedAvg) 

## Installation

### Dependencies

CLHive requires **Python 3.6+**.

- numpy>=1.22.4
- pytorch>=1.12.0
- torchvision>=0.13.0
- wandb>=0.12.19

### Conda Installation

```
conda env create -f environment.yml
```

## How To Use

### Major Arguments

| Flag            | Options     | Default |Info        |
| --------------- | ----------- | :-------: |----------|
| `--data_root` | String     | "../datasets/" | path to data directory |
| `--model_name`   | String | "cnn"|
|`--non_iid` | 1: Non-IID |
| `--n_clients` | Int     | 30 | number of the clients |
| `--n_shards` | Int     | 200 | number of shards |
| `--n_epochs` | Int     | 50 | total number of rounds |
| `--n_client_epochs` | Int     | 1 | number of local training epochs |
| `--batch_size` | Int     | 10 | batch size |
| `--lr` | Float     | 0.01 | leanring-rate |
| `--wandb` | Bool     | False | log the results to WandB |
| `--R` | Float     | 0.55 |


