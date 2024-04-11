## Overview

Reference de base de ce travail [FedAvg](https://github.com/naderAsadi/FedAvg) 


L'architecture du système proposé dans ce projet se structure autour de trois composantes principales. Premièrement, le clustering est utilisé afin d’organiser les clients participants au FL et faciliter la gestion des interactions et la collaboration au sein des groupes (ou clusters). Deuxièmement, l’apprentissage fédéré dans chaque cluster se base sur la technique SAC pour une collaboration sécurisée entre les clients. Troisièmement, l’intégration de la technologie blockchain joue un rôle essentiel dans la préservation de la transparence et de la traçabilité des mises à jour du modèle global tout au long du processus d’apprentissage

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
| `--model_name`   | String | "cnn"|Used Model
| `--R` | Float     | 0.55 | Distance between client
|`--non_iid` | Int (0 or 1) | 1 | 0: IID, 1: Non-IID |
| `--n_clients` | Int     | 30 | number of the clients |
| `--n_shards` | Int     | 200 | number of shards |
| `--n_epochs` | Int     | 50 | total number of rounds |
| `--n_client_epochs` | Int     | 1 | number of local training epochs |
| `--batch_size` | Int     | 10 | batch size |
| `--lr` | Float     | 0.01 | leanring-rate |
| `--wandb` | Bool     | False | log the results to WandB |



