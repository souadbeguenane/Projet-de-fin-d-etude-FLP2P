from typing import Optional
import numpy as np
import torch
from torchvision import datasets, transforms 
#transforms : pour prétraiter les images de l'ensemble de données MNIST avant de les utiliser dans un modèle d'apprentissage


class MNISTDataset(datasets.MNIST):

    N_CLASSES = 10  

    def __init__(self, root: str, train: bool):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(), #convertit l'image d'un format de tenseur (généralement un tenseur PyTorch) en une image de format PIL (Pillow Image Library)
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        #  ces transformations sont utilisées pour convertir les images MNIST en un format adapté à l'apprentissage automatique en PyTorch. Elles assurent que les images sont dans le bon format (tenseur) et sont prétraitées (normalisées) pour que les valeurs de pixel aient des statistiques appropriées pour l'entraînement de modèles
        super().__init__(root=root, train=train, download=True, transform=transform)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)

        return x, y
