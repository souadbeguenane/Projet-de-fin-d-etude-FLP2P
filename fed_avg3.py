from multiprocessing import Process
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from data import MNISTDataset, FederatedSampler
from models import CNN, MLP
from utils import arg_parser, average_weights, Logger


class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]): #Initialisation des arguments ,le dispositif de calcul , les ensembles de données et le modèle
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger = Logger(args)

        self.train_loader, self.test_loader = self._get_data( #Obtient les ensembles de données d'entraînement et de test en appelant la méthode _get_data
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
        )

        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=1, n_classes=10).to(self.device)
            self.target_acc = 0.99
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int

    def _get_data( #Charge les données d'entraînement et de test
        self, root: str, n_clients: int, n_shards: int, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            n_shards (int): number of shards.
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        train_set = MNISTDataset(root=root, train=True)
        test_set = MNISTDataset(root=root, train=False)

        sampler = FederatedSampler(
            train_set, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards
        )

        train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=128)

        return train_loader, test_loader

    def _train_client( #Entraine le modèle pour un client spécifique et retourne le modèle et la perte moyenne
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        model = copy.deepcopy(root_model)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

        for epoch in range(self.args.n_client_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                logits = model(data)
                loss = F.nll_loss(logits, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= idx
            epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
                end="\r",
            )

        return model, epoch_loss / self.args.n_client_epochs

   

    def test(self) -> Tuple[float, float]:#Tester le modèle du serveur et retourne la perte et la précision moyennes.
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.root_model(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx
        total_acc = total_correct / total_samples

        return total_loss, total_acc


class FedAvgClustered(FedAvg):
   

    def cluster_clients_distance(self, K: int, n_clients: int, R: float) -> List[List[int]]:
        """Cluster clients using distance based clustering.

        Args:
            K (int): Number max of clusters.
            n_clients (int): Total number of clients.
            R (float): Maximum distance for communication within a cluster.

        Returns:
            List[List[int]]: List of cluster assignments for each client.
        """
        # Générer des caractéristiques de clustering  pour chaque client
        client_features = np.random.rand(n_clients, 2)

        k0 = 1
        d_max = 2

        while d_max > R and k0 < K:
            k0 += 1

            # Initialize cluster centers
            kmeans = KMeans(n_clusters=k0, init='k-means++', n_init=1).fit(client_features)
            cluster_centers = kmeans.cluster_centers_ #les centres trouvé en utilisant le Kmeans

            # Assign clients to the nearest cluster center
            distances = np.linalg.norm(client_features[:, np.newaxis, :] - np.array(cluster_centers), axis=2)# pour calculer les distances entre chaque point et chaque centre de cluster
            nearest_centers = np.argmin(distances, axis=1)# détermine le centre de cluster le plus proche pour chaque point

            # Organize clients into cluster lists
            clusters = [[] for _ in range(k0)]
            for client_idx, cluster_idx in enumerate(nearest_centers):
                clusters[cluster_idx].append(client_idx)

            d_max = np.max([np.linalg.norm(client_features[cluster] - cluster_centers[i]) for i, cluster in enumerate(clusters) ]) #calcule la nouvelle distance maximale entre les points et les centres dans les clusters actuels
            
        return clusters

    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)
        self.client_weights_norms = {}  # Stocke la norme des poids pour chaque client
        self.global_avg_weights_norm = None
        # Utilize custom distance-based clustering
        self.cluster_clients = self.cluster_clients_distance(
            self.args.K, self.args.n_clients, self.args.R
        )

    def train_cluster(self, cluster_clients: List[int]) -> None: #Entraîne les modèles des clients au niveau d'un cluster spécifique
        """Train a server model for a specific cluster of clients."""
        train_losses = []

        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []

            # Train clients in the cluster
            self.root_model.train()

            for client_idx in cluster_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                self.client_weights_norms[client_idx] = client_model.state_dict()['fc.2.weight'].cpu().numpy()
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            self.root_model.load_state_dict(updated_weights)
            self.global_avg_weights_norm = self.root_model.state_dict()['fc.2.weight'].cpu().numpy()

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.test()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    "test/acc": total_acc,
                    "round": epoch,
                }
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )

                self.logger.log(logs)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break

    

if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvgClustered(args)

    cluster_results = []  # Structure de données pour stocker les résultats de chaque cluster

    for cluster_idx, cluster_clients in enumerate(fed_avg.cluster_clients):
        print(f"Training Cluster {cluster_idx}: {cluster_clients}")
        fed_avg.train_cluster(cluster_clients)
        total_loss, total_acc = fed_avg.test()  # Tester le modèle pour ce cluster
        cluster_results.append((cluster_idx, cluster_clients, total_loss, total_acc))  # Stocker les résultats
        print(f"\n--- Global Average Weights Norm cluster {cluster_idx}")
        print(fed_avg.global_avg_weights_norm)
    # Afficher les résultats de tous les clusters en même temps
    for cluster_idx, cluster_clients, total_loss, total_acc in cluster_results:
        print(f"Cluster {cluster_idx} (Clients {cluster_clients}) | Test Loss: {total_loss} | Test Accuracy: {total_acc}")
    for client_id, weight_norm in fed_avg.client_weights_norms.items():
        print(f"Client {client_id}: {weight_norm}")
    
