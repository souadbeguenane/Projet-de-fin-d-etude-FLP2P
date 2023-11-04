from multiprocessing import Process
import pickle
import random
import socket
import threading
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
            self.avrg = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
        elif self.args.model_name == "cnn":
            self.avrg = CNN(n_channels=1, n_classes=10).to(self.device)
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
        self, avrg: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            avrg (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        model = copy.deepcopy(avrg)
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
        self.avrg.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.avrg(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx
        total_acc = total_correct / total_samples

        return total_loss, total_acc


class FedAvgClustered(FedAvg):
    def create_server_socket(self,coord, port):
     #Créer un serveur socket pour chaque client dans un cluster
     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
     server_socket.bind(('127.0.0.1', port))
     server_socket.listen(1)

     while True:
            conn, addr = server_socket.accept()
            received_data = b""
            while True:
                data = conn.recv(50 * 1024 * 1024)
                if not data:
                    break
                received_data += data
            try:
                received_data = pickle.loads(received_data)
                print(f"{coord} (port {port}) received: {received_data}")
            except pickle.UnpicklingError as e:
                print(f"Error unpickling data: {e}")
            conn.close()
    
    def start_server_sockets(self):
        #Chaque serveur écoute sur un port spécifique pour les connexions entrantes
        port_index = 12345  # start port number
        self.server_threads = []
        for cluster_id, cluster_clients in enumerate(self.cluster_clients):
            for client_id in cluster_clients:
                port = port_index
                port_index += 1  # increment the port number for each client
                t = threading.Thread(target=self.create_server_socket, args=(f"Client {client_id}", port))
                t.start()
                self.server_threads.append(t)
                print(f"Client {client_id} (in Cluster {cluster_id}) listening on port {port}")
   
    def divide_weights(self, model_weights: Dict[str, torch.Tensor], n_partitions: int) -> List[Dict[str, torch.Tensor]]:
        """Divise les poids du modèle en plusieurs partitions."""
        partitions = []
        rn = [random.uniform(0, 1e6) for _ in range(n_partitions)]
        sum_rn = sum(rn)

        for i in range(n_partitions):
            prn = rn[i] / sum_rn
            partition = {k: v * prn for k, v in model_weights.items()}
            #Multiplie chaque poids du modèle par le  prn pour obtenir une partition des poids
            partitions.append(partition)  #Ajoute la partition créée à la liste des partitions
        
        return partitions
         #la somme de toutes les partitions reconstituera les poids originaux du modèle.

    def distribute_partitions(self, partitions: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        n_clients = len(partitions)
        client_weights = [{} for _ in range(n_clients)]
        
        for i, partition in enumerate(partitions): # itèrer sur les peers
            for j, weights in enumerate(partition): # itèrer sur  les  partitiondans chaque peers
                client_idx = (i + j) % n_clients  #calculer à quel client attribuer le poid 
                client_weights[client_idx][f"partition_{j}"] = weights  #attribue l'ensemble de poids actuel au client déterminé

        return client_weights

    def calcule_subtotals(self, client_weights: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        return [{k: sum([partition[k] for partition in partitions.values()]) for k in client_weights[0]["partition_0"]} for partitions in client_weights]

    def average_subtotals(self, subtotals: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n_clients = len(subtotals)
        return {k: sum([subtotal[k] for subtotal in subtotals]) / n_clients for k in subtotals[0]}
    

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

    def train_cluster(self, cluster_clients: List[int]) -> None:
        """Train a server model for a specific cluster of clients."""
        train_losses = []

        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []

            # Train clients in the cluster
            self.avrg.train()

            for client_idx in cluster_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, client_loss = self._train_client(
                    avrg=self.avrg,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                #print(client_model.state_dict().keys())
                self.client_weights_norms[client_idx] = client_model.state_dict()['fc.2.weight'].cpu().numpy()
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # Diviser les poids
            partitions = [self.divide_weights(model_weights, len(clients_models)) for model_weights in clients_models]

            # Distribuer les partitions
            distributed_weights = self.distribute_partitions(partitions)

            # Calculer les sous-totaux pour chaque client
            subtotals = self.calcule_subtotals(distributed_weights)

            # Calculer la mise à jour moyenne
            average_update = self.average_subtotals(subtotals)

            # Mettre à jour le modèle du serveur
            self.avrg.load_state_dict(average_update)
            self.global_avg_weights_norm = self.avrg.state_dict()['fc.2.weight'].cpu().numpy()

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
                print(f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n")

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
        print(f"\n--- Global Average Weights cluster {cluster_idx}")
        print(fed_avg.global_avg_weights_norm)

    # Afficher les résultats de tous les clusters en même temps
    for cluster_idx, cluster_clients, total_loss, total_acc in cluster_results:
        print(f"Cluster {cluster_idx} (Clients {cluster_clients}) | Test Loss: {total_loss} | Test Accuracy: {total_acc}")
        
    # Afficher les normes des poids
    print("\n\n--- Client Weights ---")
    for client_id, weight_norm in fed_avg.client_weights_norms.items():
        print(f"Client {client_id}: {weight_norm}")
    