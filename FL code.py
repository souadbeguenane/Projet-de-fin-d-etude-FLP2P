import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

# Importez la classe FedAvg du premier fichier ici
from fed_avg import FedAvg  # Assurez-vous de remplacer 'premier_fichier' par le nom de votre premier fichier

# Génération de données aléatoires pour l'exemple
np.random.seed(0)
data = np.random.rand(20, 2)

def Algo_kNN(nodes, k, R):
    k0 = 1
    d_max = 2
    clusters = [nodes]

    while d_max > R and k0 < k:
        k0 += 1
        kmeans = KMeans(n_clusters=k0, init='k-means++', n_init=1).fit(nodes)
        centers = kmeans.cluster_centers_
        distances = np.linalg.norm(nodes[:, np.newaxis, :] - centers, axis=2)
        nearest_centers = np.argmin(distances, axis=1)
        clusters = [nodes[nearest_centers == i] for i in range(k0)]
        d_max = np.max([np.linalg.norm(cluster - centers[i]) for i, cluster in enumerate(clusters)])

    return clusters, centers

# la distance (rayon integré dans la KNN)

k = 8
R = 0.05

clusters, centers = Algo_kNN(data, k, R)

# Créez une instance de la classe FedAvg avec les paramètres nécessaires
args = {
    "data_root": "votre_chemin_vers_les_données",
    "n_clients": 10,  # Réglez cela en fonction de votre nombre de clusters
    "n_shards": 2,    # Réglez cela en fonction de votre nombre de shards
    "non_iid": 0,     # 0 pour IID, 1 pour Non-IID
    "model_name": "cnn",  # ou "mlp" en fonction de votre modèle
    "device": 0,  # Utilisez 0 pour CUDA s'il est disponible, sinon "cpu"
    "lr": 0.01,    # Taux d'apprentissage
    "momentum": 0.9,
    "n_client_epochs": 5,
    "n_epochs": 50,
    "frac": 0.1,  # Fraction de clients sélectionnés à chaque tour
    "log_every": 10,
    "early_stopping": False
}

for cluster_idx, cluster_data in enumerate(clusters):
    print(f"Training cluster {cluster_idx + 1}/{len(clusters)}")
    
    # Créez une instance de la classe FedAvg pour ce cluster
    fed_avg = FedAvg(args)
    
    # Mettez à jour le DataLoader de FedAvg avec les données du cluster actuel
    fed_avg.train_loader = DataLoader(cluster_data, batch_size=128)
    
    # Appliquez la formation fédérée sur ce cluster
    fed_avg.train()

print("Tous les clusters ont été entraînés.")