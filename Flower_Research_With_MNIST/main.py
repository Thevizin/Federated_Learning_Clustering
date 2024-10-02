import pickle
from pathlib import Path
import psutil

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

def get_strategy(cluster_clients, server_id, cfg, testloader):
    class ClusterFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            server_result = [res for cid, res in results if cid == server_id]
            if server_result:
                aggregated_weights = super().aggregate_fit(rnd, results, failures)
                return aggregated_weights
            else:
                return super().aggregate_fit(rnd, results, failures)
            
    return ClusterFedAvg(
        fraction_fit=1.0,
        min_fit_clients=len(cluster_clients),
        min_evaluate_clients=len(cluster_clients),
        min_available_clients=len(cluster_clients),
        on_fit_config_fn=get_on_fit_config(cfg.config_fit), 
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )

def get_cluster_strategy(cluster_clients, server_id, cfg, testloader):

    class ClusterFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            # A agregação será feita apenas pelo servidor local do cluster
            server_result = [res for cid, res in results if cid == server_id]
            if server_result:
                aggregated_weights = super().aggregate_fit(rnd, results, failures)
                return aggregated_weights
            else:
                return super().aggregate_fit(rnd, results, failures)
    
    return ClusterFedAvg(
        fraction_fit=1.0,
        min_fit_clients=len(cluster_clients),
        min_evaluate_clients=len(cluster_clients),
        min_available_clients=len(cluster_clients),
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )



def create_clusters(client_weights, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(client_weights)
    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(idx)
    return clusters

def select_server_for_cluster(cluster):
    min_usage = float('inf')
    selected_server = None
    
    for client in cluster:
        cpu_usage, memory_usage = client.get_point()
        total_usage = cpu_usage + memory_usage
        if total_usage < min_usage:
            min_usage = total_usage
            selected_server = client
    
    return selected_server

def initial_train(clients, cfg, num_rounds: int = 1):
    print(f"Beginning initial training for {len(clients)} clients with {num_rounds} rounds.")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=len(clients),
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
        min_available_clients=len(clients),
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=None,  # Não precisamos de avaliação durante o treinamento inicial
    )

    client_ids = [str(i) for i in range(len(clients))]
    
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

def train_clusters(clients, server_client, num_rounds, cluster_id, all_results, testloader, cfg):
    # Configure a estratégia do servidor local para o cluster
    strategy = get_cluster_strategy(clients, server_client, cfg, testloader)
    
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    cluster_results = {"history": history, "anythingelse": "here"}
    
    all_results[f"cluster_{cluster_id}"] = cluster_results


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # Preparar datasets
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    # Gerar a função de criação de clientes
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)
    
    # Criar e armazenar os clientes em uma lista
    clients = [client_fn(str(cid)) for cid in range(cfg.num_clients)]
    clients_cpu_memory_array = {cid: [] for cid in range(cfg.num_clients)}  # Inicializar dicionário para armazenar os dados de CPU e memória

    for cid in range(cfg.num_clients):
        client = client_fn(str(cid))
        cpu_memory = client.get_point()
        clients_cpu_memory_array[cid].append(cpu_memory)

    # Realizar o treinamento inicial de todos os clientes
    initial_train(clients, cfg, num_rounds=cfg.initial_rounds)

    for cid in range(cfg.num_clients):
        client = client_fn(str(cid))
        cpu_memory = client.get_point()
        clients_cpu_memory_array[cid].append(cpu_memory)

    # Obter os pesos dos clientes após o treinamento inicial
    client_weights = [client.get_weights() for client in clients]

    for cid in range(cfg.num_clients):
        client = client_fn(str(cid))
        cpu_memory = client.get_point()
        clients_cpu_memory_array[cid].append(cpu_memory)


    # Criar clusters com base nos pesos dos clientes
    clusters = create_clusters(client_weights, cfg.num_clusters)

    for cid in range(cfg.num_clients):
        client = client_fn(str(cid))
        cpu_memory = client.get_point()
        clients_cpu_memory_array[cid].append(cpu_memory)

    # Dicionário para armazenar todos os resultados
    all_results = {"clusters": clusters}

    # Treinar os clusters usando o cliente servidor respectivo
    for cluster_id, client_ids in clusters.items():
        cluster_clients = [clients[cid] for cid in client_ids]
        selected_server = select_server_for_cluster(cluster_clients)
        train_clusters(cluster_clients, selected_server, cfg.num_rounds, cluster_id, all_results, testloader, cfg)

    for cid in range(cfg.num_clients):
        client = client_fn(str(cid))
        cpu_memory = client.get_point()
        clients_cpu_memory_array[cid].append(cpu_memory)

    all_results["clients_cpu_memory"] = clients_cpu_memory_array

    # Salvar os resultados
    results_path = Path(save_path) / "results.pkl"
    with open(str(results_path), "wb") as h:
        pickle.dump(all_results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

