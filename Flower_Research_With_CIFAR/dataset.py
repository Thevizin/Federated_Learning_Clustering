import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import CIFAR10  # Alterado para CIFAR10

def get_cifar10(data_path: str = "./data"):
    """Download CIFAR-10 and apply transformations."""
    tr = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalização do CIFAR-10
    ])

    trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = CIFAR10(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """Download CIFAR-10 and generate IID partitions."""
    
    # Download CIFAR-10 dataset
    trainset, testset = get_cifar10()
    
    # Dividir o dataset de forma que a soma seja igual ao número de amostras totais
    num_images = len(trainset) // num_partitions
    remainder = len(trainset) % num_partitions
    
    # Garantir que a soma seja igual ao número total de amostras
    partition_len = [num_images + 1 if i < remainder else num_images for i in range(num_partitions)]

    # Gerar splits de treino
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        # Gerar splits de treino e validação
        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
