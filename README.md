# Federated Learning with MNIST and/or CIFAR-10

This repository contains an implementation of a Federated Learning (FL) system using the [Flower framework](https://flower.dev/), applied to the MNIST dataset. The code simulates a federated learning system where multiple clients collaboratively train a global model without sharing their local data. 

The architecture is designed to allow clients to train locally, aggregate their model weights, and cluster the clients based on model similarity. Each cluster has a designated server client responsible for aggregating model updates.

## Overview

The repository provides a detailed implementation of the following features:
- **Federated Learning** using the Flower framework.
- **Client Clustering** based on the similarity of model weights.
- **CPU and Memory Monitoring** for selecting clients to act as cluster servers.
- **Initial and Cluster-based Training** rounds.

### Key Components

- **Client**: Represents each participant in the FL system. Clients train locally using the MNIST/CIFAR-10 dataset and send model updates to the server.
- **Server**: Aggregates the models from the clients using the `FedAvg` strategy, performs model evaluation, and selects cluster servers based on resource usage.
- **Dataset**: Prepares the MNIST/CIFAR-10 dataset, splitting it across clients.
- **Model**: A simple CNN designed to classify MNIST images (Or CIFAR-10).
- **Clustering**: Clients are clustered based on the similarity of their model weights.

### Main Workflow

1. **Data Preparation**: The MNIST dataset is split into partitions and distributed across clients.
2. **Initial Training**: All clients train locally for a few rounds.
3. **Clustering**: After the initial training, clients are clustered using K-means, based on their model weights.
4. **Server Selection**: Each cluster selects a client with the lowest resource usage (CPU + Memory) to act as the server for that cluster.
5. **Cluster Training**: Clients within each cluster perform training rounds, with the cluster server aggregating the updates.
6. **Evaluation**: The global model is evaluated after training.

## Setup

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- Flower
- Hydra
- Scikit-learn
- Psutil

Install the required Python packages using:

```bash
pip install -r requirements.txt


