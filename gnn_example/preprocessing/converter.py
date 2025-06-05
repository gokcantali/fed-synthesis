import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import average
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.transforms.line_graph import LineGraph

from gnn_example.preprocessing.encoder import ip_encoder, string_encoder
from gnn_example.util.constants import PROJECT_ROOT
from gnn_example.util.load_data import load_data, save_graph_data
from gnn_example.preprocessing.preprocesser import preprocess_X, construct_port_scan_label


def create_knn_graph(X, k=5):
    A = kneighbors_graph(
        X.values, n_neighbors=k, mode="connectivity",
        include_self=True, n_jobs=-1)
    A = A.tocoo()
    row = A.row.astype(np.int64)
    col = A.col.astype(np.int64)
    edge_index = np.vstack([row, col])
    return torch.tensor(edge_index, dtype=torch.long)


def convert_to_graph(X, y):
    edge_index = create_knn_graph(X)
    data = Data(x=torch.tensor(X.values, dtype=torch.float),
                      edge_index=edge_index,
                      y=torch.tensor(y.values, dtype=torch.long))

    return data


def create_randomly_partitioned_knn_graphs(df):
    graph = convert_to_graph(
        X=preprocess_X(df),
        y=df['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
    )

    return RandomNodeLoader(graph, num_parts=150, shuffle=True)


def create_graph_from_dataset(dataset_file_path: Path):
    df = load_data(file_path=dataset_file_path, sampling_ratio=1)
    df = construct_port_scan_label(df, use_diversity_index=True)
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)
    graph = convert_to_graph(
        X=preprocess_X(df, use_diversity_index=True),
        y=df['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
    )
    return graph


def create_graphs_from_existing_datasets():
    dataset_list = os.listdir(f"{PROJECT_ROOT}/data/")
    dataset_graph_list = os.listdir(f"{PROJECT_ROOT}/data/graph")

    for ds_file_name in dataset_list:
        if not ds_file_name.endswith(".csv"):
            continue

        graph_file_name = ds_file_name.replace(".csv", ".pt")
        if graph_file_name in dataset_graph_list:
            continue

        graph = create_graph_from_dataset(
            dataset_file_path=Path(f"{PROJECT_ROOT}/data/{ds_file_name}"),
        )
        save_graph_data(graph, path=f"{PROJECT_ROOT}/data/graph/{graph_file_name}")
