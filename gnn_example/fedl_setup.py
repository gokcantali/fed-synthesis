from pathlib import Path

import torch
from torch_geometric.loader import RandomNodeLoader

from gnn_example.gcn import Config, GCN
from gnn_example.util.constants import PROJECT_ROOT, VALIDATION_SIZE, TRAIN_SIZE


def get_dataset_splits(client_id):
    num_parts = 50
    removed_column_indices = [18, 19]  # Columns to be removed from the feature set

    test_graph_data = torch.load(
        Path(f'{PROJECT_ROOT}/data/graph/sample-dataset-test-client-{client_id}.pt')
    )
    for col_index in removed_column_indices:
        test_graph_data.x[:, col_index] = torch.zeros_like(test_graph_data.x[:, col_index])

    test_loader, y_true = [], []
    test_batches = RandomNodeLoader(test_graph_data, num_parts=num_parts, shuffle=True)
    for _, batch in enumerate(test_batches):
        test_loader.append(batch)
        y_true += batch.y

    train_graph_data = torch.load(
        Path(f'{PROJECT_ROOT}/data/graph/sample-dataset-train-client-{client_id}.pt')
    )
    for col_index in removed_column_indices:
        train_graph_data.x[:, col_index] = torch.zeros_like(train_graph_data.x[:, col_index])

    train_loader, validation_loader = [], []
    train_batches = RandomNodeLoader(train_graph_data, num_parts=num_parts, shuffle=True)
    for ind, batch in enumerate(train_batches):
        if ind < (VALIDATION_SIZE / (VALIDATION_SIZE + TRAIN_SIZE)) * num_parts:
            validation_loader.append(batch)
        else:
            train_loader.append(batch)

    return train_loader, validation_loader, test_loader


def initialize_model():
    config = Config()

    model = GCN(
        optimizer=config.optimizer,
        num_features=25,
        num_classes=4,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        patience=config.patience
    )
    return model
