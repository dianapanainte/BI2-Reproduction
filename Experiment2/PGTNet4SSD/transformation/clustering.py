# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:55:04 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pickle
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import torch

def load_graphs(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def extract_features(train_data, val_data, test_data):
    # Step 1: Compute maximum node label across all datasets
    all_data = train_data + val_data + test_data
    max_node_label = max([int(d.x.max().item()) for d in all_data])
    # Step 2: Feature extraction function
    def extract_features(dataset):
        X, y = [], []
        for d in dataset:
            x = d.x.view(-1).long()  # Flatten to 1D
            node_hist = torch.bincount(x, minlength=max_node_label + 1).float()
            # Edge attribute summary
            mean_edge = d.edge_attr.mean(dim=0).numpy()
            std_edge = d.edge_attr.std(dim=0).numpy()
            edge_feat = np.concatenate([mean_edge, std_edge])
            # Combine node histogram + edge features
            full_feat = np.concatenate([node_hist.numpy(), edge_feat])
            X.append(full_feat)
            y.append(d.y.item() if torch.is_tensor(d.y) else d.y)
        return np.stack(X), np.array(y)
    # Step 3: Extract features for each split
    X_train, y_train = extract_features(train_data)
    X_val, y_val     = extract_features(val_data)
    X_test, y_test   = extract_features(test_data)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_clusters(X_train, X_val, y_train, y_val, threshold=0.1):
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    min_samples = int(threshold * len(X_combined))
    tree = DecisionTreeRegressor(min_samples_leaf=min_samples, random_state=42)
    tree.fit(X_combined, y_combined)
    return tree

def assign_clusters(tree, X_train, X_val, X_test):
    return (
        tree.apply(X_train).tolist(),
        tree.apply(X_val).tolist(),
        tree.apply(X_test).tolist(),
    )
    
def apply_clustering(graph_dataset_path_raw):
    # Load graph data as pickle files
    train_path = os.path.join(graph_dataset_path_raw, 'train.pickle')
    print(train_path)
    val_path = os.path.join(graph_dataset_path_raw, 'val.pickle')
    test_path = os.path.join(graph_dataset_path_raw, 'test.pickle')
    train_data = load_graphs(train_path)
    val_data = load_graphs(val_path)
    test_data = load_graphs(test_path)
    # extract features
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = extract_features(
        train_data, val_data, test_data)
    # Create clusters
    tree = create_clusters(X_train, X_val, y_train, y_val, threshold=0.1)
    # Assign clusters
    clusters_train, clusters_val, clusters_test = assign_clusters(
        tree, X_train, X_val, X_test)
    # Attach cluster IDs back to Data objects
    for data, cid in zip(train_data, clusters_train):
        data.cluster_id = cid
    for data, cid in zip(val_data, clusters_val):
        data.cluster_id = cid
    for data, cid in zip(test_data, clusters_test):
        data.cluster_id = cid
    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_data, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    # sanity check
    print(data)
    # clusters_test is a list of cluster IDs for test graphs
    cluster_counts = Counter(clusters_test)
    print("Number of clusters:", len(cluster_counts))
    print("Number of graphs per cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} graphs")
    return cluster_counts