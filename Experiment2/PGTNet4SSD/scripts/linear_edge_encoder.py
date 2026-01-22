import os
import pickle
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()         
        if 'ablation' in cfg.dataset.name:
            self.in_dim = 6 
        else:
            # load edge dimension
            dataset_class_name = cfg.dataset.format.split('-')[1]
            encoder_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(os.path.dirname(encoder_dir))
            edge_dim_path = os.path.join(
                root_dir, 'datasets', dataset_class_name, 'edge_dim.pkl')
            with open(edge_dim_path, 'rb') as f:
                edge_dim = pickle.load(f)
            self.in_dim = int(edge_dim)           
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch