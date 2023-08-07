import torch
from torch import nn
from torch_geometric.nn import GCNConv, FastRGCNConv


class DistMultNet (nn.Module):
    def __init__(self, embedding_dim, num_relations, hidden_dim, x, device, gcn='no', num_gcn_layers=1, bs=1000):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.x = x.to(self.device)
        self.gcn = gcn
        self.num_gcn_layers = num_gcn_layers
        self.bs = bs

        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.embed_rel = torch.nn.Embedding(num_relations, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.fc_head_gcn = nn.Linear(embedding_dim, hidden_dim)
        self.fc_tail_gcn = nn.Linear(embedding_dim, hidden_dim)

        self.neighborhood_gcn_1 = GCNConv(embedding_dim, embedding_dim)
        self.neighborhood_gcn_2 = GCNConv(embedding_dim, embedding_dim)

        self.dropout = torch.nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, data):
        """
        # todo implement hard negatives like (1,r,1) or (2,r,2) such as in SimKGC because language model for LP tend to assign high scores to these triples
        # todo think about storing the embeddings of previous batches s.t. these can be used for the nodes that are placed to create false triples
        """

        edge_index = data.edge_index
        edge_index_neighborhood = data.edge_index_neighborhood

        if self.gcn == 'gcn' or self.gcn == 'rgcn':

            x = self.neighborhood_gcn_1(self.x, edge_index_neighborhood)

            if self.num_gcn_layers >= 2:
                x = self.neighborhood_gcn_2(x, edge_index_neighborhood)

            #h_head = x[edge_index[0]]
            #h_tail = x[edge_index[1]]

            h_head = self.fc_head_gcn(x[edge_index[0]])
            h_tail = self.fc_tail_gcn(x[edge_index[1]])

        else:
            h_head = self.fc_head(self.x[edge_index[0]])
            h_tail = self.fc_tail(self.x[edge_index[1]])

        h_relation = self.embed_rel(data.edge_type)  # todo use relation description embeddings
        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))

        return torch.flatten(out)
