import time

import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, FastRGCNConv
from torch_geometric.utils import k_hop_subgraph, subgraph, add_self_loops
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveDuplicatedEdges




class ClassicLinkPredNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_norm=True, dropout=0.0):
        super().__init__()
        self.batch_norm = batch_norm
        self.fc0 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation, h_tail, head_idx=None, rel_idx=None, tail_idx=None):
        out = torch.cat([h_head, h_relation, h_tail], dim=1)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc0(out))
        if self.batch_norm:
            out = self.bn0(out)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc1(out))
        if self.batch_norm:
            out = self.bn1(out)
        out = torch.sigmoid(self.fc_out(out))
        return torch.flatten(out)


class VectorReconstructionNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_norm=True, dropout=0.0):
        super().__init__()
        self.batch_norm = batch_norm
        self.fc0 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embedding_dim)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation):
        out = torch.cat([h_head, h_relation], dim=1)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc0(out))
        if self.batch_norm:
            out = self.bn0(out)
        out = torch.sigmoid(self.fc1(out))
        out = self.dropout(out)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.fc_out(out)
        return out


class DistMult (nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.head = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel = torch.nn.Embedding(num_relations, embedding_dim)
        self.tail = torch.nn.Embedding(num_entities, embedding_dim)

    def forward(self, head_idx, rel_idx, tail_idx):

        h_head = self.head(head_idx)
        h_relation = self.rel(rel_idx)
        h_tail = self.head(tail_idx)

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out

    def l3_regularization(self):
        return (self.head.weight.norm(p=3) ** 3 + self.rel.weight.norm(p=3) ** 3)


class DistMultNet (nn.Module):
    def __init__(self, embedding_dim, num_relations, hidden_dim, x, neighborhood_graphs, train_graph, include_train_graph_context, device, gcn='no', num_linear_layers=1, num_gcn_layers=1, cnn=False, bs=1000):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.x = x
        self.neighborhood_graphs = neighborhood_graphs.to(self.device)
        self.gcn = gcn
        self.num_linear_layers = num_linear_layers
        self.num_gcn_layers = num_gcn_layers
        self.cnn = cnn
        self.train_graph = train_graph
        self.edge_index_train_gpu = self.train_graph.edge_index.to(self.device)
        self.edge_type_train_gpu = self.train_graph.edge_type.to(self.device)
        self.include_train_graph_context = include_train_graph_context
        print('model uses cnn:', self.cnn)
        print('GCN:', self.gcn)
        print('include_train_graph_context:', include_train_graph_context)

        # todo hopefully this will work for rgcn
        if self.gcn == 'rgcn':
            self.transform = RemoveDuplicatedEdges(key=["edge_type"])
            # todo adjust num blocks depending on the embedding_dim
            self.neighborhood_rgcn = FastRGCNConv(embedding_dim, embedding_dim, num_relations=num_relations,
                                                  num_blocks=4)
        else:
            self.transform = RemoveDuplicatedEdges()

        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.embed_rel = torch.nn.Embedding(num_relations, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.fc_head_gcn = nn.Linear(hidden_dim, hidden_dim)
        self.fc_tail_gcn = nn.Linear(hidden_dim, hidden_dim)

        self.fc_head_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_tail_2 = nn.Linear(hidden_dim, hidden_dim)

        print('embedding_dim', embedding_dim)

        self.neighborhood_gcn_1 = GCNConv(embedding_dim, hidden_dim)
        self.neighborhood_gcn_2 = GCNConv(hidden_dim, hidden_dim)

        self.apply(self._init_weights)


        num_filters = 50
        self.conv1 = torch.nn.Conv2d(1, num_filters, (2, 1), 1, 0)
        self.conv2 = torch.nn.Conv2d(1, num_filters, (2, 1), 1, 0)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.dropout = torch.nn.Dropout(0.1)
        self.bs = bs
        # linear layers after convolution
        #self.linear_3 = torch.nn.Linear(3840, self.embedding_dim)
        #self.linear_4 = torch.nn.Linear(3840, self.embedding_dim)
        # todo changed for third try
        self.linear_3 = torch.nn.Linear(19200, self.hidden_dim)
        self.linear_4 = torch.nn.Linear(19200, self.hidden_dim)

        self.batch_norm_0 = nn.BatchNorm2d(1)
        self.batch_norm_1 = nn.BatchNorm2d(1)
        self.batch_norm_2 = nn.BatchNorm2d(num_filters)
        self.batch_norm_3 = nn.BatchNorm2d(num_filters)

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
        #print(f'device: {data.batch.device}')
        d = data.batch.device


        edge_index = data.edge_index
        edge_index_neighborhood = data.edge_index_neighborhood
        #print(edge_index)
        #print(data.edge_type)
        if self.gcn == 'gcn' or self.gcn == 'rgcn':
            #batch_entities = torch.unique(torch.cat([head, tail]))

            #start = time.time()
            # todo train_graph.edge_index is used that does not contain any node selection strategy
            #subset, edge_index, mapping, edge_mask = k_hop_subgraph(batch_entities, 1, self.neighborhood_graphs, relabel_nodes=False)
            #print('PyG graph creation:', time.time() - start)

            #start = time.time()
            #edge_index = torch.cat([self.neighborhood_graphs[i] for i in batch_entities], dim=1).int()
            #transformed = self.transform(Data(x=torch.tensor([]), edge_index=edge_index))
            #edge_index = transformed.edge_index
            #print('Look-up graph creation:', time.time() - start)



            #edge_type = torch.tensor([])

            #if self.gcn == 'rgcn':
            #    edge_type = torch.cat([self.neighborhood_graphs[i].edge_type for i in relevant_entities], dim=1).int()
            #    transformed = self.transform(Data(x=torch.tensor([]), edge_index=edge_index, edge_type=edge_type))
            #    edge_type = transformed.edge_type
            #else:

            #start = time.time()
            # todo does only work for 2 layer GCN, otherwise connection might be established that we do not want
            #if self.include_train_graph_context:
            #    relevant_entities = torch.unique(edge_index).to(self.device)
            #    induced_subgraph_edge_index, induced_subgraph_edge_type = subgraph(relevant_entities, self.edge_index_train_gpu, self.edge_type_train_gpu)
            #    print('init edge index:', edge_index.size())
            #    print('context edge index:', induced_subgraph_edge_index.size())
            #    edge_index = torch.cat((edge_index, induced_subgraph_edge_index), dim=1).to(self.device)
            #    # todo graph context also contains many irrelevant connections, or at least they have to be checked - this means connections between two neighborhoods are established between which no link is predicted
            #print('Derive graph context:', time.time() - start)

            #edge_index, _ = add_self_loops(edge_index)

            #if self.gcn == 'gcn':
            #start = time.time()

            if self.x.device != d:
                self.x = self.x.to(d)
            x = self.neighborhood_gcn_1(self.x, edge_index_neighborhood)

            if self.num_gcn_layers >= 2:
                x = self.neighborhood_gcn_2(x, edge_index_neighborhood)
            #print('GCN forward pass:', time.time() - start)
            # todo not yet tested
            #if self.gcn == 'rgcn':
            #    edge_type = edge_type.to(self.device)
            #    x = self.neighborhood_rgcn(self.x, edge_index, edge_type)
            #    # todo implement self.num_gcn_layers

            # todo try without these as there are not required in the RGCN link pred approach -> evaluate those
            #start = time.time()
            h_head = self.fc_head_gcn(x[edge_index[0]])
            h_tail = self.fc_tail_gcn(x[edge_index[1]])
            #print('Linear forward pass:', time.time() - start)
            #h_head = x[head]
            #h_tail = x[tail]

        #elif self.cnn:
        #    bs = head.size(0)
        #    #xs_head = self.dropout(self.batch_norm_0(self.x[head].reshape((bs, 1, 2, -1))))
        #    #xs_tail = self.dropout(self.batch_norm_0(self.x[tail].reshape((bs, 1, 2, -1))))
        #    xs_head = self.x[head].reshape((bs, 1, 2, -1))
        #    xs_tail = self.x[tail].reshape((bs, 1, 2, -1))


        #    xs_head = self.conv1(xs_head)
        #    xs_tail = self.conv1(xs_tail)

        #    xs_head = torch.sigmoid(xs_head)
        #    xs_tail = torch.sigmoid(xs_tail)
            #xs_head = self.dropout(torch.relu(self.batch_norm_2(xs_head)))
            #xs_tail = self.dropout(torch.relu(self.batch_norm_2(xs_tail)))

        #    xs_head = xs_head.view(bs, -1)
        #    xs_tail = xs_tail.view(bs, -1)

        #    h_head = self.linear_3(xs_head)
        #    h_tail = self.linear_4(xs_tail)
        else:
            h_head = self.fc_head(self.x[edge_index[0]])
            h_tail = self.fc_tail(self.x[edge_index[1]])

        h_relation = self.embed_rel(data.edge_type)  # todo use relation description embeddings
        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))

        return torch.flatten(out)


class ComplExNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_entities=0, num_relations=0, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.fc_rel = nn.Linear(embedding_dim, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.Ei = torch.nn.Embedding(num_entities, hidden_dim)
        self.Ri = torch.nn.Embedding(num_relations, hidden_dim)

        self.input_dropout = torch.nn.Dropout(0.1)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation, h_tail, head_idx=None, rel_idx=None, tail_idx=None):

        h_head = torch.sigmoid(self.fc_head(h_head))
        h_relation = torch.sigmoid(self.fc_rel(h_relation))
        h_tail = torch.sigmoid(self.fc_tail(h_tail))

        head_i = self.Ei(head_idx)
        rel_i = self.Ri(rel_idx)
        tail_i = self.Ei(tail_idx)

        if self.batch_norm:
            h_head = self.bn0(h_head)
            h_relation = self.bn1(h_relation)
            h_tail = self.bn2(h_tail)
            head_i = self.bn3(head_i)
            rel_i = self.bn4(rel_i)
            tail_i = self.bn5(tail_i)

        h_head = self.input_dropout(h_head)
        h_relation = self.input_dropout(h_relation)
        h_tail = self.input_dropout(h_tail)

        head_i = self.input_dropout(head_i)
        rel_i = self.input_dropout(rel_i)
        tail_i = self.input_dropout(tail_i)

        real_real_real = (h_head * h_relation * h_tail).sum(dim=1)
        real_imag_imag = (h_head * rel_i * tail_i).sum(dim=1)
        imag_real_imag = (head_i * h_relation * tail_i).sum(dim=1)
        imag_imag_real = (head_i * rel_i * h_tail).sum(dim=1)

        pred = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        pred = torch.sigmoid(pred)
        return pred
