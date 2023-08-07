import argparse
import json
import pickle
import random
import time
import os
import torch
import numpy as np
import sklearn
from numpy.random import choice
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.transforms import RemoveDuplicatedEdges
from torch_geometric.utils import k_hop_subgraph, subgraph, add_self_loops, to_undirected

import wandb
import os.path as osp
from collections import Counter
from tqdm import tqdm
from torch import nn
from torch_geometric.data import Data
from torch.optim import lr_scheduler
from sklearn.metrics.pairwise import cosine_similarity
from models import DistMultNet

# test test test jjk

@torch.no_grad()
def compute_rank(ranks):
    # print(ranks)
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


def negative_sampling(edge_index, num_nodes, eta=50, in_batch_negatives=False):
    """

    :param edge_index:
    :param num_nodes:
    :param eta:
    :param in_batch_negatives: !Attention! -> In batch negatives only work for large batch sizes otherwise the
    probability of negatively sampling a positive edge is too large. In batch negatives allow you to use a larger
    batch size without creating large graph sample for the forward pass as this creates a high overlap in the data.
    :return:
    """
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(0) * eta) < 0.5
    mask_2 = ~mask_1

    mask_1 = mask_1.to(DEVICE)
    mask_2 = mask_2.to(DEVICE)

    neg_edge_index = edge_index.clone().repeat(eta, 1)

    if in_batch_negatives:
        in_batch_entities = torch.unique(edge_index).cpu().numpy()
        neg_edge_index[mask_1, 0] = torch.from_numpy(choice(a=in_batch_entities, size=(1, mask_1.sum().cpu()))).to(
            DEVICE)
        neg_edge_index[mask_2, 1] = torch.from_numpy(choice(a=in_batch_entities, size=(1, mask_2.sum().cpu()))).to(
            DEVICE)
    elif SAMPLE:
        all_entities = torch.unique(TRAIN_EDGE_INDEX).cpu().numpy()
        neg_edge_index[mask_1, 0] = torch.from_numpy(choice(a=all_entities, size=(1, mask_1.sum().cpu()))).to(
            DEVICE)
        neg_edge_index[mask_2, 1] = torch.from_numpy(choice(a=all_entities, size=(1, mask_2.sum().cpu()))).to(
            DEVICE)
    else:
        neg_edge_index[mask_1, 0] = torch.randint(num_nodes, (1, mask_1.sum()), device=DEVICE)
        neg_edge_index[mask_2, 1] = torch.randint(num_nodes, (1, mask_2.sum()), device=DEVICE)




    return neg_edge_index


def read_lp_data(file_path, uri_to_id, relation_uri_to_id, data_sample):
    print('Preprocessing LP data:', data_sample)

    edge_index = []
    edge_type = []
    with open(file_path) as triples_in:
        for line in triples_in:
            head, relation, tail = line[:-1].split('\t')
            edge_index.append([uri_to_id[head], uri_to_id[tail]])
            edge_type.append(relation_uri_to_id[relation])

    return Data(edge_index=torch.tensor(edge_index).t(),
                edge_type=torch.tensor(edge_type))


def load_page_link_graph_inductive(file_path, test_entities):
    if os.path.isfile(INDUCTIVE_PAGE_LINK_GRAPH_PATH):
        print('Loading inductive page link graph.')
        return torch.load(INDUCTIVE_PAGE_LINK_GRAPH_PATH)
    else:
        print('Number of entities excluded:', len(test_entities))
        page_links_graph_raw = torch.load(file_path)
        mask_allowed_triples = []
        print('Start filtering the page link graph for the inductive setting.')
        for i in tqdm(range(page_links_graph_raw.size(0))):
            head, relation, tail = page_links_graph_raw[i][0].item(), page_links_graph_raw[i][1].item(), \
                                   page_links_graph_raw[i][2].item()
            # if an entity has a page link to a test entity, this link is removed if the entity itself is in
            # the training set. If the entity is also a test entity, there is no issue.
            if tail in test_entities:
                if head in test_entities:
                    mask_allowed_triples.append(True)
                else:
                    mask_allowed_triples.append(False)
            else:
                mask_allowed_triples.append(True)
        print('Ratio of triples maintained:',
              sum([1 for x in mask_allowed_triples if x]) / page_links_graph_raw.size(0))

        mask_allowed_triples = torch.tensor(mask_allowed_triples)
        page_links_graph_inductive = page_links_graph_raw[mask_allowed_triples]
        print('Page link graph filtered for the inductive setting.')
        torch.save(page_links_graph_inductive, INDUCTIVE_PAGE_LINK_GRAPH_PATH)
        return page_links_graph_inductive


def get_relevant_neighbors(graph, train_graph, node_features, selection_method='random', num_neighbors=5):
    """
    Returns a certain number of relevant neighbors of a source node based on the provided node features and the
    selection strategy. The return value is a dictionary mapping from node_idx to a list of node_idx.
    :param graph:
    :param node_features: raw node features = description embeddings (without any modification)
    :param selection_method:
    :param num_neighbors:
    :return:
    """
    neighborhood_dict_file = osp.join(DATA_PATH, 'neighborhood_dict.pt')
    if not osp.isfile(neighborhood_dict_file):
        print('Create neighborhood dict.')

        neighbor_dict = {e: [] for e in uri_to_id.values()}
        for i in tqdm(range(graph.size(0))):
            head, tail = graph[i][0].item(), graph[i][2].item()
            neighbor_dict[head].append(tail)

        with open(neighborhood_dict_file, 'wb') as o:
            pickle.dump(neighbor_dict, o, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print('Loading neighborhood dict.')
        neighbor_dict = pickle.load(open(neighborhood_dict_file, 'rb'))

    if selection_method == NO:
        return neighbor_dict

    if selection_method == RANDOM:
        print('Selecting neighbors randomly.')
        for head in tqdm(neighbor_dict.keys()):
            neighbor_dict[head] = list(set(neighbor_dict[head]))
            random.shuffle(neighbor_dict[head])
            neighbor_dict[head] = neighbor_dict[head][:num_neighbors]
        return neighbor_dict

    if selection_method == DEGREE:
        # use degree form the inductive page link graph
        print('Selecting neighbors by degree.')
        print('Building node degree dict.')
        all_tails = graph[:,
                    2].tolist()  # only consider outgoing edges because otherwise the degree would depend on the description
        degree_dict = Counter(all_tails)
        neighborhood_dict_degree = {}
        print('Ordering nodes.')
        for head, tails in tqdm(neighbor_dict.items()):
            neighborhood_dict_degree[head] = sorted(tails, key=lambda e: - degree_dict[e])[:num_neighbors]
        return neighborhood_dict_degree

    if selection_method == DEGREE_TRAIN:
        # use degree form the training graph
        all_nodes = train_graph[:, 0].tolist() + train_graph[:, 2].tolist()
        degree_dict = Counter(all_nodes)
        neighborhood_dict_degree = {}
        print('Ordering nodes.')
        for head, tails in tqdm(neighbor_dict.items()):
            neighborhood_dict_degree[head] = sorted(tails, key=lambda e: - degree_dict[e])[:num_neighbors]
        return neighborhood_dict_degree

    if selection_method == TFIDF_RELEVANCE:
        print('Select neighbors by TF-IDF similarity.')
        tfidf_loading_file = osp.join(DATA_PATH, f'tfidf_neighborhood.json')
        if not osp.exists(tfidf_loading_file):
            print('Compute neighborhood ordering by TF-IDF similarity.')
            tfidf_feature_vectors = torch.load(TFIDF_FEATURES_PATH)
            neighborhood_dict_tfidf = {}
            for head, tails in tqdm(neighbor_dict.items()):
                cosine_sim = cosine_similarity(tfidf_feature_vectors[[head] + tails])[0][1:]
                tail_to_similarity = {t: s for t, s in zip(tails, cosine_sim)}

                neighborhood_dict_tfidf[head] = sorted(tails, key=lambda tail: - tail_to_similarity[tail])

            with open(tfidf_loading_file, 'wb') as o:
                pickle.dump(neighborhood_dict_tfidf, o, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Load neighborhood ordering by TF-IDF similarity.')
            neighborhood_dict_tfidf = pickle.load(open(tfidf_loading_file, 'rb'))

        # restrict to num_neighbors
        for head, neighbors in neighborhood_dict_tfidf.items():
            neighborhood_dict_tfidf[head] = neighborhood_dict_tfidf[head][: num_neighbors]

        return neighborhood_dict_tfidf

    if selection_method == SEMANTIC_RELEVANCE:
        bert_loading_file = osp.join(DATA_PATH, f'bert_neighborhood.pt')
        if not osp.exists(bert_loading_file):
            print('Selecting neighbors by semantic similarity.')

            cos = torch.nn.CosineSimilarity(dim=1)
            neighborhood_dict_sim_ref = {}

            for head, tails in tqdm(neighbor_dict.items()):
                head_embedding = node_features[head]
                head_embedding_stacked = head_embedding.repeat((len(tails), 1))
                sim = cos(head_embedding_stacked, node_features[tails])

                sim_dict = {t: s for t, s in zip(tails, sim)}
                neighborhood_dict_sim_ref[head] = sorted(tails, key=lambda t: sim_dict[t])

            with open(bert_loading_file, 'wb') as o:
                pickle.dump(neighborhood_dict_sim_ref, o, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            neighborhood_dict_sim_ref = pickle.load(open(bert_loading_file, 'rb'))

        for head, neighbors in neighborhood_dict_sim_ref.items():
            neighborhood_dict_sim_ref[head] = neighborhood_dict_sim_ref[head][: num_neighbors]

        return neighborhood_dict_sim_ref

    if selection_method == SEMANTIC_RELEVANCE_REDUNDANCY:
        print('Selecting neighbors by semantic similarity and avoiding redundancy.')
        return {}

    raise Exception('Neighborhood selection strategy does not exist:', selection_method)


def read_entity_features(fusion, description_embeddings, node_neighbors, triples, additional_features='edge_type',
                         test_entitites=[]):
    # Prepared node features based on entity descriptions. In case of pooling, neighboring nodes are required.
    # In case of GCN, neighboring nodes are neglected. If additional_features are provided, these are concatenated after
    # pooling.

    entity_features = description_embeddings

    if fusion == POOLING or fusion == CNN:
        print('Apply pooling over entity neighborhood.')
        entity_features_pooled = torch.zeros_like(description_embeddings)
        for head, tails in tqdm(node_neighbors.items()):

            if len(tails) == 0:
                continue

            tail_features = description_embeddings[tails]
            m = nn.AvgPool1d(tail_features.size(0))
            entity_features_pooled[head] = m(tail_features.T).flatten()

        entity_features = torch.cat([description_embeddings, entity_features_pooled], dim=1)

    if fusion == ZEROS:
        entity_features_pooled = torch.zeros_like(description_embeddings)
        entity_features = torch.cat([description_embeddings, entity_features_pooled], dim=1)
    """
    else:
        # gcn or rgcn, here we do not need pooling as the features are combined in the model
        pass
    """
    if additional_features in [EDGE_TYPE, EDGE_EMBEDDING]:
        entity_relations_dict_file = osp.join(DATA_PATH, 'entity_relations_dict.pt')
        if not osp.isfile(entity_relations_dict_file):
            print('Creating entity_relations_dict.')
            # attention: only using outgoing relations
            entity_relations_dict = {}
            for i in tqdm(range(triples.size(0))):
                head, relation, tail = triples[i].tolist()
                if head not in entity_relations_dict.keys():
                    entity_relations_dict[head] = []
                entity_relations_dict[head].append(relation)
            with open(entity_relations_dict_file, 'wb') as o:
                pickle.dump(entity_relations_dict, o, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading entity_relations_dict.')
            entity_relations_dict = pickle.load(open(entity_relations_dict_file, 'rb'))

        if additional_features == EDGE_TYPE:
            print('Started adding additional edge type features.')
            x_relation_input = torch.load(RELATION_FEATURES_CLUSTER_PATH)
            x_relation_entity = torch.zeros((entity_features.size(0), x_relation_input.size(1)))

            print('Pooling relation features from one-hot encoding.')
            for head, relations in tqdm(entity_relations_dict.items()):
                relations_unique = list(set(relations))
                if len(relations_unique) == 0:
                    continue
                # like an OR combination of all one-hot-encoded relation features
                relations_features = x_relation_input[relations_unique]
                m = nn.MaxPool1d(relations_features.size(0))
                x_relation_entity[head] = m(relations_features.T).flatten()
                # x_relation_entity[head] = torch.zeros(x_relation_input.size(1))
            print('Relation type emb dim:', x_relation_entity.size(1))
            x_relation_entity = x_relation_entity - torch.mean(x_relation_entity, dim=0)
            print(entity_features.size())
            print(x_relation_entity.size())
            print(torch.cat([entity_features, x_relation_entity], dim=1).size())
            return torch.cat([entity_features, x_relation_entity], dim=1)

        if additional_features == EDGE_EMBEDDING:
            x_relation_input = torch.load(RELATION_FEATURES_DESC_PATH)
            x_relation_entity = torch.zeros((entity_features.size(0), x_relation_input.size(1)))

            print('Pooling relation features from description embeddings.')
            for head, relations in tqdm(entity_relations_dict.items()):
                relations_unique = list(set(relations))
                if len(relations_unique) == 0:
                    continue
                relations_features = x_relation_input[relations_unique]
                m = nn.AvgPool1d(relations_features.size(0))
                x_relation_entity[head] = m(relations_features.T).flatten()

            print('Relation desc emb dim:', x_relation_entity.size(1))
            return torch.cat([entity_features, x_relation_entity], dim=1)

    return entity_features


def train_standard_lp(eta=50):
    model.train()

    edge_index_batches = torch.split(TRAIN_EDGE_INDEX.t(), BS)
    edge_type_batches = torch.split(TRAIN_EDGE_TYPE, BS)

    indices = np.arange(len(edge_index_batches))
    np.random.shuffle(indices)

    loss_total = 0
    for i in tqdm(torch.tensor(indices)):
        optimizer.zero_grad()

        edge_idxs, relation_idx = edge_index_batches[i], edge_type_batches[i]

        edge_idxs_neg = negative_sampling(edge_idxs, len(uri_to_id.keys()), eta=eta,
                                          in_batch_negatives=IN_BATCH_NEGATIVES)

        heads = torch.cat([edge_idxs[:, 0], edge_idxs_neg[:, 0]], dim=0)
        rels = torch.cat([relation_idx, relation_idx.repeat(eta)])
        tails = torch.cat([edge_idxs[:, 1], edge_idxs_neg[:, 1]], dim=0)

        neighborhood_edge_index = get_batch_neighborhood_graph(heads, tails)
        # x is just a placeholder to fulfill the Data requirements
        batch_data = Data(x=torch.zeros((10,10)), edge_type=rels, edge_index=torch.stack([heads, tails], dim=0), edge_index_neighborhood=neighborhood_edge_index)
        out = model.forward(batch_data)
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx) * eta)], dim=0).to(DEVICE)

        loss = loss_function(out, gt)
        loss.backward()
        loss_total += loss.item()
        optimizer.step()

    print('loss:', loss_total / TRAIN_EDGE_INDEX.size(1))


def get_batch_neighborhood_graph(heads, tails):
    edge_index_sample = torch.tensor([])
    if FUSION == GCN:
        batch_entities = torch.unique(torch.cat([heads, tails])).to(DEVICE)
        #print('batch_entities', batch_entities.size())

        _, edge_index_sample, _, _ = k_hop_subgraph(batch_entities, 1, NEIGHBORHOOD_GRAPHS, relabel_nodes=False, flow=FLOW, directed=True)
        #print('edge_index_sample', edge_index_sample.size())  # todo debugging

        edge_index_sample = TRANSFORM(Data(x=torch.zeros((10, 10)), edge_index=edge_index_sample)).edge_index
        #print('edge_index_sample after transform', edge_index_sample.size())  # todo debugging


        if INCLUDE_TRAIN_GRAPH_CONTEXT:
            relevant_entities = torch.unique(edge_index_sample).to(DEVICE)
            induced_subgraph_edge_index, induced_subgraph_edge_type = subgraph(relevant_entities,
                                                                               TRAIN_EDGE_INDEX, TRAIN_EDGE_TYPE)

            edge_index_sample = torch.cat((edge_index_sample, induced_subgraph_edge_index), dim=1)

        # add reverse edges
        # todo removed for debugging
        edge_index_sample = to_undirected(edge_index_sample)

    return edge_index_sample


def in_k_hop(head, tail):
    _, edge_index_k_hop_head, _, _ = k_hop_subgraph(torch.tensor([head]), 1, ALL_UNDIRECTED_EDGE_INDEX, relabel_nodes=False)
    _, edge_index_k_hop_tail, _, _ = k_hop_subgraph(torch.tensor([tail]), 1, ALL_UNDIRECTED_EDGE_INDEX, relabel_nodes=False)

    if len(np.intersect1d(torch.unique(edge_index_k_hop_head).cpu(), torch.unique(edge_index_k_hop_tail).cpu())):
        print('in neighborhood:', head, tail)
        return True
    else:
        return False


def apply_re_ranking(heads, tails, out, alpha=0.05):
    print('start re_ranking')
    alphas = torch.ones_like(heads)
    for i in tqdm(range(heads.size(0))):
        if in_k_hop(heads[i], tails[i]):
            alphas[i] = 1 + alpha

    alphas = alphas.to(DEVICE)
    out_re_ranked = out.detach().clone()
    out_re_ranked = out_re_ranked * alphas

    return out_re_ranked


@torch.no_grad()
def compute_mrr_triple_scoring(model, eval_edge_index, eval_edge_type,
                               fast=False, split='train'):

    model.eval()
    ranks = []
    ranks_re_ranked = []
    num_samples = eval_edge_type.numel() if not fast else min(1000, eval_edge_type.numel())

    for triple_index in tqdm(range(num_samples)):
        (src, dst), rel = eval_edge_index[:, triple_index], eval_edge_type[triple_index]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(len(uri_to_id.keys()), dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index, data_train.edge_type),
            (data_val.edge_index, data_val.edge_type),
            (data_test.edge_index, data_test.edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        # also delete all entities that are not in the val/test set
        if split != 'train':
            tail_mask[data_train.edge_index[0]] = False
            tail_mask[data_train.edge_index[1]] = False
        if split != 'val':
            tail_mask[data_val.edge_index[0]] = False
            tail_mask[data_val.edge_index[1]] = False
        if split != 'test':
            tail_mask[data_test.edge_index[0]] = False
            tail_mask[data_test.edge_index[1]] = False

        tail = torch.arange(len(uri_to_id.keys()))[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail]).to(DEVICE)
        head = torch.full_like(tail, fill_value=src).to(DEVICE)
        eval_edge_typ_tensor = torch.full_like(tail, fill_value=rel).to(DEVICE)

        out = []
        for head_batch, eval_edge_typ_tensor_batch, tail_batch in zip(torch.split(head, BS_EVAL),
                                                                      torch.split(eval_edge_typ_tensor, BS_EVAL),
                                                                      torch.split(tail, BS_EVAL)):

            neighborhood_edge_index = get_batch_neighborhood_graph(head_batch, tail_batch)

            batch_data = Data(x=torch.zeros((10, 10)),
                              edge_index_neighborhood=neighborhood_edge_index,
                              edge_type=eval_edge_typ_tensor_batch,
                              edge_index=torch.stack([head_batch, tail_batch], dim=0))

            out_batch = model.forward(batch_data).detach().cpu()
            out.append(out_batch)

        out = torch.cat(out)
        ranks.append(compute_rank(out))
        if RE_RANK:
            out_re_ranked = apply_re_ranking(head, tail, out, alpha=0.05)
            ranks_re_ranked.append(compute_rank(out_re_ranked))

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(len(uri_to_id.keys()), dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index, data_train.edge_type),
            (data_val.edge_index, data_val.edge_type),
            (data_test.edge_index, data_test.edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        # also delete all entities that are not in the val/test set
        if split != 'train':
            head_mask[data_train.edge_index[0]] = False
            head_mask[data_train.edge_index[1]] = False
        if split != 'val':
            head_mask[data_val.edge_index[0]] = False
            head_mask[data_val.edge_index[1]] = False
        if split != 'test':
            head_mask[data_test.edge_index[0]] = False
            head_mask[data_test.edge_index[1]] = False

        head = torch.arange(len(uri_to_id.keys()))[head_mask]
        head = torch.cat([torch.tensor([src]), head]).to(DEVICE)
        tail = torch.full_like(head, fill_value=dst).to(DEVICE)
        eval_edge_typ_tensor = torch.full_like(head, fill_value=rel).to(DEVICE)

        out = []
        for head_batch, eval_edge_typ_tensor_batch, tail_batch in zip(torch.split(head, BS_EVAL),
                                                                      torch.split(eval_edge_typ_tensor, BS_EVAL),
                                                                      torch.split(tail, BS_EVAL)):

            neighborhood_edge_index = get_batch_neighborhood_graph(head_batch, tail_batch)

            batch_data = Data(x=torch.zeros((10, 10)),
                              edge_index_neighborhood=neighborhood_edge_index,
                              edge_type=eval_edge_typ_tensor_batch,
                              edge_index=torch.stack([head_batch, tail_batch], dim=0))

            out_batch = model.forward(batch_data).detach().cpu()
            out.append(out_batch)

        out = torch.cat(out)
        ranks.append(compute_rank(out))
        if RE_RANK:
            out_re_ranked = apply_re_ranking(head, tail, out, alpha=0.05)
            ranks_re_ranked.append(compute_rank(out_re_ranked))

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)

    scores = [(1. / ranks).mean().item(), ranks.mean().item(), ranks[ranks <= 10].size(0) / num_ranks, ranks[ranks <= 5].size(0) / num_ranks, ranks[ranks <= 3].size(0) / num_ranks, ranks[ranks <= 1].size(0) / num_ranks]
    #mrr, mr, hits10, hits5, hits3, hits1 = scores
    #print(f'{split} mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    if RE_RANK:
        ranks_re_ranked = torch.tensor(ranks_re_ranked, dtype=torch.float)
        scores_re_ranked = [(1. / ranks_re_ranked).mean().item(), ranks_re_ranked.mean().item(), ranks_re_ranked[ranks_re_ranked <= 10].size(0) / num_ranks, ranks_re_ranked[ranks_re_ranked <= 5].size(0) / num_ranks, ranks_re_ranked[ranks_re_ranked <= 3].size(0) / num_ranks, ranks_re_ranked[ranks_re_ranked <= 1].size(0) / num_ranks]
        mrr, mr, hits10, hits5, hits3, hits1 = scores_re_ranked
        print(f'{split} re_ranked mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    return scores


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # The following files will not be derived and have to be given:
    # ENTITY_FEATURES_PATH, PAGE_LINK_GRAPH_PATH, ENTITY_2_ID_FILE, RELATION_2_ID_FILE, RELATION_FEATURES_DESC,
    # RELATION_FEATURES_CLUSTER, TFIDF_FEATURES_PATH
    DATA_PATH = '../data/wikidata5m_inductive'
    ENTITY_FEATURES_PATH = osp.join(DATA_PATH, 'entity_description_first_sentence_embedding.pt')
    PAGE_LINK_GRAPH_PATH = osp.join(DATA_PATH, 'page_links_typed.pt')
    INDUCTIVE_PAGE_LINK_GRAPH_PATH = osp.join(DATA_PATH, 'inductive_page_links_typed.pt')
    ENTITY_2_ID_PATH = osp.join(DATA_PATH, 'uri_to_id.json')
    RELATION_2_ID_PATH = osp.join(DATA_PATH, 'relation_uri_to_id.json')
    PRE_TRAINED_KGE_PATH = osp.join(DATA_PATH, 'distmult_pretrained.pt')
    RELATION_FEATURES_DESC_PATH = osp.join(DATA_PATH, 'relation_description_embedding.pt')
    RELATION_FEATURES_CLUSTER_PATH = osp.join(DATA_PATH, 'relation_description_cluster_features.pt')
    TFIDF_FEATURES_PATH = osp.join(DATA_PATH, 'tfidf_features_inductive.pt')

    ZEROS = 'zeros'
    NO = 'no'
    POOLING = 'pooling'
    GCN = 'gcn'
    RGCN = 'rgcn'
    RANDOM = 'random'
    DEGREE = 'degree'
    DEGREE_TRAIN = 'degree_train'
    TFIDF_RELEVANCE = 'tfidf_relevance'
    SEMANTIC_RELEVANCE = 'semantic_relevance'
    SEMANTIC_RELEVANCE_REDUNDANCY = 'semantic_relevance_redundancy'
    EDGE_TYPE = 'edge_type'
    EDGE_EMBEDDING = 'edge_embedding'
    CNN = 'cnn'
    CONTINUE = 'continue'
    DEBUG = 'debug'

    parser = argparse.ArgumentParser(description='feature based link prediction')
    parser.add_argument('--mode', type=str, default="", help="debug avoids creating wandb logs, continue loads the specified model file and continues training ")
    parser.add_argument('--lr', type=float, default=.001, help="learning rate")
    parser.add_argument('--bs', type=int, default=1000, help="batch size")
    parser.add_argument('--bs_eval', type=int, default=1000, help="batch size")
    parser.add_argument('--hidden', type=int, default=500, help="hidden dim")
    parser.add_argument('--epochs', type=int, default=100, help="epochs for training")
    parser.add_argument('--model', type=str, default="", help="model path")
    parser.add_argument('--features', type=str, default="", help="model path")
    parser.add_argument('--eta', type=int, default=50, help="number of false triples to generate per true triple")
    parser.add_argument('--fusion', type=str, default="pooling", help="method to fuse node description features: no, zeros, pooling, cnn, gcn, rgcn")
    parser.add_argument('--selection', type=str, default="random", help="method to select neighbors: no, random, degree, tfidf_relevance, semantic_relevance, semantic_relevance_redundancy")
    parser.add_argument('--num_neighbors', type=int, default=5, help="number of neighbors selected as additional features")
    parser.add_argument('--additional_features', type=str, default="no", help="additional node features added after the neighborhood selection and feature processing like pooling: no, edge_type, edge_embedding")
    parser.add_argument('--cores', type=int, default=10, help="number ob CPU cores that can be used")
    parser.add_argument('--scheduler_start', type=int, default=25, help="after how many epochs to start the learning rate scheduler (if you like no scheduler, set this to -1)")
    parser.add_argument('--num_gcn_layers', type=int, default=1, help="number of GCN or R-GCN layers applied in the decoder")
    parser.add_argument('--clean', default=False, action='store_true', help="remove feature and model files from drive")
    parser.add_argument('--in_batch_negatives', default=False, action='store_true', help="generate negative samples by random sampling nodes from the batch instead of sampling from the whole graph")
    parser.add_argument('--include_train_graph_context', default=False, action='store_true', help="add edges from the train graph connecting entitiets in the neighborhood graph and between neighborhood graphs")
    parser.add_argument('--comment', type=str, default="", help="use for custom specifications implemented directly in code")
    parser.add_argument('--flow', type=str, default="target_to_source", help="target_to_source, source_to_target")
    parser.add_argument('--re_rank', default=False, action='store_true', help="re-ranking like described in SimKGC")
    # todo sample must restrict the neighbors in order to work with gcn, otherwise all entities would still be available
    parser.add_argument('--sample', default=False, action='store_true', help="use a small sample dataset")


    args = parser.parse_args()
    torch.set_num_threads(args.cores)
    HIDDEN_DIM = args.hidden
    LR = args.lr
    BS = args.bs
    EPOCHS = args.epochs
    ETA = args.eta
    FUSION = args.fusion
    SELECTION = args.selection
    NUM_NEIGHBORS = args.num_neighbors
    ADDITIONAL_FEATURES = args.additional_features
    MODE = args.mode
    SCHEDULER_START = args.scheduler_start
    NUM_GCN_LAYERS = args.num_gcn_layers
    COMMENT = args.comment
    CNN_MODE = FUSION == CNN
    IN_BATCH_NEGATIVES = args.in_batch_negatives
    INCLUDE_TRAIN_GRAPH_CONTEXT = args.include_train_graph_context
    BS_EVAL = args.bs_eval
    CLEAN = args.clean
    FLOW = args.flow
    RE_RANK = args.re_rank
    SAMPLE = args.sample

    FEATURES_FILE = args.features if args.features != '' else 'features_' + '_'.join(
        [FUSION, SELECTION, str(NUM_NEIGHBORS), ADDITIONAL_FEATURES]) + '.pt'
    FEATURES_PATH = osp.join(DATA_PATH, FEATURES_FILE)

    MODEL_FILE = args.model if args.model != '' else 'model_' + '_'.join(
        [FUSION, SELECTION, str(NUM_NEIGHBORS), ADDITIONAL_FEATURES,
         str(NUM_GCN_LAYERS)]) + '.pt'
    MODEL_PATH = osp.join(DATA_PATH, MODEL_FILE)

    NEIGHBORHOOD_GRAPHS_FILE = args.model if args.model != '' else 'neighborhood_graphs_' + '_'.join(
        [FUSION, SELECTION, str(NUM_NEIGHBORS), ADDITIONAL_FEATURES]) + '.pt'
    NEIGHBORHOOD_GRAPHS_PATH = osp.join(DATA_PATH, NEIGHBORHOOD_GRAPHS_FILE)

    config = {
        "FUSION": FUSION,
        "SELECTION": SELECTION,
        "NUM_NEIGHBORS": NUM_NEIGHBORS,
        "ADDITIONAL_FEATURES": ADDITIONAL_FEATURES,
        "LR": LR,
        "EPOCHS": EPOCHS,
        "HIDDEN_DIM": HIDDEN_DIM,
        "ETA": ETA,
        "SCHEDULER_START": SCHEDULER_START,
        "NUM_GCN_LAYERS": NUM_GCN_LAYERS,
        "IN_BATCH_NEGATIVES": IN_BATCH_NEGATIVES,
        "INCLUDE_TRAIN_GRAPH_CONTEXT": INCLUDE_TRAIN_GRAPH_CONTEXT,
        "COMMENT": COMMENT,
        "FLOW": FLOW
    }

    print(f"Using {torch.cuda.device_count()} GPUs!")
    print('--- config ---')
    for key, value in config.items():
        print(key, '=', value)

    if MODE != DEBUG:
        run = wandb.init(project="cbilp_pool", config=config)

    with open(ENTITY_2_ID_PATH) as uri_to_id_in:
        uri_to_id = json.load(uri_to_id_in)

    with open(RELATION_2_ID_PATH) as uri_to_id_in:
        relation_uri_to_id = json.load(uri_to_id_in)

    TRANSFORM = RemoveDuplicatedEdges()

    print('Start preprocessing LP dataset.')
    if not os.path.isfile(f'../data/wikidata5m_inductive/train.pt'):
        data_train = read_lp_data(f'../data/wikidata5m_inductive/wikidata5m_inductive_train.txt', uri_to_id=uri_to_id,
                                  relation_uri_to_id=relation_uri_to_id, data_sample='train')
        torch.save(data_train, '../data/wikidata5m_inductive/train.pt')
    if not os.path.isfile(f'../data/wikidata5m_inductive/val.pt'):
        data_val = read_lp_data('../data/wikidata5m_inductive/wikidata5m_inductive_valid.txt', uri_to_id=uri_to_id,
                                relation_uri_to_id=relation_uri_to_id, data_sample='valid')
        torch.save(data_val, '../data/wikidata5m_inductive/val.pt')
    if not os.path.isfile(f'../data/wikidata5m_inductive/test.pt'):
        data_val = read_lp_data('../data/wikidata5m_inductive/wikidata5m_inductive_test.txt', uri_to_id=uri_to_id,
                                relation_uri_to_id=relation_uri_to_id, data_sample='test')
        torch.save(data_val, '../data/wikidata5m_inductive/test.pt')

    start = time.time()
    if SAMPLE:
        data_train = torch.load(f'../data/wikidata5m_inductive/train_sample.pt')
        data_val = torch.load(f'../data/wikidata5m_inductive/val_sample.pt')
        data_test = torch.load(f'../data/wikidata5m_inductive/test_sample.pt')
        print('Loaded sampled LP dataset:', time.time() - start)
    else:
        data_train = torch.load(f'../data/wikidata5m_inductive/train.pt')
        data_val = torch.load(f'../data/wikidata5m_inductive/val.pt')
        data_test = torch.load(f'../data/wikidata5m_inductive/test.pt')
        print('Loaded LP dataset:', time.time() - start)

    TRAIN_EDGE_INDEX = data_train.edge_index.to(DEVICE)
    TRAIN_EDGE_TYPE = data_train.edge_type.to(DEVICE)

    if not osp.exists(FEATURES_PATH):
        start = time.time()
        print('Features will be derived.')
        # load the raw description embeddings
        x_desc = torch.load(ENTITY_FEATURES_PATH)
        print('Description embedding dim:', x_desc.size(1))
        print('Entity description embeddings loaded.')
        inductive_entities = torch.unique(torch.cat((data_val.edge_index[0],
                                                     data_val.edge_index[1],
                                                     data_test.edge_index[0],
                                                     data_test.edge_index[1]))).tolist()

        page_link_graph = load_page_link_graph_inductive(PAGE_LINK_GRAPH_PATH, inductive_entities)

        print('Page link graph loaded.')
        neighbors = get_relevant_neighbors(page_link_graph, data_train.edge_index, x_desc,
                                           selection_method=SELECTION,
                                           num_neighbors=NUM_NEIGHBORS)
        print('Relevant neighbors derived.')
        x = read_entity_features(FUSION, x_desc, neighbors, page_link_graph, additional_features=ADDITIONAL_FEATURES,
                                 test_entitites=inductive_entities)
        torch.save(x, FEATURES_PATH)
        print('Derived entity features')

        if FUSION == GCN and not osp.isfile(NEIGHBORHOOD_GRAPHS_PATH):
            print('Start deriving neighborhood graphs for GCN.')
            NEIGHBORHOOD_GRAPHS = []
            for entity_idx in tqdm(range(len(uri_to_id.keys()))):
                tails = neighbors[entity_idx]
                NEIGHBORHOOD_GRAPHS.append(torch.tensor([[entity_idx for _ in tails], tails]).int())
                #NEIGHBORHOOD_GRAPHS.append(torch.tensor([[entity_idx], [entity_idx]]))

            NEIGHBORHOOD_GRAPHS = torch.cat(NEIGHBORHOOD_GRAPHS, dim=1)
            torch.save(NEIGHBORHOOD_GRAPHS, NEIGHBORHOOD_GRAPHS_PATH)
            print('Derived neighborhood graphs.')
            print('Derivation of all features:', time.time() - start)

    print('Load features from files.')
    x = torch.load(FEATURES_PATH)
    print('Final feature dim:', x.size(1))
    x = x.to(DEVICE)
    EMBEDDING_DIM = x.size(1)

    if FUSION == GCN:
        print('Start loading neighborhood graphs for GCN.')
        NEIGHBORHOOD_GRAPHS = torch.load(NEIGHBORHOOD_GRAPHS_PATH)
        NEIGHBORHOOD_GRAPHS, _ = add_self_loops(NEIGHBORHOOD_GRAPHS, num_nodes=x.size(0))
        NEIGHBORHOOD_GRAPHS = NEIGHBORHOOD_GRAPHS.type(torch.int64).to(DEVICE)

        model = DistMultNet(EMBEDDING_DIM, len(relation_uri_to_id.keys()), HIDDEN_DIM, x, device=DEVICE, gcn='gcn', bs=BS, num_gcn_layers=NUM_GCN_LAYERS)

    else:
        model = DistMultNet(EMBEDDING_DIM, len(relation_uri_to_id.keys()), HIDDEN_DIM, x, device=DEVICE, gcn='no', bs=BS)

    if RE_RANK:
        # load page link graph
        if osp.isfile(INDUCTIVE_PAGE_LINK_GRAPH_PATH):
            PAGE_LINK_GRAPH = torch.load(INDUCTIVE_PAGE_LINK_GRAPH_PATH)
        else:
            print('File {INDUCTIVE_PAGE_LINK_GRAPH_PATH} foes not exist. Please run data pre-processing to create it. ')
            exit(0)

        PAGE_LINK_GRAPH_EDGE_INDEX = TRANSFORM(Data(x=torch.zeros(10,10), edge_index=torch.stack([PAGE_LINK_GRAPH[:,0], PAGE_LINK_GRAPH[:,2]]))).edge_index
        ALL_UNDIRECTED_EDGE_INDEX = to_undirected(torch.cat([TRAIN_EDGE_INDEX, data_val.edge_index.to(DEVICE), data_test.edge_index.to(DEVICE), PAGE_LINK_GRAPH_EDGE_INDEX.to(DEVICE)], dim=1).to(DEVICE))

    model = model.to(DEVICE)
    print('Model is set-up, start training.')

    if MODE == CONTINUE:
        print('Load pre-trained model')
        model.load_state_dict(torch.load(MODEL_PATH))

    if not os.path.isfile(MODEL_PATH) or MODE == CONTINUE or True:
        print('Start training.')
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        if SCHEDULER_START >= 0:
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.10,
                                              total_iters=EPOCHS - SCHEDULER_START)
        model.train()

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_standard_lp(ETA)
            print(f'Save model weights after {epoch} epochs:', MODEL_PATH)
            torch.save(model.state_dict(), MODEL_PATH)

            if epoch > SCHEDULER_START and SCHEDULER_START >= 0:
                scheduler.step()

            if epoch % 5 == 0:
                model.x = x

                mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                                  data_val.edge_index,
                                                                                  data_val.edge_type,
                                                                                  fast=True,
                                                                                  split='val')

                if MODE != 'debug':
                    wandb.log({"mrr": mrr, "mr": mr, "hits10": hits10, "hits5": hits5, "hits3": hits3, "hits1": hits1})
                print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:',
                      hits1)

    print('--- Final Evaluation ---')
    print('train')
    compute_mrr_triple_scoring(model, data_train.edge_index, data_train.edge_type, fast=True, split='train')
    print('val')
    compute_mrr_triple_scoring(model, data_val.edge_index, data_val.edge_type, fast=True, split='val')
    print('test')
    compute_mrr_triple_scoring(model, data_test.edge_index, data_test.edge_type, fast=True, split='test')

    if CLEAN:
        print('Start cleaning up data.')
        if osp.exists(FEATURES_PATH): os.remove(FEATURES_PATH)
        if osp.exists(MODEL_PATH): os.remove(MODEL_PATH)
        if osp.exists(NEIGHBORHOOD_GRAPHS_PATH): os.remove(NEIGHBORHOOD_GRAPHS_PATH)
