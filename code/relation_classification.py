import argparse

from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import IterableDataset
from tqdm import tqdm
import json
import transformers
from datasets import Dataset
transformers.logging.set_verbosity_error()
import torch

"""
Uses a trained transformer model to classify the relation on two entities with a page link.
python relation_classification.py --start_idx 0 --end_idx 25000000
python relation_classification.py --start_idx 25000001 --end_idx 50000000
python relation_classification.py --start_idx 50000001 --end_idx 75000000
python relation_classification.py --start_idx 75000001 --end_idx 100109234

"""
parser = argparse.ArgumentParser(description='relation classification between wikidata entities')
parser.add_argument('--start_idx', type=int, default=0, help="")
parser.add_argument('--end_idx', type=int, default=0, help="")

args = parser.parse_args()
START_IDX = args.start_idx
END_IDX = args.end_idx


# load sentences (=entity descriptions)
sentences = []
with open('../data/wikidata5m_inductive/entity_description_first_sentence.txt') as sentences_in:
    for line in sentences_in:
        sentences.append(line[:-1])

with open('../data/wikidata5m_inductive/relation_uri_to_id.json') as uri_to_id_in:
    relation_uri_to_id = json.load(uri_to_id_in)

# load transformer model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(relation_uri_to_id.keys()))
model.load_state_dict(torch.load('../data/wikidata5m_inductive/relation_classification_model.pt'))
model.to('cuda')
model.eval()

# load page link graph (=edges to be classified)
page_links = torch.load('../data/wikidata5m_inductive/page_links.pt')

# create tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


triples_out = open(f'../data/wikidata5m_inductive/page_link_graph_{START_IDX}.txt', 'a+', buffering=1)
dataset = []
batch_edge_index = []
bs = 1000
for idx in tqdm(range(START_IDX, END_IDX + 1)):
    head, tail = page_links[idx]
    batch_edge_index.append([idx, head, tail])
    encoding = tokenizer(sentences[head], sentences[tail], padding="max_length", truncation=True)
    encoding['idx'] = idx
    dataset.append(encoding)
    if idx % bs == 0:
        d = Dataset.from_list(dataset)
        d.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
        dataloader = torch.utils.data.DataLoader(d, batch_size=bs, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=1)

        for (idx, head, tail), relation in zip(batch_edge_index, predictions.tolist()):
            triples_out.write('\t'.join([str(head.item()), str(relation), str(tail.item())]) + '\n')

        dataset = []
        batch_edge_index = []

triples_out.close()
