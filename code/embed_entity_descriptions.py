from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import spacy
import json


nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

text_file = '../data/wikidata5m_inductive/wikidata5m_text.txt'

num_entities = sum(1 for _ in open(text_file))

uri_to_id = {}
page_links = {}
x = torch.zeros((num_entities, 384))
first_sentences = []

with open(text_file) as descriptons_file_in:
    for i, line in enumerate(tqdm(descriptons_file_in.readlines())):
        line = line[:-1]
        uri, description = line.split('\t', 1)
        doc = nlp(description)
        uri_to_id[uri] = i

        for sent in doc.sents:
            first_sentences.append(sent.text)
            embeddings = model.encode([sent.text])[0]
            x[i] = torch.tensor(embeddings)
            break

        page_links[uri] = []
        all_linked_entities = doc._.linkedEntities
        for entity in all_linked_entities:
            subj_uri = 'Q' + str(entity.get_id())
            page_links[uri].append(subj_uri)

page_links_id = []
for head, tails in page_links.items():
    for tail in tails:
        if tail in uri_to_id:
            page_links_id.append([uri_to_id[head], uri_to_id[tail]])

page_links_tensor = torch.tensor(page_links_id)


torch.save(x, '../data/wikidata5m_inductive/entity_description_first_sentence_embedding.pt')

torch.save(page_links_tensor, '../data/wikidata5m_inductive/page_links.pt')


with open('../data/wikidata5m_inductive/uri_to_id.json', 'w') as entity_2_id_out:
    json.dump(uri_to_id, entity_2_id_out)


relations = set()
for graph in ['train', 'valid', 'test']:
    with open(f'../data/wikidata5m_inductive/wikidata5m_inductive_{graph}.txt') as triples_in:
        for line in triples_in:
            _, relation, _ = line[:-1].split('\t')
            relations.add(relation)

relation_uri_2_id = {uri: i for i, uri in enumerate(relations)}

with open('../data/wikidata5m_inductive/relation_uri_to_id.json', 'w') as entity_2_id_out:
    json.dump(relation_uri_2_id, entity_2_id_out)

with open('../data/wikidata5m_inductive/entity_description_first_sentence.txt', 'w') as sentences_out:
    for sentence in first_sentences:
        sentences_out.write(sentence + '\n')


