# Corpus based Inductive Link Prediction 

## Abstract
Inductive Link Prediction plays a crucial role in Knowledge Graph Completion, focusing on predicting relations to unseen 
entities. Current approaches rely on the textual descriptions of entities to derive features for Link Prediction, 
as the graph context for the new entities is limited. However, entity descriptions are often short and may not provide 
valuable information for feature derivation.
In this paper, we propose a novel approach that goes beyond entity descriptions by leveraging extensive textual data 
sources. Our methodology incorporates the implicit knowledge graph (KG) structure derived from textual sources to 
construct a graph context for unseen entities. We demonstrate how information extraction and linguistic 
features can be utilized to derive a graph that enables Inductive Link Prediction.
To evaluate the effectiveness of our approach, we conduct experiments on two datasets specifically designed for this 
task. Our results demonstrate the value of the features derived from textual sources for Inductive Link Prediction. 
This highlights the significance of incorporating textual data and the implicit KG structure in improving the performance 
of Inductive Link Prediction.



### Files and Data Formats

* Page Link Graphs (`PAGE_LINK_GRAPH_PATH` & `INDUCTIVE_PAGE_LINK_GRAPH_PATH`) are stored as triples instead of `edge_index` and `edge_type`.
`PAGE_LINK_GRAPH_PATH` can be created by using `entity_linking.ipynb` for entity linking and `classic_link_pred.py` for classifying the link.


### ToDos
* implement re-ranking s.t. not the whole page link graph is used, instead only the page link graph of the inductive entities (as no re-ranking is done for the train entities anyway)
* implement to only connect entities in the local neighborhoods of a triple and not between neighborhoods not of the triple
* remove CNN code



