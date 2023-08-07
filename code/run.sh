# normal without any features
done python -u classic_link_pred.py --fusion no --selection no --scheduler_start -1
done python -u classic_link_pred.py --fusion no --selection no --clean

# pooling experiments
done python -u classic_link_pred.py --fusion pooling --selection random --num_neighbors 3
done python -u classic_link_pred.py --fusion pooling --selection random --num_neighbors 5
done python -u classic_link_pred.py --fusion pooling --selection random --num_neighbors 10
done python -u classic_link_pred.py --fusion pooling --selection random --num_neighbors 20
done python -u classic_link_pred.py --fusion pooling --selection random --num_neighbors 5 --eta 10

# CNN experiment
does not work  python -u classic_link_pred.py --fusion cnn --selection random --num_neighbors 3

# zeros experiment
done python -u classic_link_pred.py --fusion zeros --selection random --num_neighbors 5

# relation features on normal and pooling
done python -u classic_link_pred.py --fusion no --additional_features edge_type
done python -u classic_link_pred.py --fusion no --additional_features edge_embedding
done python -u classic_link_pred.py --fusion pooling --selection random --num_neighbors 10 --additional_features edge_embedding

# degree pooling
done python -u classic_link_pred.py --fusion pooling --selection degree --num_neighbors 3
done python -u classic_link_pred.py --fusion pooling --selection degree --num_neighbors 5
done python -u classic_link_pred.py --fusion pooling --selection degree --num_neighbors 10

# degree_train pooling
done python -u classic_link_pred.py --fusion pooling --selection degree_train --num_neighbors 3
done python -u classic_link_pred.py --fusion pooling --selection degree_train --num_neighbors 5
done python -u classic_link_pred.py --fusion pooling --selection degree_train --num_neighbors 10

# tfidf pooling
# preprocessing on job-2
done python -u classic_link_pred.py --fusion pooling --selection tfidf_relevance --num_neighbors 3
done python -u classic_link_pred.py --fusion pooling --selection tfidf_relevance --num_neighbors 5
done python -u classic_link_pred.py --fusion pooling --selection tfidf_relevance --num_neighbors 10

# bert pooling
done python -u classic_link_pred.py --fusion pooling --selection semantic_relevance --num_neighbors 3
done python -u classic_link_pred.py --fusion pooling --selection semantic_relevance --num_neighbors 5

# in batch negatives
done python -u classic_link_pred.py --fusion no --selection no --in_batch_negatives

# GCN

worker-2 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 200 --epochs 20 --bs 1000
worker-3 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 200 --epochs 20 --bs 20000


worker-4 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 300 --epochs 20 --bs 1000
worker-5 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 300 --epochs 20 --bs 20000



python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --include_train_graph_context --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 20000


python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 20000 --flow source_to_target
python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 20000 --flow target_to_source
python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 1000 --flow source_to_target
python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 1000 --flow target_to_source




python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --include_train_graph_context --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 20000



python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 20000
python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --include_train_graph_context --num_gcn_layers 2 --hidden 300 --epochs 20 --bs 20000





memory issue python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 3 --additional_features edge_type
memory issue python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 3 --additional_features edge_embedding









worker-1 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 200 --epochs 50 --bs 1000 --bs_eval 1000
worker-2 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 200 --epochs 50 --bs 2000 --bs_eval 2000
worker-3 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 300 --epochs 50 --bs 1000 --bs_eval 1000
worker-4 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --num_gcn_layers 1 --hidden 300 --epochs 50 --bs 2000 --bs_eval 2000
worker-5 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 200 --epochs 50 --bs 1000 --bs_eval 1000
worker-6 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 200 --epochs 50 --bs 2000 --bs_eval 2000





# fc after gcn, gcn: embedding dim -> embedding dim, not directed, low eta
worker-4 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 500 --epochs 50 --bs 1000 --bs_eval 1000 --eta 5

# fc after gcn, gcn: embedding dim -> embedding dim, not directed, low eta
worker-3 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 500 --epochs 50 --bs 1000 --bs_eval 1000 --eta 10

# fc after gcn, gcn: embedding dim -> embedding dim, undirected, low eta
worker-1 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 500 --epochs 50 --bs 1000 --bs_eval 1000 --eta 1

# fc after gcn, gcn: embedding dim -> embedding dim, undirected, low eta
worker-5 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 500 --epochs 50 --bs 1000 --bs_eval 1000 --eta 5


# experiment with graph creation time
worker-6 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --num_gcn_layers 1 --hidden 500 --epochs 50 --bs 1000 --bs_eval 1000