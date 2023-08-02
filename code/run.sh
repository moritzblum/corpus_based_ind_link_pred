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
slurm-958 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --hidden 200
slurm-962 (long run) python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --hidden 200 --epochs 20
job-1 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --include_train_graph_context --num_gcn_layers 2 --hidden 200

# GCN with additional linear layer
slurm-1115 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --hidden 200 --epochs 11 --comment "with linear layer"
slurm-1116 python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 5 --in_batch_negatives --include_train_graph_context --num_gcn_layers 2 --hidden 200 --epochs 11 --comment "with linear layer"





memory issue python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 3 --additional_features edge_type
memory issue python -u classic_link_pred.py --fusion gcn --selection random --num_neighbors 3 --additional_features edge_embedding


