# DeFedGCN: Privacy-preserving Decentralized Federated GCN for Recommender System

Implemented based on PyTorch

***
- The copyright of the code in this project belongs to the authors of the DeFedGCN paper (Submitted to INFOCOM 2024) and can only be used for academic research. If you need to quote or reprint, please indicate the source and the original link.

- The final interpretation right of this copyright and disclaimer belongs to the authors.
***

## Contents

### Data_processing.py 
Local data partition for clients.

### Parameters.py
Define the basic parameters for the implementation of DeFedGCN.

### LightGCN_model.py
Define the basic LightGCN model in DeFedGCN.
For more details about origninal LightGCN,please refer to [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401063)

### DeFedGCN_main.py 
The entire implementation of DeFedGCN, including the training and aggregation processes.

### Client_local_training.py 
Local training on clinets local data (sub-graphs)

### Dataset.py 
Define methods of data loading.

### Evaluation_DeFedGCN.py 
Evaluation for DeFedGCN and corresponding attacks.

### attack_main.py 
Attack experiment main program entrance, attack experiment preliminary preparation

### attack_model.py 
Define the basic attack model.
For more details, please refer to [Model Inversion Attacks against Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9895303)

### attack_optimizer.py 
Model inversation attack proceedings.

### Subgraph_expansion.cpp
Sub-graph expansion in DeFedGCN

### Communication.py  
Generates the communication topology for all clients


![cfab5359f3df42e09a0f3dcf823af00](https://github.com/kqkq926/DeFedGCN/assets/97420917/3085060b-f43f-4db3-839d-e30552b0420f)








