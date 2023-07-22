# DeFedGCN: Decentralized Federated GCN Algorithm for Recommender System

Implemented based on PyTorch

***
- The copyright of the code in this project belongs to the authors of the DeFedGCN paper and can only be used for academic research. If you need to quote or reprint, please indicate the source and the original link.

- The final interpretation right of this copyright and disclaimer belongs to the authors.
***

## Contents

### Data_processing.py 
data partitioning rule

### Parameters.py
define the basic parameters of the experiment

### LightGCN_model.py
define the basic LightGCN model.for more details,Please refer to [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401063)

### DeFedGCN_main.py 
an entry to the entire DeFedGCN program,perform the entire training process and aggregation

### Client_local_training.py 
the clients performs local training

### Dataset.py 
define methods for loading data

### Evaluation_DeFedGCN.py 
define how to evaluate

### attack_main.py 
attack experiment main program entrance, attack experiment preliminary preparation

### attack_model.py 
define the basic attack model.for more details,Please refer to [Model Inversion Attacks against Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9895303)

### attack_optimizer.py 
model inversation attack





