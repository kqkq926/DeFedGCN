from time import time
import os 
import random 
import numpy as np
from torch.nn import MSELoss
from Dataset import Dataset, load_ml_100k
from Client_local_training import Client ,Data 
from collections import OrderedDict
from attack_model import *
from DeFedGCN.Parameters import parse_args
from attack_optimizer import PGDAttack
from scipy.sparse import csr_matrix
import pandas as pd
from Parameters import parse_args
from Evaluation_DeFedGCN import metric

class Server:
    """Define parameter
    """
    def __init__(self, path,  client_batch, local_epoch, epochs , lr ,verbose ,cilp_norm ,data_name ):
        self.epochs = epochs
        self.verbose = verbose
        self.C = cilp_norm
        self.lr = lr
        self.data_name = data_name
        self.p = path + self.data_name
        #dataset
        dataset = Dataset(path + self.data_name, self.data_name)
        self.num_users, self.num_items = dataset.get_train_data_shape()
        #model
        self.model = LGCN(self.num_users,self.num_items).cuda()
        #init clients
        self.client = Client(client_batch, local_epoch, self.data_name, path)
        #Gets all user ids
        self.all_clientId = dataset.load_all_clientId()

    """The code used to hack the experiment
    """
    def transfer_state_dict(self,pretrained_dict, model_dict):
        state_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                state_dict[k] = v
            else:
                print("Missing key(s) in state_dict :{}".format(k))
        return state_dict
    
    def init_matrix(self,adj):
        n = adj.shape[0]
        result = np.zeros((n, n))
        return sp.csr_matrix(result)
    
    #Construct the adjacency matrix
    def construct_adj(self,dataset):
        self.Graph = self.getSparseGraph(dataset)
        return self.Graph
        

    def getSparseGraph(self, dataset):
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        UserItemNet = csr_matrix((np.ones(len(dataset['itemId'])), (dataset['userId'],dataset['itemId'])),
                                      shape=(self.num_users, self.num_items))
       
        R = UserItemNet.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T

        adj_mat = adj_mat.todok()
        norm_adj = adj_mat.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce()
       
        return Graph
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
   
       

            
    def attack_run(self,path):
        learning_rate = self.lr 
        save_model_path = path+'/'+'save_model'+ '/' + self.data_name+ '/'  +'victim_model.pt'
       
        victim_model = self.model
        
        victim_model.load_state_dict(torch.load(save_model_path))
   
        Neighbor = False
        #Select any user
        id = args.seed
        id = np.array(id)
        index_Id = self.all_clientId.index(id)
        if(Neighbor == False):
            train_data = self.client.client_train_datas[index_Id]
        else:
            train_data = self.client.client_neigh_train_data[index_Id]
        
        adj = self.construct_adj(train_data)
        #initial adj is set to zero matrix
        init_adj = self.init_matrix(adj)
        init_adj = torch.FloatTensor(init_adj.todense())
        #choose the target nodes
        idx_attack = np.array(random.sample(range(len(train_data)), int(len(train_data)*args.nlabel)))
        num_edges = int(0.5 * args.density * torch.sparse.sum(adj)/len(train_data)**2 * len(idx_attack)**2) 
       
        list = []
        for i in range(len(idx_attack)):
            a = idx_attack[i]
            list.append(train_data.iloc[a,:].values)
        train_data=pd.DataFrame(list,columns = train_data.columns)

        embedding = embedding_GCN(self.num_users,self.num_items).cuda()
        embedding.load_state_dict(self.transfer_state_dict(victim_model.state_dict(), embedding.state_dict()) )

        # Setup Attack Model
        model = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0],usernodes=self.num_users, itemnodes = self.num_items, device=device )
        model = model.to(device)

        model.attack(train_data, init_adj,  num_edges,lr = learning_rate,epochs=args.attack_epochs)
        
        inference_adj = model.modified_adj.cpu()
 
        print(inference_adj)
        adj = adj.to_dense()
  
        output = metric(adj.numpy(), inference_adj.numpy(), train_data,self.num_users)
        print(output)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parse_args()
base_path = './Process-Data/data/'
ser = Server(base_path,  args.client_batch, args.local_epoch, args.epochs, args.lr, 2, 0.5, args.dataset)
ser.attack_run(args.path)
