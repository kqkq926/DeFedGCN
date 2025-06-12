from time import time
import os 
import random 
import numpy as np
from torch.nn import MSELoss
from Dataset import Dataset, load_ml_100k
from Client_local_training import Client ,Data 
from collections import OrderedDict
from LightGCN_model import *
from Parameters import parse_args
import torch
import pandas as pd
from Evaluation_DeFedGCN import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
lossfn = MSELoss()



class Server:
    """Define parameter
    """
    def __init__(self, path,  client_batch, local_epoch, epochs , lr ,verbose ,cilp_norm ,data_name = 'yahoo_music' ):
        self.epochs = epochs
        self.verbose = verbose
        self.C = cilp_norm
        self.lr = lr
        self.data_name = data_name
        self.p = path + self.data_name
        #dataset
        t1 = time()
        dataset = Dataset(path + self.data_name, self.data_name)
        self.num_users, self.num_items = dataset.get_train_data_shape()
        #true users number
        self.real_num_users = len(dataset.load_all_clientId())
        self.test_datas = dataset.load_test_file()
        print("Server Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
          % (time()-t1, self.num_users, self.num_items, len(self.test_datas)))
        #model
        self.model = LGCN(self.num_users,self.num_items).cuda()
        #init clients
        self.client = Client(client_batch, local_epoch, self.data_name, path)
        #Gets all user ids
        self.all_clientId = dataset.load_all_clientId()
        #Neighbors of all users (including their own ids)
        self.NeighId = self.client.get_Neignbor_userId()
        #id of the user to be trained
        self.client_Id = self.client.construct_userId(self.all_clientId)
        #Gets all the test sets
        self.all_test_datas = dataset.load_all_testdatas()
        #Gets all the train sets
        self.all_train_datas = dataset.load_all_traindatas()
    

    def distribute_task(self, connection, learning_rate,NeighId,userEmbdMan,Neighbor):
        """Distribute weights and get parameters for each client
        """
        client_weight_datas = []
        server_weights = OrderedDict()
        for k, v in self.model.state_dict().items():
            server_weights[k] = v
        pair = []

        client_ids = [i for i in connection.keys()]

        for index_Id in client_ids:
            train_id = self.all_clientId[index_Id]
            user_server_weights = reload_model_params(NeighId[index_Id], userEmbdMan,server_weights )
            weights = self.client.train_epoch(self.model, index_Id ,train_id ,user_server_weights, learning_rate,userEmbdMan,Neighbor)
            client_weight_datas.append(weights)

        return client_weight_datas
    
 
    def has_converged(self, client_vectors, epsilon=1e-5):
        """
        Determine convergence 
        """

        reference_vector = {key: torch.tensor(val, device=device) for key, val in client_vectors[0].items()}
    
        w = list(client_vectors[0].keys())  
        
        for vector in client_vectors[1:]: 
            current_vector = {key: torch.tensor(val, device=device) for key, val in vector.items()}
            
            for key in w:
                
                if torch.norm(current_vector[key] - reference_vector[key]).item() > epsilon:
                    return False
    
        return True

        
    
    def pair_federated_average(self, index_id, neighbor_indexid,client_weight_datas):
        """Pairwise average
        """

        rat = self.caculate_weight(index_id, neighbor_indexid,)
        w = list(client_weight_datas[index_id].keys())
        index_id, neighbor_indexid = index_id-1, neighbor_indexid-1
        for key in w:
            averaged_value = (
            client_weight_datas[index_id][key] * rat[0] +
            client_weight_datas[neighbor_indexid][key] * rat[1]
        )
    
            client_weight_datas[index_id][key] = averaged_value
       
            client_weight_datas[neighbor_indexid][key] = averaged_value
        
 

        return client_weight_datas

    def caculate_weight(self,index_id, neighbor_indexid,):
        all_length = 0
        rat = []
        path = self.p + '/' + self.data_name + '_all' 
        i=0
        pair = [index_id]+[neighbor_indexid]
        for client_ID in pair:
            filename = path  + '/localUser'+str(client_ID)+'.txt'
            rat.append(len(load_ml_100k(filename)))
            all_length += rat[i]
            i += 1
       
        for j in range(i):
            rat[j] = rat[j] / all_length
        
        return rat



    def federated_average(self, client_weight_datas,connection):
        """Average the parameters of the training model and return
        """

             
        while True:
            for index_id, neighbors_index in connection.items():
                for neighbor_indexid in neighbors_index:
                    if index_id < neighbor_indexid: 
                        after_client_weight_datas = self.pair_federated_average(index_id, neighbor_indexid, client_weight_datas)

            if self.has_converged(after_client_weight_datas, epsilon = 1e-5):
                break
        
        fed_state_dict = OrderedDict()
        w  = list(client_weight_datas[0].keys())
        for key in w:
            fed_state_dict[key] = client_weight_datas[0][key] 
       
        self.model.load_state_dict(fed_state_dict)

        return fed_state_dict 




    def run(self,path):
        """Complete training process,From the fifth round, neighbor users participated,
        After the training, the corresponding data and weights are saved
        """
        connect_path = './Process-Data/'
        with open('connection.txt', 'r') as file:
            content = file.read()

        connection = {}
        for line in content.split('\n'):
            if line:
                key, value = line.split(':')
                value = eval(value)
                connection[int(key)] = value

 

        userEmbdMan = []
        embedding_user = torch.rand((self.num_users , 64), requires_grad=True)
        nn.init.normal_(embedding_user, std=0.1)
        user_params = embedding_user
        for i in range(self.num_users):
            userEmbdMan += [user_params[i]] 

        #1.Neighbor：false    NeighId:only one 
        #2.Neighbor：True    NeighId:all neighbor
        Neighbor = False
        if(Neighbor == False):
            NeighId =  self.client_Id
        else:
            NeighId =  self.NeighId
        

        t1 = time()
        rmse,alltest_rmse,variance =evaluate_model(self.model,self.client,0,self.client_Id,userEmbdMan,Neighbor)
        print(rmse)
        
        print('[%.1f s]' % (time()-t1))
        
        # Train model federated

        learning_rate = self.lr
        evaluate_rmse = []
        evaluate_global_rmse = []
        allvariance = []
        allepochtest_rmse = []
        for epoch in range(self.epochs):
            t1 = time()
            client_weight_datas = self.distribute_task(connection, learning_rate, NeighId, userEmbdMan, Neighbor)
            avg_state_dict = self.federated_average(client_weight_datas,connection)
           
            rmse_all = evaluate_globalmodel(self.model,self.client,epoch+1, self.client_Id, userEmbdMan, Neighbor)
            evaluate_global_rmse.append(rmse_all)

            t2 = time()
            print('Iteration %d [%.1f s]' % (epoch,  t2-t1))

            t1 = time() 
            rmse,alltest_rmse,variance = evaluate_model(self.model,self.client,epoch+1, self.client_Id, userEmbdMan, Neighbor)
            print(rmse)
            print('[%.1f s]' % (time()-t1))

            # change Round R
            # if(epoch + 1 == 4): 
            #    Neighbor = True

            if(Neighbor == False):
                NeighId =  self.client_Id
            else:
                NeighId =  self.NeighId

            evaluate_rmse.append(rmse)
            allepochtest_rmse.append(alltest_rmse)
            allvariance.append(variance)

            print(evaluate_rmse)
        #Save training data 
        save_path  = path+'/'+'save_data'+ '/' + self.data_name
        #evaluate rmse
        f = open(save_path +'/'+ 'evaluate_rmse.txt','w')
        f.write(str(evaluate_rmse)+'\n')
        #Test values for the global model
        f = open(save_path +'/'+'evaluate_global_rmse.txt','w')
        f.write(str(evaluate_global_rmse)+'\n')
        #rmse value for all clients in each round
        f = open(save_path +'/'+'allepochtest.txt','w')
        for i in allepochtest_rmse:
            f.write(str(i)+'\n')
        #variance
        f = open(save_path +'/'+'allvariance.txt','w')
        f.write(str(allvariance)+'\n')

        
        save_model_path =path+'/'+'save_model'+ '/' + self.data_name+ '/'  +'victim_model.pt'
        torch.save(self.model.state_dict(), save_model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parse_args()
base_path = './Process-Data/data/'
ser = Server(base_path,  args.client_batch, args.local_epoch, args.epochs, args.lr, 2, 0.5, args.dataset)
ser.run(args.path)



