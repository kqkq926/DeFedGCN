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
from attack import PGDAttack
from scipy.sparse import csr_matrix
import pandas as pd
from numpy import dot
from numpy.linalg import norm
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
        pair_client_weight_datas = []
        client_ids = []
        all_ids = []
        for k, v in result.items():
                all_ids.append(k)
             
        self.epoch_client = all_ids
        server_weights = OrderedDict()
        for k, v in self.model.state_dict().items():
            server_weights[k] = v
        pair = []
      
        for client_id, pair_ids  in connection.items():
            pair.append(client_id)
            for i in range(len(pair_ids)):
                pair.append(pair_ids[i])
            for train_id in pair:
                index_Id = self.all_clientId.index(train_id)
                user_server_weights = reload_model_params(NeighId[index_Id], userEmbdMan,server_weights )
                weights = self.client.train_epoch(self.model, index_Id ,train_id ,user_server_weights, learning_rate,userEmbdMan,Neighbor)
                pair_client_weight_datas.append(weights)
                rat = pair_caculate_weight(pair)
                client_weight_datas = pair_federated_average(pair_client_weight_datas)
            pair.clear()

       
        client_weight_datas.append(client_weight_datas)

        return client_weight_datas

    def caculate_weight(self,pair):
        all_length = 0
        rat = []
        path = self.p + '/' + self.data_name + '_all' 
        i=0
        for client_ID in pair:
            filename = path  + '/localUser'+str(self.all_clientId.index(client_ID)+1)+'.txt'
            rat.append(len(load_ml_100k(filename)))
            all_length += rat[i]
            i += 1
       
        for j in range(i):
            rat[j] = rat[j] / all_length
        
        return rat

    def caculate_weight(self):
        all_length = 0
        rat = []
        path = self.p + '/' + self.data_name + '_all' 
        i=0
        for client_ID in self.epoch_client:
            filename = path  + '/localUser'+str(self.all_clientId.index(client_ID)+1)+'.txt'
            rat.append(len(load_ml_100k(filename)))
            all_length += rat[i]
            i += 1
       
        for j in range(i):
            rat[j] = rat[j] / all_length
        
        return rat

    def pair_federated_average(self, client_weight_datas):
        """Average the parameters of the training model and return
        """

        rat = self.caculate_weight()
        client_num = len(client_weight_datas)
        assert client_num != 0
        w = list(client_weight_datas[0].keys())
        fed_state_dict = OrderedDict()
        for key in w:
            key_sum = 0
            for i in range(client_num):
                key_sum = key_sum + client_weight_datas[i][key] * rat[i]
            fed_state_dict[key] = key_sum 

        return fed_state_dict 


    def federated_average(self, client_weight_datas):
        """Average the parameters of the training model and return
        """

        rat = self.caculate_weight()
        client_num = len(client_weight_datas)
        assert client_num != 0
        w = list(client_weight_datas[0].keys())
        fed_state_dict = OrderedDict()
        for key in w:
            key_sum = 0
            for i in range(client_num):
                key_sum = key_sum + client_weight_datas[i][key] * rat[i]
            fed_state_dict[key] = key_sum 

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
            avg_state_dict = self.federated_average(client_weight_datas)
           
            rmse_all = evaluate_globalmodel(self.model,self.client,epoch+1, self.client_Id, userEmbdMan, Neighbor)
            evaluate_global_rmse.append(rmse_all)

            t2 = time()
            print('Iteration %d [%.1f s]' % (epoch,  t2-t1))

            t1 = time() 
            rmse,alltest_rmse,variance = evaluate_model(self.model,self.client,epoch+1, self.client_Id, userEmbdMan, Neighbor)
            print(rmse)
            print('[%.1f s]' % (time()-t1))


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



