from time import time
import torch
from torch.optim import SGD,Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import numpy as np 
from Dataset import Dataset
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

lossfn = MSELoss()

#load data 
class Data(Dataset):
    def __init__(self,rt):
        super(Dataset,self).__init__() 
        self.uId = list(rt['userId'])
        self.iId = list(rt['itemId'])
        self.rt = list(rt['rating'])
    #data length
    def __len__(self):
        return len(self.uId)
    #train data
    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])

class Client:
    # Initialize Client
    def __init__(self, batch_size = 64, epochs = 1,
                 data_name = 'ml_100k',path='./Process-Data/data/'):
        #local training round
        self.epochs = epochs
        self.batch_size = batch_size
        self.path = path
        self.data_name = data_name
        #get dataset
        t1 = time()
        dataset = Dataset(self.path+ self.data_name, self.data_name) 
        #user number item number
        self.num_users, self.num_items = dataset.get_train_data_shape()
        self.real_num_users = len(dataset.load_all_clientId())
        #load the client training dataset
        self.client_train_datas = dataset.load_client_train_date()
        # add neighbor train data 
        self.client_neigh_train_data = dataset.load_client_neigh_train_data()
        #test data
        self.test_datas = dataset.load_test_file()
        #Gets all the test sets
        self.all_test_datas = dataset.load_all_testdatas()
        print("Client Load data done [%.1f s]. #user=%d, #item=%d"
          % (time()-t1, self.num_users, self.num_items))


    #The Id of all users fetching neighbor users will be obtained  
    def get_Neignbor_userId(self):
        NeighUserId = []
        userId = []
        psi_path = self.path + self.data_name + '/' + self.data_name + '_psi_all' 
        for i in range(self.real_num_users):
            user_path = '/localUser'+str(i + 1)+'.txt'
            with open(psi_path + user_path) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.split('\t')
                        userId.append(int(l[0]))
            userId = list(set(userId))
            userId.sort()
            NeighUserId.append(userId)
            userId = []
        return NeighUserId


    #gets the user Id
    def construct_userId(self,all_clientId):
        client_Id = []
        for i in range (self.real_num_users):
            client_Id.append([all_clientId[i]])
        return client_Id

    #local training
    def train_epoch(self,server_model, indx, client_id, server_weights, learning_rate, userEmbdMan,Neighbor ):
        client_train_data = self.client_train_datas[indx]
        server_model.load_state_dict(server_weights)

        optim = SGD(server_model.parameters(), lr = learning_rate ,weight_decay=0.001)

        if(Neighbor):
            Neig_train_data = self.client_neigh_train_data[indx]
            server_model.construct_adj(Neig_train_data)  
        else:
            server_model.construct_adj(client_train_data)
       
        train_data = Data(client_train_data)
        if(train_data): 
            dl = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,pin_memory=True)
            for i in range(self.epochs):
                for id, batch in enumerate(dl):
                    optim.zero_grad()
                    prediction = server_model(batch[0].cuda(), batch[1].cuda(),train = True)
                    loss = lossfn(batch[2].float().cuda(),prediction)
                    loss.backward()
                    optim.step()
                    

        #Only the parameters of itemEmbd and the parameters of the model are uploaded
        weights = OrderedDict()
        for k, v in server_model.state_dict().items():
            if(k == 'embedding_user.weight'):  
                userEmbdMan[client_id] = v[client_id]
                weights[k] = v
            else:
                weights[k] = v
                sigma = 0.0005
                a=np.array(weights[k].cpu())
                noise = np.random.normal(0, sigma, a.shape)
                noise = torch.tensor(noise)
                noise = noise.to(torch.float32)
                weights[k] += noise.cuda()
  
        return weights


