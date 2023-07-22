import torch
from torch import nn   
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
import warnings
warnings.filterwarnings("ignore")

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

class LGCN(nn.Module):
    """model part,there will be minor changes to the normal training and attack sections
    """
    def __init__(self,userNum,itemNum,embedSize=64,layers=[64,64,64],useCuda=True):
        super(LGCN,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        #user embedding  userNum * 64
        #item embedding  itemNum * 64
        self.embedding_user = nn.Embedding(userNum,embedSize)
        self.embedding_item = nn.Embedding(itemNum,embedSize)
        #initialize
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        #Construct a Laplace matrix
        self.f = nn.ReLU()
        #three layers of fully connected layers
        self.transForm1 = nn.Linear(in_features=(layers[-1]) * 2,out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32, out_features=1)

    def getEmbd(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()
        
        userEmbd = self.embedding_user(uidx)
        itemEmbd = self.embedding_item(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def get_train_test_Embd(self,finalEmbd, Idx):
        Embd = []
        for i in range(len(Idx)):
            Embd.append(finalEmbd[Idx[i].type(torch.long)])
            # Embd.append(finalEmbd[Idx[i]])
        Embd = torch.stack(Embd, 0)
        return Embd

   
    def forward(self,userIdx,itemIdx,train = True): 
        if(train):
            num_layer = 3
            Graph = self.Graph
     
        else :
            num_layer = 0

        # num_layer = 3
        itemIdx = itemIdx + self.userNum
        userIdx= list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)
       
        #Convolution part
        all_embs = self.getEmbd() 
    
        embs = [all_embs]
        if(train):
            for k in range(num_layer):
                all_embs = torch.sparse.mm(Graph , all_embs)
                all_embs = all_embs * (1/(k+2))
                embs.append(all_embs)
        embs = torch.stack(embs, dim=1)
        finalEmbd = torch.sum(embs, dim=1)
       
        '''
        Improved part： 
        Problem：Because the matrix is too large, there are issues where multiplication can go beyond the boundary
        therefore, the method of segmental multiplication is used
        '''
        '''
        for k in range(num_layer):
            if self.A_split:
                temp_emb = []
                for f in range(len(Graph)):
                    mul_emb = torch.sparse.mm(Graph[f], all_embs) * (1/(k+2))
                    temp_emb.append(mul_emb)
                side_emb = torch.cat(temp_emb, dim=0)
                all_embs = side_emb
            else:
                all_embs = torch.sparse.mm(Graph , all_embs) * (1/(k+2))
            embs.append(all_embs)
        embs = torch.stack(embs, dim=1)
        finalEmbd = torch.sum(embs, dim=1)
        '''
        #Obtain the Embd of the corresponding user
        userEmbd = self.get_train_test_Embd(finalEmbd, userIdx) #userEmbd = finalEmbd[userIdx]
        itemEmbd = self.get_train_test_Embd(finalEmbd, itemIdx) #itemEmbd = finalEmbd[itemIdx]
        #Integrate the convolved vectors
        embd = torch.cat([userEmbd ,itemEmbd ],dim=1)
 
        #three layers of fully connected layers
        embd = self.f(self.transForm1(embd)) # 64 - 32
        embd = self.transForm2(embd) # 32 -16  
        embd = self.transForm3(embd) # 16 - 1
        #prediction
        prediction = embd.flatten()

    
        return prediction
    
        #Construct the adjacency matrix
    def construct_adj(self,dataset):
        self.Graph = self.getSparseGraph(dataset)


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, dataset):
        adj_mat = sp.dok_matrix((self.userNum + self.itemNum, self.userNum + self.itemNum), dtype=np.float32)
        adj_mat = adj_mat.tolil()
       
        UserItemNet = csr_matrix((np.ones(len(dataset['itemId'])), (dataset['userId'],dataset['itemId'])),
                                      shape=(self.userNum, self.itemNum))
        
        R = UserItemNet.tolil()
        adj_mat[:self.userNum, self.userNum:] = R
        adj_mat[self.userNum:, :self.userNum] = R.T
        #adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0. #0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(device)
       
        return Graph



