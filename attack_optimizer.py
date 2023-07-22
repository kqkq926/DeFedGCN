import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import scipy.sparse as sp
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from Client_local_training import Data
from torch.nn.modules.module import Module



lossfn = MSELoss()

class PGDAttack(Module):
    
    def __init__(self, model, embedding, nnodes,usernodes,  itemnodes,  attack_structure=True, attack_features=False,device='gpu',batch_size=512):
        super(PGDAttack, self).__init__()

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.surrogate = model
        self.embedding = embedding
        self.nnodes = nnodes
        self.usernodes =usernodes
        self.itemnodes =itemnodes
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        self.batch_size =  batch_size
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.complementary = None
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(self.nnodes * (self.nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'



    def attack(self, train_data, ori_adj,   num_edges,lr,
               epochs, sample=False, **kwargs):
 
        victim_model = self.surrogate
        sparse = sp.issparse(ori_adj)
        ori_adj= self.to_tensor(ori_adj, device=self.device)

        train_data = Data(train_data)
        loss_list = []
  
        batch_size = len(train_data)
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = self.normalize_adj_tensor(modified_adj)
            dl = DataLoader(train_data,batch_size=batch_size ,shuffle=True,pin_memory=True)
            for id, batch in enumerate(dl):
                prediction = victim_model(batch[0].cuda(), batch[1].cuda(), adj_norm,train = True)
                loss = lossfn(batch[2].float().cuda(),prediction)

            loss =loss + torch.norm(self.adj_changes,p=2) * 0.001
            loss_list.append(loss.item())
            adj_grad = -torch.autograd.grad(loss, self.adj_changes)[0]

            lr = 0.2
            self.adj_changes.data.add_(lr * adj_grad)

            # if t > 200:
            #     self.adj_changes.data = self.SVD()
            self.projection(num_edges)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

        dl = DataLoader(train_data,batch_size=batch_size,shuffle=True,pin_memory=True)
        for id, batch in enumerate(dl):
            em = self.embedding(batch[0].cuda(), batch[1].cuda(),adj_norm,train = True)

        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        np.savetxt('loss.txt', loss_list)
      
        return prediction.detach()

        

    
    def dot_product_decode(self, Z):
        Z = F.normalize(Z, p=2, dim=1)
        A_pred = torch.relu(torch.matmul(Z, Z.t()))
        # A_pred = torch.matmul(Z, Z.t())
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        return A_pred[tril_indices[0], tril_indices[1]]

    
    def get_modified_adj(self, ori_adj):
    
        if self.complementary is None:
            self.complementary = torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device)
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def SVD(self):
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.detach()
        m = m + m.t()
        U, S, V = np.linalg.svd(m.cpu().numpy())
        U, S, V = torch.FloatTensor(U).to(self.device), torch.FloatTensor(S).to(self.device), torch.FloatTensor(V).to(
            self.device)
        alpha = 0.02
        tmp = torch.zeros_like(S).to(self.device)
        diag_S = torch.diag(torch.where(S > alpha, S, tmp))
        adj = torch.matmul(torch.matmul(U, diag_S), V)
        return adj[tril_indices[0], tril_indices[1]]

    def projection(self, num_edges):
        if torch.clamp(self.adj_changes, 0, 1).sum() > num_edges:
            # print('high')
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, num_edges, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))
    
    def bisection(self, a, b, num_edges, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - num_edges
        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def to_tensor(self,adj,device = 'gpu'):
        """Convert adj from array or sparse matrix to torch Tensor.

        Parameters
        ----------
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        device : str
            'cpu' or 'cuda'
        """
        if sp.issparse(adj):
            adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = torch.FloatTensor(adj)

        return adj.to(device)
    
    def is_sparse_tensor(self,tensor):
        """Check if a tensor is sparse tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            given tensor

        Returns
        -------
        bool
            whether a tensor is sparse tensor
        """
        # if hasattr(tensor, 'nnz'):
        if tensor.layout == torch.sparse_coo:
            return True
        else:
            return False

    def to_scipy(self,tensor):
        """Convert a dense/sparse tensor to scipy matrix"""
        if self.is_sparse_tensor(tensor):
            values = tensor._values()
            indices = tensor._indices()
            return sp.csr_matrix((values.cpu().detach().numpy(), indices.cpu().detach().numpy()), shape=tensor.shape)
        else:
            indices = tensor.nonzero().t()
            values = tensor[indices[0], indices[1]]
            return sp.csr_matrix((values.cpu().detach().numpy(), indices.cpu().detach().numpy()), shape=tensor.shape)

    def normalize_adj(self,mx):
        """Normalize sparse adjacency matrix,
        A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        Row-normalize sparse matrix

        Parameters
        ----------
        mx : scipy.sparse.csr_matrix
            matrix to be normalized

        Returns
        -------
        scipy.sprase.lil_matrix
            normalized matrix
        """

        if type(mx) is not sp.lil.lil_matrix:
            mx = mx.tolil()
        if mx[0, 0] == 0 :
            mx = mx + sp.eye(mx.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)
        return mx

    def normalize_adj_tensor(self,adj, sparse=False):
        """Normalize adjacency tensor matrix.
        """
        device = torch.device("cuda" if adj.is_cuda else "cpu")
        if sparse:
            # TODO if this is too slow, uncomment the following code,
            # but you need to install torch_scatter
            # return normalize_sparse_tensor(adj)
            adj = self.to_scipy(adj)
            mx = self.normalize_adj(adj)
            return self.sparse_mx_to_torch_sparse_tensor(mx).to(device)
        else:

            mx = adj + torch.eye(adj.shape[0]).to(device)
            rowsum = mx.sum(1)
            r_inv = rowsum.pow(-1/2).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)
            mx = r_mat_inv @ mx
            mx = mx @ r_mat_inv

        return mx.to(device)
    



        

  

