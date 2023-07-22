from collections import OrderedDict
from Client_local_training import *
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from numpy import dot
from numpy.linalg import norm

def evaluate_model(model,Client,epoch,NeighId,userEmbdMan,Neighbor):
    """Read training data for each user,Put it to the test in each user's training model
    """

    server_weights = OrderedDict()
    for k, v in model.state_dict().items():
        server_weights[k] = v
    alltest_rmse = []
    test_rmse = 0
    length = 0
    sum = 0
    for i in range(Client.real_num_users):
        test_data = Client.test_datas[i]
        testData = Data(test_data)
        if(testData):

            user_server_weights = reload_model_params(NeighId[i], userEmbdMan,server_weights)
            square_r, len_m = test_model(model,i,user_server_weights,Neighbor, testData)
            test_rmse += square_r
            length += len_m
            if(Client.data_name == 'yahoo_music'):
                every_test_rmse = np.sqrt(test_rmse / length) * 20
                alltest_rmse.append(every_test_rmse)
            else:
                every_test_rmse = np.sqrt(test_rmse / length)
                alltest_rmse.append(every_test_rmse)
        else:
            pass

    if(Client.data_name == 'yahoo_music'):
        test_rmse = np.sqrt(test_rmse / length) * 20
    else:
        test_rmse = np.sqrt(test_rmse / length)
    
    for i in range(len(alltest_rmse)):
        sum += np.square(alltest_rmse[i]-test_rmse)
    variance = sum /Client.real_num_users

    return test_rmse,alltest_rmse,variance


def reload_model_params( NeighId, userEmbdMan,server_weights):
    """Get the initial parameters of the model,When the neighbor user node is not added,
    Do not add the Embd of the neighbor user. Otherwise, the neighbor does
    """

    user_server_weights = OrderedDict()
    for k, v in server_weights.items():
        if(k == 'embedding_user.weight'):
            user_server_weights[k]  = v
            for i in range(len(NeighId)):
                user_server_weights[k][NeighId[i]] = userEmbdMan[NeighId[i]].clone()
        else:        
            user_server_weights[k] = v
        
    return user_server_weights

    
# test local model 
def test_model(server_model,clientId,server_weights,Neighbor,testData):
    pre_rating = []
    true_rating = []
    server_model.load_state_dict(server_weights)
    
    dl = DataLoader(testData , batch_size=len(testData),shuffle=True,pin_memory=True)
    for id , batch in enumerate(dl):
        prediction = server_model(batch[0].cuda(), batch[1].cuda(),train = False)
    pre_rating = prediction.cpu().detach().numpy().tolist()
    true_rating = batch[2].float().cpu().detach().numpy().tolist()
    len_m = len(pre_rating)
    square_r =  mean_squared_error(pre_rating ,true_rating) * len_m

    return square_r, len_m

    
def evaluate_globalmodel(model,Client,epoch,NeighId,userEmbdMan,Neighbor):
    """evaluate 'globalmodel'
    """

    server_weights = OrderedDict()
    for k, v in model.state_dict().items():
        server_weights[k] = v

    test_data = Client.all_test_datas 
    testData = Data(test_data)

    test_rmse,length = test_model(model,0,server_weights,Neighbor, testData)
    if(Client.data_name == 'yahoo_music'):
        test_rmse = np.sqrt(test_rmse / length) * 20
    else:
        test_rmse = np.sqrt(test_rmse / length)

    return test_rmse 

 # evaluation index
def metric(ori_adj, inference_adj, train_data,num_users):
    adj_list = []
    inference_adj_list = []
    for i in range(len(train_data)):
        a,b = int(train_data.iloc[i,0]),int(train_data.iloc[i,1])
        adj_list.append(ori_adj[a][num_users+b])
        inference_adj_list.append(inference_adj[a][num_users+b])

    adj_list=np.array(adj_list)
    inference_adj_list=np.array(inference_adj_list)

    output =  dot(adj_list,inference_adj_list)/(norm(adj_list)*norm(inference_adj_list))
    return output

