import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def load_ml_100k(path):
    #read train dataset
    df = pd.read_table(path ,sep='\t',names=['userId','itemId','rating','timestamp'])
    return df

class Dataset(object):

    def __init__(self, path, data_name):
        #get train dataset path
        self.path = path
        self.data_name = data_name
        self.num_users, self.num_items = self.get_train_data_shape()
        self.real_num_users = len(self.load_all_clientId())

    def get_train_data_shape(self):
        num_users = 3000   #douban 3000   yahoo_music: 3000   ml-100k:943   flixster：3000
        num_items = 3000    #douban 3000   yahoo_music: 3000   ml-100k:1682  flixster：3000
        return num_users, num_items

   
    def load_test_file(self):
        p = self.path + '/' + self.data_name + '_all' 
        client_test_data = []
        for i in range(self.real_num_users):
            filename = p + '/localUser'+str(i+1)+'_test.txt'
            client_test_data += [load_ml_100k(filename)]
           
        return client_test_data
    
    def load_all_testdatas(self):
        p = self.path + '/' + self.data_name + '_test.txt'
        all_testdatas = []
        all_testdatas = load_ml_100k(p)
        return all_testdatas

    def load_all_traindatas(self):
        p = self.path + '/' + self.data_name + '_train.txt'
        all_testdatas = []
        all_testdatas = load_ml_100k(p)
        return all_testdatas
    
 
    def load_client_train_date(self):
    
        p = self.path + '/' + self.data_name + '_all' 
        client_datas = []
        for i in range(self.real_num_users):
            filename = p  + '/localUser'+str(i+1)+'.txt'
            client_datas += [load_ml_100k(filename)]
           
        return client_datas

    def load_client_neigh_train_data(self):
        p = self.path + '/' + self.data_name + '_psi_all' 
        client_Neg_datas=[] 
        for i in range(self.real_num_users):
            filename = p  + '/localUser'+str(i+1)+'.txt'
            client_Neg_datas += [load_ml_100k(filename)]
        
        return client_Neg_datas 

      
    def load_all_clientId(self):
        all_clientId = []
        p = self.path + '/alluser.txt'
        with open(p) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.split(',')
        for i in range(len(l)):
            all_clientId.append(int(l[i]))
        return all_clientId


 