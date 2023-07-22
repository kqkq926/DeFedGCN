from torch.utils.data import Dataset
import pandas as pd
from os import path
from torch.utils.data import random_split
from torch import nn as nn

'''split all the data into train and test'''

p = './Process-Data/data'
path = p+ '/douban/douban.txt' 

class ML1K(Dataset):
    def __init__(self,rt):
        super(Dataset,self).__init__()
        self.uId = list(rt['userId'])
        self.iId = list(rt['itemId'])
        self.rt = list(rt['rating'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])

def loadRatings(path):
  
    df = pd.read_table(path,sep='\t',names=['userId','itemId','rating','timestamp'])
    return df

def splite_train_test(path,userID,itemId,rating):
    with open(path) as f:
        a = f.readlines()
    userID_all = []
    itemId_all = []
    rating_all = []
    for l in a :
        if len(l) > 0:
            l = l.strip('\n').split('\t')
            userID_all.append(int(l[0]))
            itemId_all.append(int(l[1]))
            rating_all.append(float(l[2]))

    i, j, k = 0, 0, 0
    while(i < len(usersID)):
        if(userID[i] == userID_all[j]):
            itemId[i].append(itemId_all[j])
            rating[i].append(rating_all[j])
            j += 1
            #i += 1
        elif(userID[i] > userID_all[j]):
            j += 1
        else:
            i += 1
            #j += 1
        if(i >= len(userID)):
            break
        if(j >= len(userID_all)):
            break


    return itemId, rating

#split train:test 8:2
rt = loadRatings(path)
ds = ML1K(rt)
print(ds.uId, ds.iId, ds.rt)

trainLen = int(0.8 * len(ds))
train,test = random_split(ds,[trainLen,len(ds)-trainLen])


train = sorted(train, key = lambda x:(x[0], x[1], x[2]))
test = sorted(test, key = lambda x:(x[0], x[1], x[2]))

path_train = p + '/douban/douban_train.txt'
path_test  = p + '/douban/douban_test.txt'


f = open(path_train,'w')
for i in range(len(train)):
    f.write(str(train[i][0])+'\t')
    f.write(str(train[i][1])+'\t')
    f.write(str(train[i][2]))
    f.write('\n')


f = open(path_test,'w')
for i in range(len(test)):
    f.write(str(test[i][0])+'\t')
    f.write(str(test[i][1])+'\t')
    f.write(str(test[i][2]))
    f.write('\n')

'''split train/test to ? users'''

alluser_path = p + '/douban/alluser.txt'


with open(alluser_path) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.split(',')

usersID = []
train_itemId, train_rating ,test_itemId ,test_rating = [], [], [], []
for i in range(len(l)):
    usersID.append(int(l[i]))
    train_itemId.append([])
    train_rating.append([])
    test_itemId.append([])
    test_rating.append([])


train_itemId, train_rating = splite_train_test(path_train, usersID, train_itemId, train_rating)
test_itemId ,test_rating = splite_train_test(path_test, usersID, test_itemId, test_rating)

train_len = 0
test_len = 0
for i in range(len(usersID)):
    train_len += len(train_itemId[i])
    test_len += len(test_itemId[i])
print('Number of training sets',train_len)
print('Number of test sets',test_len)

 #yahoo_music rating/20

for i in range(len(usersID)):
    headpath = p + '/douban/douban_all/localUser'+str(i+1)+'.txt'
    f = open(headpath,'w')
    for j in range(len(train_itemId[i])):
        f.write(str(usersID[i])+'\t')
        f.write(str(train_itemId[i][j])+'\t')
        #f.write(str(train_rating[i][j]/20))
        f.write(str(train_rating[i][j]))
        f.write('\n')


for i in range(len(usersID)):
    headpath = p + '/douban/douban_all/localUser'+str(i+1)+'_test.txt'
   
    f = open(headpath,'w')
    for j in range(len(test_itemId[i])):
            f.write(str(usersID[i])+'\t')
            f.write(str(test_itemId[i][j])+'\t')
            #f.write(str(train_rating[i][j]/20))
            f.write(str(test_rating[i][j]))
            f.write('\n')


    
 

 

  
