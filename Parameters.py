import argparse
import numpy as np
import scipy.sparse as sp
import torch


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

def parse_args():
    """Define parameter
    """
    parser = argparse.ArgumentParser(description="Go FedLightGCN")
    #client_batch：client batch size
    parser.add_argument('--client_batch', type=int,default=32,
                        help="the batch size for loss training procedure")
    #recdim：lightGCN embedding size 64
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    #layer：lightGCN layer number 3
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    #lr：0.01               
    parser.add_argument('--lr', type=float,default=0.01, 
                        help="the learning rate")
    #dataset：ml_100k, douban, flixster, yahoo_music
    parser.add_argument('--dataset', type=str,default='yahoo_music', 
                        help="available datasets: [ml_100k, douban, flixster, yahoo_music]")
    #save path
    parser.add_argument('--path', type=str,default="./Save",
                        help="path to save weights")
    #epochs: ml_100k 350,douban 450, flixster 250, yahoo_music 300
    parser.add_argument('--epochs', type=int,default = 1)
    parser.add_argument('--attack_epochs', type=int,default = 10)
    parser.add_argument('--local_epoch', type=int,default = 1)
    parser.add_argument('--nlabel', type=float, default=0.8)
    parser.add_argument('--density', type=float, default=0.05, help='Edge density estimation')
    #seed：random seed  
    parser.add_argument('--seed', type=int, default=542, help='random seed')

    return parser.parse_args()

