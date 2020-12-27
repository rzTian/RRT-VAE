#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn import Parameter
import torch.nn.functional as F
import pickle
import argparse
from torch.nn import init
import time

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10)
np.random.seed(10)

#-------------------hyperparameters----------------#

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--num_topics', type=int,   default=50) #number of topics
parser.add_argument('-e', '--num_epochs', type=int,   default=150) # number of training epochs
parser.add_argument('-b', '--batch_size', type=int,   default=100) # training batch size
parser.add_argument('-l', '--lr', type=float,   default=0.001) # learning rate
parser.add_argument('-a', '--prior_alpha', type=float,   default=1.0) # Dirichlet prior parameter
parser.add_argument('-d', '--delta', type=float,   default=1e10) # Delta parameter for RRT
parser.add_argument('-lam', '--LAMBDA', type=float,   default=0.01) # Lambda paramter for RRT

args = parser.parse_args()

#----------------------------------------------------#

prior_alpha = torch.Tensor(1, args.num_topics).fill_(args.prior_alpha).to(device)
SAVE_FILE = 'RRT.pt'
path_tr = 'data/20news_clean/train.txt.npy'
path_te = 'data/20news_clean/test.txt.npy'
path_vocab = 'data/20news_clean/vocab.pkl'

#--------------------RRT-VAE------------------------#

class RRT_VAE(nn.Module):

    def __init__(self, input_size, prior=prior_alpha):
        super(RRT_VAE, self).__init__()
        
        self.input_size = input_size
        self.prior_alpha = prior
        
        #Encoder
        self.encoder = nn.Sequential(
            
            nn.Linear(self.input_size, 500),   
            nn.ReLU(True),
            
            nn.Linear(500, 500),   
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            
            nn.Linear(500, args.num_topics),
            nn.BatchNorm1d(args.num_topics),

            )
        
        #Decoder
        self.decoder    = nn.Linear(args.num_topics, self.input_size)             
        self.decoder_bn = nn.BatchNorm1d(self.input_size)
        self.decoder.weight.data.uniform_(0, 1)       

    def RealSampler(self, parameter, multi):
        # Draw multiple samples
        if multi:
             m = torch.distributions.dirichlet.Dirichlet(parameter)
             data = m.sample((2000,))
             return 
         
        # Draw one sample
        else:
            m = torch.distributions.dirichlet.Dirichlet(parameter)
            data = m.sample()
            return data.to(device)
            
    # Sampling from Dirichlet distributions using RRT
    def RRT(self, parameter):
        # Round the Dirichlet parameter to its delta decimal place
        param_round = torch.floor(args.delta*parameter)/args.delta
        # Sampling from a "Rounded" Dirichlet distribution
        sample = self.RealSampler(param_round, multi=False)
        # Construct the target sample
        sample = sample + (parameter - param_round)*args.LAMBDA
        sample = sample/torch.sum(sample, dim=1, keepdim=True)
        
        return sample
        
    
    def forward(self, inputs, avg_loss=True):
        
        #Encoder
        alpha = self.encoder(inputs)
        alpha = torch.exp(alpha/4)
        alpha = F.hardtanh(alpha, min_val=0., max_val=30)
        #Sampling usnig RRT
        p = self.RRT(alpha)
        # Decoder
        recon = F.softmax(self.decoder_bn(self.decoder(p)), dim=1)  # Reconstruct a distribution over vocabularies

        return recon, self.loss(inputs=inputs, recon=recon, alpha=alpha, avg=avg_loss)
      

    def loss(self, inputs, recon, alpha, avg=True):
        # Negative log likelihood
        NL  = -(inputs * (recon+1e-10).log()).sum(1)
        # Dirichlet prior
        prior_alpha = self.prior_alpha.expand_as(alpha)
        # KLD between two Dirichlet distributions
        KLD = torch.mvlgamma(alpha.sum(1), p=1)-torch.mvlgamma(alpha, p=1).sum(1)-torch.mvlgamma(prior_alpha.sum(1), p=1)+torch.mvlgamma(prior_alpha, p=1).sum(1)+((alpha-prior_alpha)*(torch.digamma(alpha)-torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1) 
        # loss
        loss = (NL + KLD)
        # In the training mode, return averaged loss. In the testing mode, return individual loss
        if avg:
            return loss.mean(), KLD.mean()
        else:
            return loss, KLD
    

#--------------- train the model ------------------#
        
def train(): 
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        all_indices = torch.randperm(data_tr.size(0)).split(args.batch_size)
        loss_epoch = 0.0
        kld_epoch = 0.0
        model.train()                   # switch to training mode
        
        for batch_indices in all_indices:
            batch_indices = batch_indices.to(device)
            inputs = data_tr[batch_indices]
            recon, (loss, kld_loss) = model(inputs, avg_loss=True)
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.item()    # add loss to loss_epoch
            kld_epoch += kld_loss.item()
            
        print('====> Epoch: {} Average loss: {:.4f}, KL Div: {:.4f}'.format(
          epoch, loss_epoch / len(all_indices), kld_epoch / len(all_indices)))
             
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
              
        torch.save(model.state_dict(), SAVE_FILE)
        
   

#----------------compute perplexity-------------#

def print_perp():
    
    model.eval()                        # switch to testing mode
    inputs = data_te
    recon, (loss, KLD) = model(inputs, avg_loss=False)
    loss = loss.data
    counts = data_te.sum(1)
    avg = (loss / counts).mean()
    print('The approximated perplexity is: ', torch.exp(avg))


#--------------print topic words --------------#

def topic_words(beta, feature_names, n_top_words=10):
    print ('---------------Extracted Topic words------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        
        print('{}'.format(line))
    print (' ----------------------------------------------------')


#----------------------------------------------#  

if __name__=='__main__':
    
    global data_tr, data_te, vocab, model, opitmizer
    
    data_tr, data_te, vocab = load_data(path_tr, path_te, path_vocab)
    data_tr = torch.Tensor(data_tr).to(device)
    data_te = torch.Tensor(data_te).to(device)
    
    model = RRT_VAE(input_size=data_tr.size()[1]).to(device)
    model.apply(weights_init)
    
    print(f'The model has {count_parameters(model)[0]:,} trainable parameters')
    print(f'The total number of parameter is {count_parameters(model)[1]:,}') 
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    
    train_loss = train()
    print('Training completed! Saving model.')
    
    print_perp()
    emb = model.decoder.weight.data.cpu().numpy().T
    topic_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])
    
    
    
