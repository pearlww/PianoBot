import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

hidden_size = 70 #Default was 50

import config

class MusicLSTM(nn.Module):
    def __init__(self):
        super(MusicLSTM, self).__init__()
        
        # Recurrent layer
        # YOUR CODE HERE!
        self.lstm = nn.LSTM(input_size=int(config.vocab_size),
                         hidden_size=hidden_size,
                         num_layers=1,
                         bidirectional=True)
        
        # Output layer
        self.l_out = nn.Linear(in_features=2*hidden_size,
                            out_features=config.vocab_size,
                            bias=False)
        
        #self.embedding = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        
    def forward(self, x):
        #x = self.embedding(x.to(torch.long))
        # RNN returns output and last hidden state
        #print("x: " + str(x.shape))
        x=x.permute(1,0,2)
        #print("permuatre 1"+ str(x.shape))
        x, (h, c) = self.lstm(x)
        #x=x.permute(1,0,2)
        #print('x esteeee', x.shape)   
        # Flatten output for feed-forward layer
        #x = x.reshape(8, 128*2*self.lstm.hidden_size)
        #print('x este' , x.shape)
        # Output layer
        x = self.l_out(x)
        #print("iuhuuuuuuuuu")
        
        return x
    
    def generate(self, x):
        result = self.forward(x)
        result = result.softmax(-1)
        pdf = dist.OneHotCategorical(probs=result[:, -1])
        result = pdf.sample().argmax(-1).unsqueeze(-1)
        return result

net = MusicLSTM()
#print(net) 
