import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.l_out = nn.Linear(in_features=hidden_size,
                            out_features=config.vocab_size,
                            bias=False)
        
        #self.embedding = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        
    def forward(self, x):
        #x = self.embedding(x.to(torch.long))
        # RNN returns output and last hidden state
        print("x: " + str(x))
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x

net = MusicLSTM()
print(net) 
