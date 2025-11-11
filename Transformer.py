import torch
import torch.nn as nn
import math 
print("GPU available", torch.cuda.is_available())

class Embedding(nn.Module):
    def __init__(self,vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size 
        self.emb_dim = emb_dim 
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim)
    def forward(self,input):
        return self.embedding(input)

class PositionalEncoding(nn.Module):
    def __init__(self,seq_len, emb_dim,batch_size = 1):
        super().__init__()
        self.seq_len = seq_len 
        self.emb_dim = emb_dim 
        self.batch_size = batch_size
        pos_enc = torch.zero(self.seq_len,self.emb_dim)
        for pos in range(self.seq_len):
            for i in range(0,self.emb_dim):
                pos_enc[pos,2*i] = math.sin(pos/10000**(2*i/self.emb_dim))
                pos_enc[pos,2*i+1] = math.cos(pos/10000**(2*i/self.emb_dim))
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pe", pos_enc)

    def forward(self,input):
        pass 

class FFNN(nn.Module):
    def __init__(self,d_model,d_hidden,dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = dropout

        self.fc1 = nn.Linear(self.d_model,self.d_hidden, bias = True)
        self.fc2 = nn.Linear(self.d_hidden,self.d_model, bias = True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout) 

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self,input):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(input))))) 
    

