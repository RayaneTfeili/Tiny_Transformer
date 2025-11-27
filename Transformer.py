import torch
import torch.nn as nn
import math 
import torch.functional as F 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding(nn.Module):
    def __init__(self,vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size 
        self.emb_dim = emb_dim 
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim)
    def forward(self,input):
        return self.embedding(input)

class PositionalEncoding(nn.Module):
    def __init__(self,seq_len, emb_dim,batch_size=1 ):
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

    def forward(self, input):
        x= input * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        x = x + pe
        return x

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
    
class Attention(nn.Module):
    def __init__(self, d_model: int, d_head: int, dropout: float, masked : bool):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.masked_default = masked 
        self.query_layer = nn.Linear(d_model, d_head, bias=True)
        self.key_layer   = nn.Linear(d_model, d_head, bias=True)
        self.value_layer = nn.Linear(d_model, d_head, bias=True)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, masked :bool = None):
       
        batch_size, seq_len, model_dim  = x.size()
        assert model_dim == self.d_model , f"Input dimension {model_dim} doesn't match the model dimension {self.d_model}"

        query = self.query_layer(x)  
        key   = self.key_layer(x)     
        value = self.value_layer(x) 

        att_scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

       
        if masked is None:
            masked = self.masked_default
        if masked:
            
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
           
            att_scores = att_scores.masked_fill(causal_mask, float("-inf"))


        att_weights = F.softmax(att_scores, dim=-1)      # [B, S, S]
        att_weights = self.dropout_layer(att_weights)
        output = att_weights @ value
        return output


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float,masked: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.heads = nn.ModuleList([
            Attention(d_model=self.d_model, d_head=self.d_head, dropout=dropout, masked = masked)
            for _ in range(self.num_heads)
        ])

        self.output_projection = nn.Linear(self.num_heads * self.d_head, self.d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,masked : bool = None ):
        
        head_outputs = [h(x,masked = masked ) for h in self.heads]
        output = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(output)  
        output = self.dropout_layer(output)
        return output
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model : int, num_heads : int ,d_hidden : int, dropout : float, masked : bool = False):
        super().__init__()
        
        self.att_layer = MHA(d_model = d_model, num_heads = num_heads, dropout = dropout, masked = masked)
        self.feed_forward_layer = FFNN(d_model = d_model, d_hidden=  d_hidden,dropout = dropout)
        self.drop = nn.Dropout(dropout)
        self.LayerNorm_att = nn.LayerNorm(d_model)
        self.LayerNorm_ffnn = nn.LayerNorm(d_model)

    def forward(self,embed_input : torch.Tensor ,masked : bool  = None):
        att_sublayer = self.att_layer(self.LayerNorm_att(embed_input), masked = masked  )
        x = embed_input + self.dropout(att_sublayer)


        ffnn_sublayer = self.feed_forward_layer(self.LayerNorm_ffnn(x)) 
        x = x +  self.dropout(ffnn_sublayer)
        

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model : int, num_heads : int ,d_hidden : int, dropout : float, masked : bool = True):
        super().__init__()
        
        self.att_layer = MHA(d_model = d_model, num_heads = num_heads, dropout = dropout, masked = masked)
        self.feed_forward_layer = FFNN(d_model = d_model, d_hidden=  d_hidden,dropout = dropout)
        self.drop = nn.Dropout(dropout)
        self.LayerNorm_att = nn.LayerNorm(d_model)
        self.LayerNorm_att2 = nn.LayerNorm(d_model)
        self.LayerNorm_ffnn = nn.LayerNorm(d_model)

    def forward(self,embed_input : torch.Tensor ,masked : bool  = True):

        att_sublayer = self.att_layer(self.LayerNorm_att(embed_input), masked = masked  )
        x = embed_input + self.dropout(att_sublayer)

        att_sublayer2 = self.att_layer(self.LayerNorm_att2(embed_input), masked = False  )
        x = x + self.dropout(att_sublayer2)

        ffnn_sublayer = self.feed_forward_layer(self.LayerNorm_ffnn(x)) 
        x = x +  self.dropout(ffnn_sublayer)
        
        return x 


