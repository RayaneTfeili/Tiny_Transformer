import torch
import torch.nn as nn
import math 
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
    def __init__(self, d_model: int, d_head: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head

        self.query_layer = nn.Linear(d_model, d_head, bias=True)
        self.key_layer   = nn.Linear(d_model, d_head, bias=True)
        self.value_layer = nn.Linear(d_model, d_head, bias=True)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, masked = False):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_head]
        """
        batch_size, seq_len, model_dim  = x.size()
        assert model_dim == self.d_model , f"Input dimension {model_dim} doesn't match the model dimension {self.d_model}"

        query = self.query_layer(x)   # [B, S, d_head]
        key   = self.key_layer(x)     # [B, S, d_head]
        value = self.value_layer(x)   # [B, S, d_head]

        # [B, S, d_head] @ [B, d_head, S] -> [B, S, S]
        att_scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

       

        if masked:
            # matrice [S, S] avec 1 au-dessus de la diagonale
            # True la ou on DOIT MASQUER (futur)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            # on remplace ces positions par -inf avant softmax
            att_scores = att_scores.masked_fill(causal_mask, float("-inf"))


        att_weights = F.softmax(att_scores, dim=-1)      # [B, S, S]
        att_weights = self.dropout_layer(att_weights)
        output = att_weights @ value
        return output


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.heads = nn.ModuleList([
            Attention(d_model=self.d_model, d_head=self.d_head, dropout=dropout)
            for _ in range(self.num_heads)
        ])

        self.output_projection = nn.Linear(self.num_heads * self.d_head, self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,masked = False) -> torch.Tensor:
        """
        x: [B, S, d_model]
        return: [B, S, d_model]
        """
        # liste de [B, S, d_head] -> concat [B, S, num_heads*d_head]
        head_outputs = [h(x,masked) for h in self.heads]
        output = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(output)   # [B, S, d_model]
        output = self.dropout_layer(output)
        return output
    