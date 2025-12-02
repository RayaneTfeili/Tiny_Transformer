import torch
import torch.nn as nn
import math 
import torch.nn.functional as F 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding(nn.Module):
    def __init__(self,vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size 
        self.emb_dim = emb_dim 
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
    def forward(self,input):
        return self.embedding(input)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.seq_len = seq_len 
        self.emb_dim = emb_dim 
        pe = torch.zeros(self.seq_len, self.emb_dim)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float()* (-math.log(10000.0) / self.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, input):
        S = input.size(1)
        return input + self.pe[:, :S]

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
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
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
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),diagonal=1)
           
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
        output = self.drop(output)
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
        x = embed_input + self.drop(att_sublayer)


        ffnn_sublayer = self.feed_forward_layer(self.LayerNorm_ffnn(x)) 
        x = x +  self.drop(ffnn_sublayer)
        

        return x

class CrossAttention(nn.Module):
    def __init__(self,d_model : int, d_head : int, dropout : float):
        super().__init__()
        self.d_head = d_head 
        self.query = nn.Linear(d_model,d_head, bias = True )
        self.key = nn.Linear(d_model,d_head,bias = True)
        self.value = nn.Linear(d_model,d_head,bias = True)
        self.drop = nn.Dropout(dropout)

    
    def forward(self, x : torch.Tensor, context : torch.Tensor):
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)

        att_score = (Q @ K.transpose(-2,-1)) * (1.0/math.sqrt(self.d_head))
        att_weight = F.softmax(att_score,dim =-1)
        att_weight = self.drop(att_weight)
        out = att_weight @ V  
        return out 

class CrossMHA(nn.Module):
    def __init__(self,d_model : int , num_heads : int, dropout : float):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model//self.num_heads

        self.heads = nn.ModuleList([CrossAttention(self.d_model, self.d_head,dropout) for _ in range(num_heads)])
        self.output_projection = nn.Linear(self.num_heads*self.d_head,self.d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, context : torch.Tensor):
        outs = [h(x,context)for h in self.heads]
        y = torch.cat(outs,dim = -1)
        y = self.output_projection(y)
        y = self.drop(y)
        return y 
    

class DecoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_hidden, dropout):
        super().__init__()
        self.LayerNorm_att1 = nn.LayerNorm(d_model)
        #self.LayerNorm_att2 = nn.LayerNorm(d_model)  uncomment if you use the encoder part 
        self.LayerNorm_ffnn = nn.LayerNorm(d_model)
        self.att_layer =  MHA(d_model = d_model, num_heads = num_heads, dropout = dropout, masked = True)
        #self.crossatt_layer = CrossMHA(d_model,num_heads,dropout)  uncoment if you use the encoder part 
        self.ffnn = FFNN(d_model,d_hidden,dropout)
        self.drop = nn.Dropout(dropout)
    #add encoder_out : torch.Tensor if you want to use the encoder part  
    def forward(self,embed_input : torch.Tensor):
        x = embed_input + self.drop(self.att_layer(self.LayerNorm_att1(embed_input),masked = True))
        #x = x + self.drop(self.crossatt_layer(self.LayerNorm_att2(x),encoder_out)) uncoment if you use the encoder part
        x = x + self.drop(self.ffnn(self.LayerNorm_ffnn(x)))

        return x 


class TransformerEncoderDecoder(nn.Module):

    def __init__(self, num_layer,d_model, d_hidden,num_heads, drop=0.1,bias=True):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.d_hidden = d_hidden 
        self.num_heads = num_heads
        self.drop = drop
        self.bias = bias
        
        #Encoder stack 
        self.encoder_stack = nn.ModuleList([EncoderLayer(d_model = self.d_model, d_hidden = self.d_hidden,num_head = self.num_heads, drop = self.drop,bias = self.bias) for _ in range(self.num_layer)])

        #Decoder stack
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model = self.d_model,d_hidden = self.d_hidden,num_head = self.num_heads, drop = self.drop,bias = self.bias) for _ in range(self.num_layer)])

    
    def forward(self, embed_encoder_input, embed_decoder_input):
        # Process through all encoder layers first
        encoder_output = embed_encoder_input
        for encoder in self.encoder_stack:
            encoder_output = encoder(encoder_output)
        
        # Use final encoder output for all decoder layers
        decoder_output = embed_decoder_input
        for decoder in self.decoder_stack:
            decoder_output = decoder(decoder_output, encoder_output)
        
        return decoder_output
####
#Since I have only a GTX 1650, I'll not use TransformerEncoderDecoder Class 
#but rather use the decoder class 
#If you want to use the full architecture use 
#TransformerEncoderDecoder class in self.transform_blocks 
class TransformerModel(nn.Module):
    def __init__(self,d_model, num_layer, d_hidden,num_heads,vocab_size,context_length, drop=0.1):
        super().__init__()
        self.num_layer = num_layer 
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_hidden = d_hidden 
        self.vocab_size = vocab_size
        self.context_length = context_length 
        self.drop = drop

        self.token_embedding = Embedding(self.vocab_size ,self.d_model)
        self.positional_encoding = PositionalEncoding(context_length, self.d_model )
        self.blocks = nn.ModuleList([DecoderLayer(self.d_model, self.num_heads, self.d_hidden, self.drop) for _ in range(num_layer)])

        self.LN = nn.LayerNorm(self.d_model)
        self.linear_classifier_layer = nn.Linear(in_features=self.d_model, out_features=self.vocab_size)
        self.linear_classifier_layer.weight = self.token_embedding.embedding.weight



    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_length
        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.LN(x)
        logits = self.linear_classifier_layer(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1))

        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length:]
            logits, loss = self(idx_crop)
            probs = F.softmax(input=logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    



