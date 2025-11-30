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

        att_score = (Q @ K.transpose(-2,1)) * (1.0/math.sqrt(self.d_head))
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
        self.LayerNorm_att2 = nn.LayerNorm(d_model)
        self.LayerNorm_ffnn = nn.LayerNorm(d_model)
        self.att_layer =  MHA(d_model = d_model, num_heads = num_heads, dropout = dropout, masked = True)
        self.crossatt_layer = CrossMHA(d_model,num_heads,dropout)
        self.ffnn = FFNN(d_model,d_hidden,dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self,embed_input : torch.Tensor, encoder_out : torch.Tensor):
        x = embed_input + self.drop(self.att_layer(self.LayerNorm_att1(embed_input),masked = True))
        x = x + self.drop(self.crossatt_layer(self.LayerNorm_att2(x),encoder_out))
        x = x + self.drop(self.ffnn(self.LayerNorm_ffnn(x)))

        return x 


class TransformerEncoderDecoder(nn.Module):

    def __init__(self, num_layer,d_model, d_hidden,num_heads, drop=0.1,bias=True):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.d_hidden = d_hidden 
        self.num_head = num_heads
        self.drop = drop
        self.bias = bias
        
        # Encoder stack
        self.encoder_stack = nn.ModuleList([ EncoderLayer(
                                        d_model = self.d_model, 
                                        d_hidden = self.d_hidden,
                                        num_head = self.num_heads, 
                                        drop = self.drop,
                                        bias = self.bias) for _ in range(self.num_layer)])

        # Decoder stack
        self.decoder_stack = nn.ModuleList([ DecoderLayer(
                                        d_model = self.d_model, 
                                        d_hidden = self.d_hidden,
                                        num_head = self.num_heads, 
                                        drop = self.drop,
                                        bias = self.bias) for _ in range(self.num_layer)])

    
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
class TransformerModel(nn.Module):
    def __init__(self,d_model, num_layer,num_blocks, d_hidden,num_heads,max_token_value,context_length,vocab_size, drop=0.1,bias=True):
        super().__init__()
        self.num_layer = num_layer 
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.num_blocks = num_blocks
        self.d_hidden = d_hidden 
        self.max_token_value = max_token_value 
        self.vocab_size = vocab_size
        self.context_lenght = context_length 


        self.drop = drop

        self.token_embedding = Embedding(num_embeddings = self.max_token_value + 1 , embedding_dim = self.d_model)

        self.transformer_blocks = nn.Sequential([TransformerEncoderDecoder(num_layer = self.num_layer,d_model=self.d_model, d_hidden=self.d_hidden,num_heads= self.num_heads, drop=0.1,bias=True) for _ in range(self.num_blocks)])


        self.linear_classifier_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)


    def forward(self, idx, targets=None):
        B, T = idx.shape
      
        x = self.token_embedding + PositionalEncoding(self.context_lenght, embedding_dim = self.d_model )
        x = self.transformer_blocks(x)
        logits = self.linear_classifier_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx