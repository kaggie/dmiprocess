import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PEwithPeak(nn.Module): #positional encoding with learned peaks
    def __init__(self, embed_dim=256, max_len=256, num_peaks=4, **kwargs):
        super(PEwithPeak,self).__init__()
        self.dim = embed_dim
        #standard positional encoding
        pe = torch.zeros(max_len,embed_dim)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(1000.0) / embed_dim))   
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        #learned embeddings for peaks
        self.peak_embedding = nn.Embedding(max_len,embed_dim)

    def forward(self,x,peak_positions): #x (sequence_length,batch_size,embed_dim)
        x = x + self.pe[:x.size(0),:]
        peak_embed = torch.zeros_like(x)
        seq_len, batch_size, _ = x.shape
        
        for i in range(batch_size):
            valid_peaks = (peak_positions[i] >= 0) & (peak_positions[i] < seq_len)
            valid_positions = peak_positions[i][valid_peaks]

            if valid_positions.numel() > 0:  # If there are valid peak positions
                embeddings = self.peak_embedding(valid_positions)  # [num_valid_peaks, embedding_dim]
                peak_embed[valid_positions,i,:] = embeddings
        x = x + peak_embed
        return x


class MHA(nn.Module): #MultiheadAttention
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.head_dim = embed_dim // num_heads
        
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.attention_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, embed_dim = query.size()
        
        # Linear projections
        query = self.query_linear(query).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = self.key_linear(key).view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = self.value_linear(value).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: (batch_size, num_heads, seq_length, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, value)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        
        # Final linear projection
        output = self.out_linear(attention_output)
        
        return output, attention_weights

class TELayer(nn.Module): #TransformerEncoderLayer
    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super(TELayer, self).__init__()
        self.self_attn = MHA(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention sub-layer
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feed-Forward sub-layer
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class TransformerDenoiser(nn.Module):
    def __init__(self, num_layers, input_dim=1, conv_dim=64, embed_dim=256, num_heads=8, dim_feedforward=1024, max_len=500, dropout=0.1):
        super(TransformerDenoiser, self).__init__()
        self.embedding = nn.Linear(conv_dim, embed_dim)
        self.pos_encoder = PEwithPeak(embed_dim, max_len)
        #encoder_layer = TELayer(embed_dim=embed_dim, num_heads=num_heads, 
         #                                       dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,dim_feedforward=dim_feedforward,dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) #Maybe need to implement a nn.TransformerEncoder myself
        self.output_layer = nn.Linear(embed_dim, conv_dim)
        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_dim, input_dim, kernel_size=1)#, padding=1)

    def forward(self, x, peak_positions): # x [sequence_len, batch, input_dim]
        #x = x.permute(1, 2, 0)
        #x [batch, input_dim, sequence_len]
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x = self.embedding(x) # x [sequence_len, batch, embed_dim]
        #x = x.permute(2, 0, 1)
        x = self.pos_encoder(x, peak_positions) # x [sequence_len, batch, embed_dim]
        x = self.transformer_encoder(x) # x [sequence_len, batch, embed_dim]
        x = self.output_layer(x)
        x = x.permute(1, 2, 0)
        x = self.conv2(x)
        #x = x.permute(2, 0, 1)
        return x

