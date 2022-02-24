import torch
import math
from torch.nn import init
import json
import torch.nn as nn
import torch.nn.functional as F





def get_activation_function(activation: str='PReLU') -> nn.Module:
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')




class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self,hidden_dim , ffn_hidden_dim, activation_fn="GELU", dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = get_activation_function(activation_fn)

    def forward(self, x):
        residual=x
        x = self.dropout(self.fc2(self.act_dropout(self.ffn_act_func(self.fc1(x)))))
        x+=residual
        x = self.ffn_layer_norm(x)
        return x



class MultiheadAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 dropout=0.1,
                 attn_dropout=0.1,
                 temperature = 1):
        super().__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads  # number of heads
        self.temperature =temperature
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim,eps=1e-6)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.a_proj.weight)

    def forward(self, x, mask=None, attn_bias=None):
        residual = x
        batch_size = x.size(0)

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        #ScaledDotProductAttention
        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        scores = torch.matmul(query/self.temperature, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if attn_bias is not None:
            scores = scores+attn_bias

        if mask is not None:
            if scores.shape==mask.shape:#different heads have different mask
                scores = scores * mask
                scores = scores.masked_fill(scores == 0, -1e12)
            else:
                scores = scores.masked_fill(mask == 0, -1e12)

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        #ScaledDotProductAttention

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.dropout(self.a_proj(out))
        out += residual
        out = self.layer_norm(out)

        return out, attn


class Transformer_Layer(nn.Module):
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 ffn_hidden_dim,
                 dropout=0.1,
                 attn_dropout=0.1,
                 temperature = 1,
                 activation_fn='GELU'):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.attention = MultiheadAttention(num_heads,
                                            hidden_dim, 
                                            dropout, 
                                            attn_dropout,
                                            temperature)
        self.ffn_layer = PositionwiseFeedForward(hidden_dim,ffn_hidden_dim,activation_fn=activation_fn)


    def forward(self, x, attn_mask, attn_bias=None):
        x, attn = self.attention(x, mask=attn_mask, attn_bias=attn_bias)
        x = self.ffn_layer(x)

        return x, attn


