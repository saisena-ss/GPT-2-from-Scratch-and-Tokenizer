from dataclass import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTconfig:
    block_size:int = 256
    vocab_size:int = 65
    n_layer:int = 6
    n_head:int = 6
    n_embed:int = 384
 
#multi head attention
class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)
        
        
        
#let's define the attention block
class Block(nn.Module):
    """
    Steps:
    1. Layer norm
    2. Attention
    3. Residual connection
    4. layer norm
    5. feed forward
    6. Residual connection
    """
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attention = CasualSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x
            

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(4*config.n_embed,config.n_embed)
        
    def forward(self,x):
        return self.fc2(self.gelu(self.fc1(x)))
    
        
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
                                wte = nn.Embedding(config.vocab_size,config.n_embed),
                                wpe = nn.Embedding(config.block_size,config.n_embed),
                                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                ln_f = nn.LayerNorm(config.n_embed) 
                                ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        