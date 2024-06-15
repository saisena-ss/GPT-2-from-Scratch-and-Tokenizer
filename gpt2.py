from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time

@dataclass
class GPTconfig:
    block_size:int = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embed:int = 768
 
#multi head attention
class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        #actually it is mask
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        
        
    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embed,dim=2)
        
        #split into multiple heads
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # -> B, nheads, Time, head size
        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # -> B, nheads, Time, head size 
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # -> B, nheads, Time, head size
        
        #attention
        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        
        # #mask and softmax
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        
        # y = att @ v #(B,nheads, T, T) x (B, nheads, T, headsize) -> B, nheads, T, headsize
        
        #flash attention
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        #last projection
        y = self.c_proj(y)
        
        return y
        
        
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
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x
            

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
        
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
        
        #weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        #weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        std = 0.02
        
        if isinstance(module,nn.Linear):
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=std)
    #function to load pre-trained weights
    @classmethod
    def from_pretrained(cls,model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained model:",model_type)
        
        config_args = {
            'gpt2': dict(n_layer = 12, n_head = 12, n_embed = 768),
            'gpt2-medium': dict(n_layer = 24, n_head = 16, n_embed = 1024),
            'gpt2-large': dict(n_layer = 36, n_head = 20, n_embed = 1280),
            'gpt2-xl': dict(n_layer = 48, n_head = 25, n_embed = 1600),
            
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #these are not parameters but masks
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # print(set(sd_keys_hf)-set(sd_keys))

        assert len(sd_keys_hf)==len(sd_keys), f"mismatched_keys:{len(sd_keys_hf)!=len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        return model
        
    
    def forward(self,idx,targets = None):
        B,T = idx.shape

        assert T <= self.config.block_size,f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx) #(B,T,C)
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device) #(T)
        pos_emb = self.transformer.wpe(pos) #(T,C)
        x = tok_emb + pos_emb

        #forward pass
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits , loss
    
    def configure_optimizers(self,weight_decay, learning_rate,device):
        # collect all parameters that require grad
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        ndecay_params = [p for n,p in param_dict.items() if p.dim()<2]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in ndecay_params)
        
        print(f"num decayed paramter tensors:{len(decay_params)}, with {num_decay_params} ")
        print(f"num non decayed paramter tensors:{len(ndecay_params)}, with {num_non_decay_params}")
        
        optim_groups = [
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':ndecay_params,'weight_decay':0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9,0.95),eps=1e-8,fused=use_fused)
        
        return optimizer

import tiktoken

#implement dataloader
class DataLoaderLite():
    def __init__(self,B,T):
        self.B = B
        self.T = T
        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        self.current_position = 0
    
    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+(B*T)+1]
        # buf = buf.to(device)
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B*T
        if self.current_position + (B*T) + 1 > len(self.tokens):
            self.current_position = 0
        
        return x,y
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)    

torch.set_float32_matmul_precision('high')


total_batch_size = 524288 #2**19 tokens
B = 16
T = 1024
assert total_batch_size % (B*T) == 0, "make sure total_batch_size is divisible by B*T"
grad_accum_steps = total_batch_size//(B*T)
print(f"grad_accum_steps:{grad_accum_steps}")

train_loader = DataLoaderLite(B=B,T=T)

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTconfig(vocab_size=50304))


model.to(device)
# model = torch.compile(model)


max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
weight_decay = 0.1

# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
optimizer = model.configure_optimizers(weight_decay,learning_rate = max_lr,device = device)


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    
    if it>max_steps:
        return min_lr
    
    decay_ratio = (it-warmup_steps)/(max_steps-warmup_steps)
    coeff = 0.5*(1.0 + math.cos(math.pi*decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    
    loss_accum = 0.0
    #gradient accumulation
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss = model(x,y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    torch.cuda.synchronize()
    t1= time.time()
    dt = t1-t0
    print(f"step {step} dt:{dt*1000:.2f}ms loss:{loss_accum.item():.6f}")
    
#get logits
# logits,loss = model(x,y)
# print(loss)
#we need tokenizer to convert text to numbers
num_return_sequences = 5
max_length = 30

import sys;sys.exit(0)
model.eval()

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) <  max_length:
    with torch.no_grad():
        logits = model(x)
        
        #take last time step
        logits = logits[:,-1,:]
        
        probs = F.softmax(logits,dim=-1)
        
        topk_probs, topk_indices = torch.topk(probs,50,dim=-1)
        
        #sample from topk
        ix = torch.multinomial(topk_probs,1)
        
        xcol = torch.gather(topk_indices,-1,ix)
        
        x = torch.cat([x,xcol],dim=1)


for i in range(num_return_sequences):
    tokens = x[i,:].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)