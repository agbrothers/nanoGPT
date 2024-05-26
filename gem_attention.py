import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x, mn=-5, mx=5, bias=0.25):
    # return ( x-mn) / (mx-mn) + bias
    # return  (1-bias) * ( x-mn) / (mx-mn) + bias
    return  (x-mn) / (mx-mn)

def unnorm(y, mn=-5, mx=5, bias=0.25):
    # return (y-bias) * (mx-mn) + mn
    # return ((y-bias) * (mx-mn) / (1-bias)) + mn
    return y * (mx-mn) + mn

# def norm(x, mn, mx, bias=0.9, eps=1e-10):
#     return  (1-bias+eps) * ( x-mn) / (mx-mn+eps) + bias

# def unnorm(x, mn, mx, bias=0.9, eps=1e-10):
#     return ((x-bias) * (mx-mn+eps) / (1-bias+eps)) + mn

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalGemAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        ## GEM PARAMETERS
        p_min=1e-4
        p_max=1e+3

        # self.norm = LayerNorm(config.n_embd, bias=config.bias)
        # self.p = nn.Parameter(torch.normal(mean=1/p_scale, std=0.02, size=(config.n_embd,))) ## B
        # self.p = nn.Parameter(torch.ones((config.n_embd,)), requires_grad=False)
        self.p = nn.Parameter(torch.ones((config.n_embd,)))
        # self.p = torch.ones((config.n_embd,))
        self.p_max = p_max
        self.p_min = p_min
        self.v_min = 1e-10
        self.shift = 5
        self.debug = False
        self.float_min=1e-45,

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        ## NOTE - OVERRIDE INPUTS TO VERIFY GeM ATTENTION OUTPUT
        # q = q*0
        # v = x

        # ## NOTE - GEM ATTENTION: CLAMPS AND SHIFTS TO PREVENT DISCONTINUITIES
        # p = self.p
        # # p = torch.ones((self.n_embd,))
        # # p = (self.p.sign()+0.5).sign() * torch.clamp(self.p.abs(), min=self.p_min, max=self.p_max)
        # # assert not torch.any(torch.isnan(p)), f"Nans in GeM p: {p}"
        # # v_min = torch.min(v, dim=1, keepdim=True)[0]
        # # v_max = torch.max(v, dim=1, keepdim=True)[0]
        # # v = norm(v, v_min, v_max)
        # v = norm(v)
        # ## LAYERNORM?
        # # t = self.norm(v)
        # # v = torch.pow(v, p)
        # # v = torch.exp(torch.log(v) * p)

        ## NOTE: FROM WIKITEXT2
        ## CLAMP AND SHIFT p,v TO PREVENT GeM DISCONTINUITIES
        p = (self.p.sign()+0.5).sign() * torch.clamp(self.p.abs(), min=self.p_min, max=self.p_max)
        # p = (self.p.sign()+0.5).sign() * torch.clamp(self.p.abs()*self.p_scale, min=self.p_min, max=self.p_max)
        # p = (self.p.sign()+0.5).sign() * torch.clamp(symexp(self.p).abs(), min=self.p_min, max=self.p_max)
        # p = (self.p.sign()+0.5).sign() * torch.clamp(symten(self.p).abs(), min=self.p_min, max=self.p_max)
        v = torch.clamp(torch.abs(v + self.shift), min=self.v_min)
        
        ## RAISE v TO p VIA THE LOG-SUM-EXP TRICK 
        z = p * torch.log(v)
        z_max = z.max(dim=1)[0].unsqueeze(1) 
        v = torch.exp(z - z_max) 

        ## SPLIT ATTENTION HEADS
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            mean = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
            # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            mean = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        mean = mean.transpose(1, 2).contiguous().view(B, T, C) 
        # assert not torch.any(torch.isnan(mean)), f"Nans in GeM mean max: {mean.max()} \n min: {mean.min()}"

        # ## GeM COMPUTE INVERSE POWER
        # # y = torch.exp(torch.log(mean) / p)
        # # y = torch.pow(mean, 1/p)
        # # y = mean
        # # assert not torch.any(torch.isnan(y)), f"Nans in GeM y max: {y.max()} \n min: {y.min()}"
        # # y = unnorm(y, v_min, v_max)
        # y = unnorm(y)

        ## NOTE: FROM WIKITEXT2
        y = torch.exp((z_max + torch.log(mean)) / p)
        y = y - self.shift

        # ## NOTE: GROUND TRUTH
        # def gen_mean(x, p=2):
        #     return torch.pow(torch.pow(x, p).mean(dim=0), 1/p)
        
        # def gen_mean_safe(x, p=2, s=5):
        #     z = p * torch.log(x+s)
        #     z_max = z.max(dim=0)[0].unsqueeze(0) 
        #     mean = torch.exp(z - z_max).mean(dim=0)
        #     return torch.exp((z_max + torch.log(mean)) / p) - s
        
        # gen_mean(x[0,:1], 2)
        # gen_mean(x[0,:2], 2)
        # gen_mean(x[0,:3], 2)
        # y
        # gen_mean_safe(x[0,:1], 2, 5)
        # gen_mean_safe(x[0,:2], 2, 5)
        # gen_mean_safe(x[0,:3], 2, 5)

        # x[0,:1].mean(dim=0)
        # x[0,:2].mean(dim=0)
        # x[0,:3].mean(dim=0)

        # if self.debug:
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))
        #     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #     att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #     att = F.softmax(att, dim=-1)
        #     att = self.attn_dropout(att)
            
        #     p = torch.ones_like(p) * -300

        #     if self.p[0] == 1:
        #         y_test = att @ x
        #         assert torch.all(torch.abs(y - y_test) < 1e-5)
        #     if self.p[0] == 2:
        #         y_test = torch.sqrt(att @ x.pow(2))
        #         assert torch.all(torch.abs(y - y_test) < 1e-5)
        #     if self.p[0] == 3:
        #         y_test = torch.pow(att @ x.pow(3), 1/3)
        #         assert torch.all(torch.abs(y - y_test) < 1e-5)
        #     if self.p[0] == 4:
        #         y_test = torch.pow(att @ x.pow(4), 1/4)
        #         assert torch.all(torch.abs(y - y_test) < 1e-5)


        # torch.max(old_y - y)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    


if __name__ == "__main__":

    from model import GPTConfig
    
    # config = GPTConfig(**{
    #     "gem": True,
    #     "block_size": 1024, #5
    #     "n_head": 8, # 1
    #     "n_embd": 768, # 3
    #     "dropout": 0.0,
    # })
    # x = torch.rand(32,1024,768)

    config = GPTConfig(**{
        "gem": True,
        "block_size": 3,
        "n_head": 1, # 1
        "n_embd": 5, # 3
        "dropout": 0.0,
    })    
    x = torch.Tensor([
        [
            [1,2,3,4,5],
            [0,1,2,3,4],
            [2,2,2,2,2],
            # [-2,-2,-2,-2,-2],
        ],
    ])

    gem = CausalGemAttention(config)
    y = gem(x)
