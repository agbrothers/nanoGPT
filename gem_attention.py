import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x, mn, mx, bias=0.9, eps=1e-10):
    return  (1-bias+eps) * ( x-mn) / (mx-mn+eps) + bias

def unnorm(x, mn, mx, bias=0.9, eps=1e-10):
    return ((x-bias) * (mx-mn+eps) / (1-bias+eps)) + mn

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

        # self.p = nn.Parameter(torch.normal(mean=1/p_scale, std=0.02, size=(config.n_embd,))) ## B
        self.p = nn.Parameter(torch.ones((config.n_embd,)))
        self.p_max = p_max
        self.p_min = p_min
        self.debug = False

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

        ## NOTE - GEM ATTENTION: CLAMPS AND SHIFTS TO PREVENT DISCONTINUITIES
        # p = self.p
        p = (self.p.sign()+0.5).sign() * torch.clamp(self.p.abs(), min=self.p_min, max=self.p_max)
        # assert not torch.any(torch.isnan(p)), f"Nans in GeM p: {p}"
        v_min = torch.min(v, dim=1, keepdim=True)[0]
        v_max = torch.max(v, dim=1, keepdim=True)[0]
        v = norm(v, v_min, v_max)
        v = torch.pow(v, p)

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
        assert not torch.any(torch.isnan(mean)), f"Nans in GeM mean max: {mean.max()} \n min: {mean.min()}"

        ## GeM COMPUTE INVERSE POWER
        y = torch.pow(mean, 1/p) 
        assert not torch.any(torch.isnan(y)), f"Nans in GeM y max: {y.max()} \n min: {y.min()}"
        y = unnorm(y, v_min, v_max)

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

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    


if __name__ == "__main__":

    from model import GPTConfig
    
    config = GPTConfig(**{
        "gem": True,
        "block_size": 1024, #5
        "n_head": 8, # 1
        "n_embd": 768, # 3
        "dropout": 0.0,
    })

    gem = CausalGemAttention(config)

    x = torch.rand(32,1024,768) * -500

    # x = torch.Tensor([
    #     [
    #         [1,2,3,4,5],
    #         [0,1,2,3,4],
    #         [2,2,2,2,2],
    #         # [-2,-2,-2,-2,-2],
    #     ],
    # ])
    gem(x)
