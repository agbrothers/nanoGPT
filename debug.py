import os
import torch

from model import GPTConfig, GPT



# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = './out'
eval_interval = 100 #2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True #False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gem_gpt2_squash' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
gem = True
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------



model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) #, gem=gem) # start with model_args from command line

print(f"Resuming training from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join("/home/brothag1/code/gem/nanoGPT/out", 'ckpt_baseline_3.52.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']: #,'gem']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']



## EVALUATE SIMILARITY OF EMB + OUT PROJ TO PSEUDO INVERSE
# model.transformer.h[1].attn.c_proj.weight
A = model.transformer.h[1].attn.c_proj.weight
P = model.transformer.wpe.weight
U = model.transformer.wte.weight
V = model.lm_head.weight
I = V @ U.T
# I = V.T @ U
i = torch.eye(n=len(I))


(I-i).norm()
(I/I.norm() - i/i.norm()).norm()

torch.linalg.matrix_rank(A) / A.shape[1]

## COLUMNS OF U ARE ALL LINEARLY INDEPENDENT??????
torch.linalg.matrix_rank(U) / U.shape[1]
torch.linalg.matrix_rank(V) / V.shape[1]

## COLUMNS OF U ARE NEARLY ORTHOGONAL???
N = U / U.norm(dim=0)
i = torch.eye(n=N.shape[1])
(N.T @ N - i).norm()
RMS(N.T @ N - i)
RMS(i)
RMS(N.T @ N)

torch.linalg.det(N.T @ N)

def RMS(X):
    return torch.sqrt(X.square().sum()/X.shape.numel())

torch.dot(N[:, 0], N[:, 1])
torch.dot(A[:, 0], A[:, 1])

import matplotlib.pyplot as plt
plt.imshow(I.detach().numpy(), cmap='inferno', interpolation='nearest')
plt.savefig("heatmap_hidden.png")
plt.imshow((A.T @ A).abs().detach().numpy(), cmap='inferno', interpolation='nearest')
plt.savefig("heatmap_proj_abs.png")

x = torch.zeros((gptconf.vocab_size))
x[0] = 1
y = torch.zeros((gptconf.vocab_size))
y[1] = 1

## ADDITIVITY
torch.all(U.T @ x + U.T @ y == U.T @ (x+y))
## HOMOGENEITY
torch.all(U.T @ (10 * x) == 10 * U.T @ x)

## U/V token embedding matrices are approximately pseudo-invertible
y = V @ U.T @ x
logits = torch.softmax(y, dim=-1)

## AA^{+} need not be the general identity matrix, but it maps all column vectors of A to themselves
diff = U @ U.T @ U - U
diff = V @ U.T @ V - V


## COMPUTE ACTUAL PSEUDO INVERSE AND COMPARE
## lm_head.weigth pinv(U) rather than U itself?
Vt = torch.linalg.pinv(V)
diff = V @ Vt @ V - V
diff = V @ torch.linalg.lstsq(V, V).solution - V

## U/V token embedding matrices are approximately pseudo-invertible
# Ut = torch.linalg.pinv(U.T, rtol=len(U.T)*torch.finfo().eps)
Ut = torch.linalg.pinv(U.T)
x = torch.zeros((gptconf.vocab_size))
x[0] = 1
y = Ut @ U.T @ x
logits = torch.softmax(y, dim=-1)



## CHECK COMBINED WPE + WTE THEN PSEUDO INVERSE
x = torch.zeros((gptconf.vocab_size))
x[0] = 1
y = V @ (U.T @ x + P[0])
logits = torch.softmax(y, dim=-1)

y = V @ (U.T @ x + P[1])



