import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import time
import math
import os
from functools import partial
from torchao.float8 import convert_to_float8_training

# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=2 train.py
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "need cuda for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing
else:
    # non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

from dataloader import DataLoaderLite
torch.set_float32_matmul_precision('high')

def filter_linear_layers(module, fqn, first_layer_name=None, last_layer_name=None):
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    # For stability reasons, we skip the first and last linear layers
    # Otherwise can lead to the model not training or converging properly
    if fqn in (first_layer_name, last_layer_name):
        return False
    return True

model = GPT(GPTConfig(vocab_size=50304))
first_linear = None
last_linear = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if first_linear is None:
            first_linear = name
        last_linear = name

func = partial(filter_linear_layers, first_layer_name=first_linear, last_layer_name=last_linear)

convert_to_float8_training(model, module_filter_fn=func)

model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

# optimize
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
training_steps = 50
bias_update_gamma = 0.01

total_batch_size = 524288 # 2**19, around 0.5M, in number of tokens
B = 32
T = 1024

assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8, lr=6e-4, weight_decay=0.1, fused=True)
optimizer = raw_model.configure_optimizers(
    learning_rate=max_lr, 
    device=device, 
    weight_decay=0.1, 
    grad_accum_steps=grad_accum_steps, 
    bias_update_gamma=bias_update_gamma,
    master_process=master_process
    )

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

if master_process:
    print(f"total_batch_size: {total_batch_size} | B: {B} | T: {T} | grad_accum_steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, master_process, split="train")

for step in range(training_steps):
    model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    balance_loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss, balance_loss = model(x, y)
        balance_loss = balance_loss / grad_accum_steps
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        balance_loss_accum += balance_loss.detach()
        loss += balance_loss
        loss.backward()
    # if ddp:
    #     dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    #     dist.all_reduce(balance_loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU finish work
    t1 = time.time()
    dt = t1-t0
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1-t0)
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | bl: {balance_loss_accum.item():.6f} | norm:{norm:.4f} | lr:{lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec:{tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

# Example usage
num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to(device)

# generate, x (B, T) with B=5, T=6

while x.size(1) < max_length:
    # forward pass
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
    # take the logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(logits, dim=-1) # (B, vocab_size)
    # do top-50 sampling
    topk_probs, topk_idx = torch.topk(probs, 50, dim=-1) # (B, 50)
    # select a token from the top-k tokens
    ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
    # gather the corresponding indices
    xcol = torch.gather(topk_idx, -1, ix) # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1) # (B, T+1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens) 
    print(decoded)