import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import time
from transformers import get_cosine_schedule_with_warmup

device = "cpu"
torch.manual_seed(42)
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(42)
elif torch.mps.is_available() and hasattr(torch.backends, "mps"):
    device = "mps"
    torch.mps.manual_seed(42)
print("device:", device)

# # prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to(device)

# with open('input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long)
# x = buf[:-1].view(B, T).to(device)
# y = buf[1:].view(B, T).to(device)

from dataloader import DataLoaderLite
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

# optimize
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
training_steps = 50
bias_update_gamma = 0.01



total_batch_size = 524288 # 2**19, around 0.5M, in number of tokens
B = 8
T = 1024

train_loader = DataLoaderLite(B, T)
assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
# optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8, lr=6e-4, weight_decay=0.1, fused=True)
optimizer = model.configure_optimizers(learning_rate=max_lr, device=device, weight_decay=0.1, grad_accum_steps=grad_accum_steps, bias_update_gamma=bias_update_gamma)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
print(f"total_batch_size: {total_batch_size} | B: {B} | T: {T} | grad_accum_steps: {grad_accum_steps}")

for step in range(training_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    balance_loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss, balance_loss = model(x, y)
        balance_loss = balance_loss / grad_accum_steps
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        balance_loss_accum += balance_loss.detach()
        loss += balance_loss
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    torch.cuda.synchronize() # wait for the GPU finish work
    t1 = time.time()
    dt = t1-t0
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1-t0)
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | bl: {balance_loss_accum.item():.6f} | norm:{norm:.4f} | lr:{lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec:{tokens_per_sec:.2f}")

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