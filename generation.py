import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import time

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
train_loader = DataLoaderLite(B=16, T=1024)
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU finish work
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {i}, loss: {loss.item()}, dt: (dt:.2f)ms, tok/sec:{tokens_per_sec:.2f}")

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