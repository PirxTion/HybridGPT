import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import inspect
import numpy as np
from src.kernel import RMSNormTritonAutogradFuncClass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    rope_theta: float = 500000
    n_activated_experts: int = 1
    n_routed_experts: int = 8
    load_balance_alpha =  0.0001

class RMSNormTriton(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(H)).to("cuda")

    def forward(self, x):
        return RMSNormTritonAutogradFuncClass.apply(x, self.weight)

    @staticmethod
    def apply(x, weight, eps=1e-5):
        return RMSNormTritonAutogradFuncClass.apply(x, weight, eps)
        
class Gate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd
        self.topk = config.n_activated_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.weight = nn.Linear(self.dim, config.n_routed_experts, bias=False)
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts), requires_grad=False)
        self.f_i_accum = None
    
    def forward(self, x):
        scores = self.weight(x)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        original_scores = scores
        scores = scores + self.bias
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = torch.gather(original_scores, 1, indices)
        expert_counts = torch.bincount(indices.flatten(), minlength=self.weight.out_features)
        expert_probs = original_scores.mean(dim=0)
        shape = x.shape
        T = np.prod(shape[:-1])
        f_i = (expert_counts.float() * self.n_routed_experts) / (self.n_activated_experts * T + 1e-6)
        if self.f_i_accum is None:
            self.f_i_accum = torch.zeros_like(f_i)
            self.f_i_accum.requires_grad = False
        self.f_i_accum += f_i
        stats = {
            "f_i": f_i,
            "expert_probs": expert_probs
        }
        return weights.type_as(x), indices, stats

class MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate = Gate(config)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_routed_experts)])
        self.shared_experts = MLP(config)
        self.alpha = config.load_balance_alpha
    
    def forward(self, x):
        shape = x.shape
        x = x.view(-1, shape[-1])
        weights, indices, gate_stats = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        f_i = gate_stats["f_i"]
        expert_probs = gate_stats["expert_probs"]
        load_balance_loss = self.alpha * torch.sum(f_i * expert_probs)
        return (y + z).view(shape), load_balance_loss

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads in a batch.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def reshape_for_broadcast(self, freqs_cis, x):
        ndim = x.ndim # x has shape (B, T, n_head, C // n_head)
        assert 0 <= 1 < ndim
        freqs_cis = freqs_cis[0:x.shape[1],:]
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"x:{x.shape}, freqs_cis:{freqs_cis.shape}" # match with T and hd
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # shape is (1, T, 1, C // n_head)
        return freqs_cis.reshape(*shape)
    
    def apply_rotary_emb(self, xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # (B, T, n_head, C // 2 * n_head, 2), then turned into complex number
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # merge the last two dimensions
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, freqs_cis):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        # calculate query, key, values for all heads in batch and move head forward to the batch dim
        # n_head=12, C=768, C // n_head=64
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis) # apply rotary embedding
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 2)
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(config.n_embd * 2, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNormTriton(config.n_embd)
        self.ln_2 = RMSNormTriton(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.moe = MoE(config)
        self.freqs_cis = self.precompute_freqs_cis(config.n_embd // config.n_head, config.block_size, config.rope_theta)

    def forward(self, x):
        self.freqs_cis = self.freqs_cis.to(x.device)
        x = x + self.attn(self.ln_1(x), self.freqs_cis) # normalize before self-attention, not after
        o, load_balance_loss = self.moe(self.ln_2(x))
        x = x + o
        return x, load_balance_loss
    
    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # generate list of \theta^{-2i / d}
        t = torch.arange(end, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs) # multiply each position idx by each frequency, t x freqs
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # store each elements as complex number, get sin and cos
        return freqs_cis

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # input embedding
            # wpe = nn.Embedding(config.block_size, config.n_embd), # positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # the transformer
            ln_f = RMSNormTriton(config.n_embd), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # output embedding

        self.transformer.wte.weight = self.lm_head.weight # tie weights of input and output embeddings
        # self.transformer.wpe.POSITION_SCALE_INIT = 1
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            # if hasattr(module, 'POSITION_SCALE_INIT'):
            #     std = 0.01
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def configure_optimizers(self, weight_decay, learning_rate, device, grad_accum_steps, bias_update_gamma, master_process):
        # get all parameters that require gradients
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}

        # any parameters taht is 2D will be weight decayed
        decay_params = []
        nodecay_params = []
        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        # create optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_available and device == 'cuda'
        if master_process:
            print(f"number of weight decay parameters: {num_decay_params}")
            print(f"number of no weight decay parameters: {num_nodecay_params}") 
            print(f"using fused AdamW: {used_fused}")
        optimizer = CustomOptimizer(
            model=self,  
            grad_accum_steps=grad_accum_steps,
            bias_update_gamma=bias_update_gamma,
            params=optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=used_fused
            )
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward sequence of length {T}, model block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        # pos_emb = self.transformer.wpe(pos) # (T, C)
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        # x = tok_emb + pos_emb
        # forward the blocks of the transformer
        # x = self.transformer.rope(tok_emb)
        total_load_balance_loss = 0.0
        x = tok_emb
        for block in self.transformer.h:
            x, block_load_loss = block(x) # (B, T, C)
            total_load_balance_loss += block_load_loss
        # forward the final layer norm
        x = self.transformer.ln_f(x)
        # forward the classifier head
        logits = self.lm_head(x)

        # calculate the loss
        loss = None
        if targets is not None:
            # print("targets shape: ", targets.shape)
            # print("logits shape: ", logits.shape)
            # print(targets.view(-1).shape)
            # print(logits.view(-1, logits.size(-1)).shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, total_load_balance_loss
    
class CustomOptimizer(torch.optim.AdamW):
        def __init__(self, model, grad_accum_steps, bias_update_gamma, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model
            self.grad_accmu_steps = grad_accum_steps
            self.bias_update_gamma = bias_update_gamma

        def step(self, closure=None):
            super().step(closure)
            
            # update bias in MoE gate
            for module in self.model.modules():
                if isinstance(module, Gate):
                    f_i_avg = module.f_i_accum / self.grad_accmu_steps
                    for i in range(module.n_routed_experts):
                        if f_i_avg[i] > 1.0:
                            module.bias.data[i] -= self.bias_update_gamma
                        elif f_i_avg[i] < 1.0:
                            module.bias.data[i] += self.bias_update_gamma
                    # print(f_i_avg)
                    # zero out f_i
                    module.f_i_accum = None
