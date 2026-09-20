import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MeghaConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MeghaAttention(nn.Module):
    def __init__(self, config: MeghaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout_p = config.dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Fast PyTorch Scaled Dot Product Attention with automatic Causal Mask
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class MeghaBlock(nn.Module):
    def __init__(self, config: MeghaConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MeghaAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        hidden_dim = int(4 * config.d_model * 2 / 3)   # SwiGLU standard dimension scaling
        self.mlp = SwiGLU(config.d_model, hidden_dim, dropout=config.dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MeghaModel(nn.Module):
    def __init__(self, config: MeghaConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.blocks = nn.ModuleList([MeghaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Ignore -100 index so loss is ONLY calculated on answer tokens!
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=40, eos_token_id=None, repetition_penalty=1.05, do_sample=False):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            
            if not do_sample or temperature <= 0.0:
                # Greedy Search: pick exact highest probability token
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

        return idx
