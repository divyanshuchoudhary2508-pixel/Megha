import torch
import torch.nn as nn
from .config import MeghaConfig

class MeghaBlock(nn.Module):
    def __init__(self, config: MeghaConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model, 
            num_heads=config.n_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attention_mask=None):
        # x shape: (B, T, C)
        # Using causal=True supported in torch >= 2.0
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False, is_causal=True)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class MeghaModel(nn.Module):
    def __init__(self, config: MeghaConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.blocks = nn.Sequential(*[MeghaBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
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
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        # Basic greedy generation for testing
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # focus on last time step
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
