import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from einops import rearrange

# Assuming this will be part of the openworld package
# from ....config import LongContextConfig # If LongContextConfig moves to openworld.config
# For now, let's redefine it here or assume it's passed in.
# The one in src/openworld/config.py is a good central place.
# We need to ensure imports are consistent once files are settled.

logger = logging.getLogger(__name__)

@dataclass
class LongContextConfig:
    # This definition is duplicated from the main src/openworld/config.py
    # Ideally, this class would be imported from there to avoid redundancy.
    # For this move operation, I'll keep it to ensure the file is self-contained initially.
    d_model: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    max_seq_len: int = 131072
    attention_type: str = "sliding_window" # Or other types like "flash_attention"
    window_size: int = 4096
    use_alibi: bool = True
    use_flash_attention: bool = True # Check availability during init
    rope_base: int = 10000
    vocab_size: int = 32000 # Should match tokenizer

class ALiBiPositionalEmbedding(nn.Module):
    """Attention with Linear Biases for better extrapolation"""
    
    def __init__(self, n_heads: int, max_seq_len: int = 131072):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.slopes = self._get_slopes(n_heads)
        self.register_buffer("bias", self._compute_bias(max_seq_len))
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]
            
        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest_power = 2 ** math.floor(math.log2(n_heads))
            return torch.tensor(
                get_slopes_power_of_2(closest_power) + 
                get_slopes_power_of_2(2 * closest_power)[0::2][:n_heads - closest_power]
            )
    
    def _compute_bias(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float)
        pos_diff = -torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))
        return pos_diff.unsqueeze(0) * self.slopes.view(-1, 1, 1)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        if self.bias.shape[1] < seq_len:
            # Recompute bias if needed for longer sequence
            # This might happen if max_seq_len was just an initial estimate
            logger.warning(f"Recomputing ALiBi bias for seq_len {seq_len} (was {self.bias.shape[1]})")
            self.bias = self._compute_bias(seq_len).to(self.bias.device)
        return self.bias[:, :seq_len, :seq_len]

class SlidingWindowAttention(nn.Module):
    """Memory-efficient attention with sliding window"""
    
    def __init__(self, config: LongContextConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.window_size = config.window_size
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        if config.use_alibi:
            self.alibi = ALiBiPositionalEmbedding(config.n_heads, config.max_seq_len)
        
        self.flash_attn = None
        if config.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn = flash_attn_func
                logger.info("Flash Attention enabled for SlidingWindowAttention.")
            except ImportError:
                logger.warning("Flash Attention requested but not available. Install with `pip install flash-attn`.")

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.flash_attn is not None and q.is_cuda: # FlashAttention typically requires CUDA
            # Causal mask is implicit for causal=True
            # Sliding window is handled by window_size argument in flash_attn_func
            output = self.flash_attn(
                q, k, v,
                dropout_p=0.1 if self.training else 0.0,
                causal=True, # Assuming causal language model
                window_size=(self.window_size, self.window_size) # Ensure this is correct for flash_attn version
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            if hasattr(self, 'alibi'):
                attn_scores += self.alibi(seq_len).to(x.device)
            
            # Manual sliding window mask + causal mask
            # Create a causal mask (upper triangle)
            causal_mask_val = torch.finfo(attn_scores.dtype).min
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            
            # Create sliding window mask
            # A position i can attend to positions j where max(0, i - window_size) <= j <= i
            # Or for bidirectional: max(0, i - window_size//2) <= j <= min(seq_len-1, i + window_size//2)
            # For causal, it's max(0, i - window_size + 1) <= j <= i
            row_indices = torch.arange(seq_len, device=x.device).view(-1, 1)
            col_indices = torch.arange(seq_len, device=x.device).view(1, -1)
            
            # Window constraint for causal: j > i - window_size AND j <= i
            outside_window = (col_indices > row_indices) | (col_indices <= row_indices - self.window_size)
            mask = mask | outside_window # Combine causal and window

            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), causal_mask_val)
            
            if attention_mask is not None:
                # Expand attention_mask to match attn_scores dimensions
                # Expected shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
                attn_scores = attn_scores + attention_mask # Additive mask
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_probs, v)
        
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)

class LongContextTransformerBlock(nn.Module):
    """Transformer block optimized for long sequences"""
    
    def __init__(self, config: LongContextConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = SlidingWindowAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_norm = self.ln1(x)
        attn_output = self.attention(x_norm, attention_mask=attention_mask)
        x = residual + attn_output
        
        residual = x
        x_norm = self.ln2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output
        return x

class LongContextTransformer(nn.Module):
    """Complete long-context transformer implementation"""
    
    def __init__(self, config: LongContextConfig):
        super().__init__()
        self.config = config
        # vocab_size should be part of config, using default from dataclass if not overridden
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            LongContextTransformerBlock(config, i) for i in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        
        # Prepare attention mask if provided (for padding)
        # FlashAttention and manual attention might handle it differently.
        # For manual attention, a common way is to convert padding mask to additive bias.
        extended_attention_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: # [batch_size, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
            elif attention_mask.dim() == 3: # [batch_size, seq_len, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1) # [B, 1, S, S]
            extended_attention_mask = extended_attention_mask.to(dtype=x.dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(x.dtype).min

        for block in self.blocks:
            x = block(x, attention_mask=extended_attention_mask)
        
        x_norm = self.ln_f(x)
        # Output logits are typically calculated by projecting back to vocab size
        # This usually involves tying weights with the embedding layer or a separate LM head
        logits = torch.matmul(x_norm, self.embedding.weight.t()) # Tied weights
        
        return {
            "logits": logits,
            "last_hidden_state": x_norm # Common to return last hidden state
        } 