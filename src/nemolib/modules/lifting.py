import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .base_module.cross_attention import TransformerCrossAttnLayer

# ================================================================
# FlashMHA — drop-in replacement for nn.MultiheadAttention
# ================================================================
class FlashMHA(nn.Module):
    """
    1:1 API compatible replacement for nn.MultiheadAttention,
    internally using torch.nn.functional.scaled_dot_product_attention
    which dispatches to FlashAttention kernels on supported GPUs.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
                need_weights=False, average_attn_weights=True):

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        # in_proj: Q, K, V
        proj = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = proj.chunk(3, dim=-1)

        # shape → (B, heads, N, head_dim)
        head_dim = C // self.num_heads
        q = q.view(B, Nq, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.num_heads, head_dim).transpose(1, 2)

        # Flash attention path
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # merge heads
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.out_proj(x)

        if need_weights:
            # Compute weights manually (slower)
            attn_scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = attn_scores.softmax(dim=-1)
            if average_attn_weights:
                attn = attn.mean(dim=1)
            return x, attn
        else:
            return x, None


# ================================================================
# FlashEncoderLayer (self-attention only)
# ================================================================
class FlashEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, batch_first, norm_first):
        super().__init__()
        self.self_attn = FlashMHA(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = activation

    def forward(self, x):
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn(x, x, x)[0])
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ================================================================
# FlashCrossAttnLayer (cross attention + FF)
# ================================================================
class FlashCrossAttnLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward, dropout, activation,
        batch_first=True, norm_first=False
    ):
        super().__init__()
        self.multihead_attn = FlashMHA(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = activation

    def forward(self, tgt, memory, value=None):
        if value is None:
            value = memory

        if self.norm_first:
            tgt = tgt + self.multihead_attn(self.norm1(tgt), self.norm1(memory), self.norm1(value))[0]
            tgt = tgt + self._ff_block(self.norm2(tgt))
        else:
            tgt = self.norm1(tgt + self.multihead_attn(tgt, memory, value)[0])
            tgt = self.norm2(tgt + self._ff_block(tgt))
        return tgt

    def _ff_block(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ================================================================
# FlashDecoderLayer (self + cross attention)
# ================================================================
class FlashDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, batch_first=True, norm_first=False):
        super().__init__()

        self.self_attn = FlashMHA(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = FlashMHA(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = activation

    def forward(self, tgt, memory):
        if self.norm_first:
            tgt = tgt + self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt))[0]
            tgt = tgt + self.cross_attn(self.norm2(tgt), self.norm2(memory), self.norm2(memory))[0]
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            tgt = self.norm1(tgt + self.self_attn(tgt, tgt, tgt)[0])
            tgt = self.norm2(tgt + self.cross_attn(tgt, memory, memory)[0])
            tgt = self.norm3(tgt + self._ff_block(tgt))
        return tgt

    def _ff_block(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ================================================================
# Updated factory functions
# ================================================================
def lifting_make_self_attention_layers(in_dim, layers, norm_first=False, use_flash_attn=False):
    latent_dim = int(4 * in_dim)

    if use_flash_attn:
        blocks = [
            FlashEncoderLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=latent_dim,
                dropout=0.0,
                activation=F.gelu,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    else:
        blocks = [
            nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=latent_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    return nn.ModuleList(blocks)


def lifting_make_cross_attention_layers(in_dim, layers, norm_first=False, use_flash_attn=False):
    latent_dim = int(4 * in_dim)

    if use_flash_attn:
        blocks = [
            FlashCrossAttnLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=latent_dim,
                dropout=0.0,
                activation=F.gelu,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    else:
        blocks = [
            TransformerCrossAttnLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=latent_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    return nn.ModuleList(blocks)


def lifting_make_decoder_layers(in_dim, layers, norm_first=False, use_flash_attn=False):
    latent_dim = int(4 * in_dim)

    if use_flash_attn:
        blocks = [
            FlashDecoderLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=latent_dim,
                dropout=0.0,
                activation=F.gelu,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    else:
        blocks = [
            nn.TransformerDecoderLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=latent_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    return nn.ModuleList(blocks)


# ================================================================
# Final lifting module
# ================================================================
class lifting(nn.Module):
    def __init__(
        self,
        in_dim,
        decoder_layers: int = 0,
        cross_attention_layers: int = 0,
        self_attention_layers: int = 0,
        norm_first: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.decoder_layer = lifting_make_decoder_layers(
            in_dim, decoder_layers, norm_first, use_flash_attn
        )

        self.cross_attention_layer = lifting_make_cross_attention_layers(
            in_dim, cross_attention_layers, norm_first, use_flash_attn
        )

        self.self_attention_layer = lifting_make_self_attention_layers(
            in_dim, self_attention_layers, norm_first, use_flash_attn
        )

    def forward(self, x: torch.Tensor, latent_emb: torch.Tensor):
        """
        x: [b,t,c,h,w]
        latent_emb: [b, n, c]
        """
        b, t, c, h, w = x.shape

        latent = latent_emb
        x = rearrange(x, "b t c h w -> b (t h w) c")

        # Decoder layers
        for block in self.decoder_layer:
            latent = block(latent, x)

        # Cross-attention
        for block in self.cross_attention_layer:
            latent = block(latent, x)

        # Self-attention
        for block in self.self_attention_layer:
            latent = block(latent)

        return latent
