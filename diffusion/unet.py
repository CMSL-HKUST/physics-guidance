# unet.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


jax.config.update("jax_default_matmul_precision", "float32")

# ==========================================
# 1. base func
# ==========================================

def diffusers_init():
    return nn.initializers.xavier_uniform()


class SinusoidalPosEmb(nn.Module):
    """
    embbed time.
    """
    dim: int

    @nn.compact
    def __call__(self, time):
        # time shape: (batch_size,)
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = time[:, None].astype(jnp.float32) * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        if self.dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb


# ==========================================
# 2. Resnet
# ==========================================

class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.0
    groups: int = 32
    output_scale_factor: float = 1.0

    @nn.compact
    def __call__(self, x, t_emb, train: bool = True):
        # --- Block 1 ---
        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        h = nn.silu(h)
        h = nn.Conv(
            self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
            kernel_init=diffusers_init(), dtype=jnp.float32
        )(h)

        # --- Time Step Injection ---
        if t_emb is not None:
            time_proj = nn.Dense(self.out_channels, kernel_init=diffusers_init())(nn.silu(t_emb))
            h = h + time_proj[:, None, None, :]

        # --- Block 2 ---
        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(h)
        h = nn.silu(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        h = nn.Conv(
            self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
            kernel_init=diffusers_init(), dtype=jnp.float32
        )(h)

        # --- Shortcut / Residual ---
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(
                self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding='VALID',
                kernel_init=diffusers_init(), dtype=jnp.float32
            )(x)

        return (x + h) / self.output_scale_factor


# ==========================================
# 3. Attention
# ==========================================

class AttentionBlock(nn.Module):
    num_heads: int = 1
    norm_num_groups: int = 32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        
        # 1. Norm
        residual = x
        x = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-5)(x)
        
        # 2. Projections
        x = x.reshape(B, H * W, C)
        
        q = nn.Dense(C, kernel_init=diffusers_init())(x)
        k = nn.Dense(C, kernel_init=diffusers_init())(x)
        v = nn.Dense(C, kernel_init=diffusers_init())(x)

        head_dim = C // self.num_heads
        q = q.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn_weights = nn.softmax(attn_weights, axis=-1)
        
        out = jnp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, H * W, C)
        
        out = nn.Dense(C, kernel_init=diffusers_init())(out)
        out = out.reshape(B, H, W, C)
        
        return residual + out


# ==========================================
# 4. up/down sample
# ==========================================

class Downsample2D(nn.Module):
    out_channels: int
    
    @nn.compact
    def __call__(self, x):
        return nn.Conv(
            self.out_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
            kernel_init=diffusers_init(), dtype=jnp.float32
        )(x)


class Upsample2D(nn.Module):
    out_channels: int
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method='nearest')
        return nn.Conv(
            self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
            kernel_init=diffusers_init(), dtype=jnp.float32
        )(x)


# ==========================================
# 5. UNet 
# ==========================================

class UNet(nn.Module):
    sample_size: int = 64
    in_channels: int = 1
    out_channels: int = 1
    block_out_channels: Tuple[int, ...] = (64, 128, 256, 512)
    layers_per_block: int = 2
    down_block_types: Tuple[str, ...] = (
        'DownBlock2D',
        'DownBlock2D',
        'AttnDownBlock2D',
        'AttnDownBlock2D',
    )
    up_block_types: Tuple[str, ...] = (
        'AttnUpBlock2D',
        'AttnUpBlock2D',
        'UpBlock2D',
        'UpBlock2D',
    )
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, timesteps, train: bool = True):

        # 1. Time Embedding
        time_embed_dim = self.block_out_channels[0] * 4

        t_emb = SinusoidalPosEmb(dim=self.block_out_channels[0])(timesteps * 1000)
        t_emb = nn.Dense(time_embed_dim, kernel_init=diffusers_init())(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(time_embed_dim, kernel_init=diffusers_init())(t_emb)

        # 2. Pre-process
        x = nn.Conv(
            self.block_out_channels[0], kernel_size=(3, 3), padding='SAME',
            kernel_init=diffusers_init(), dtype=jnp.float32
        )(x)
        
        down_block_res_samples = [x]

        # 3. Down Blocks
        for i, block_type in enumerate(self.down_block_types):
            is_final_block = (i == len(self.block_out_channels) - 1)
            current_channels = self.block_out_channels[i]
            
            for _ in range(self.layers_per_block):
                x = ResnetBlock(
                    in_channels=x.shape[-1],
                    out_channels=current_channels,
                    dropout_rate=self.dropout_rate
                )(x, t_emb, train)
                
                if "Attn" in block_type:
                    x = AttentionBlock(num_heads=4)(x)
                
                down_block_res_samples.append(x)

            if not is_final_block:
                x = Downsample2D(out_channels=current_channels)(x)
                down_block_res_samples.append(x)

        # 4. Mid Block
        mid_channels = self.block_out_channels[-1]
        x = ResnetBlock(
            in_channels=x.shape[-1], out_channels=mid_channels, dropout_rate=self.dropout_rate
        )(x, t_emb, train)
        
        x = AttentionBlock(num_heads=4)(x)
        
        x = ResnetBlock(
            in_channels=x.shape[-1], out_channels=mid_channels, dropout_rate=self.dropout_rate
        )(x, t_emb, train)

        # 5. Up Blocks
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        reversed_up_block_types = list(self.up_block_types)

        for i, block_type in enumerate(reversed_up_block_types):
            is_final_block = (i == len(self.block_out_channels) - 1)
            current_channels = reversed_block_out_channels[i]
            
            for _ in range(self.layers_per_block + 1):
                res_sample = down_block_res_samples.pop()
                
                if res_sample.shape[1] != x.shape[1] or res_sample.shape[2] != x.shape[2]:
                    res_sample = jax.image.resize(res_sample, x.shape, method='nearest')
                
                x = jnp.concatenate([x, res_sample], axis=-1)
                
                x = ResnetBlock(
                    in_channels=x.shape[-1],
                    out_channels=current_channels,
                    dropout_rate=self.dropout_rate
                )(x, t_emb, train)
                
                if "Attn" in block_type:
                    x = AttentionBlock(num_heads=4)(x)
            
            if not is_final_block:
                x = Upsample2D(out_channels=current_channels)(x)

        # 6. Post-process
        x = nn.GroupNorm(num_groups=32, epsilon=1e-5)(x)
        x = nn.silu(x)
        x = nn.Conv(
            self.out_channels, kernel_size=(3, 3), padding='SAME',
            kernel_init=diffusers_init(), dtype=jnp.float32
        )(x)

        return x