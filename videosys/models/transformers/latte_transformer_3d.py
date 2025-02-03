# Adapted from Latte

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Latte: https://github.com/Vchitect/Latte
# --------------------------------------------------------


from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    ImagePositionalEmbeddings,
    PatchEmbed,
    PixArtAlphaCombinedTimestepSizeEmbeddings,
    PixArtAlphaTextProjection,
    SinusoidalPositionalEmbedding,
    get_1d_sincos_pos_embed_from_grid,
)
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange, repeat
from torch import nn

from videosys.core.distributed.comm import all_to_all_with_pad, gather_sequence, get_pad, set_pad, split_sequence
from videosys.core.distributed.parallel_mgr import ParallelManager
from videosys.core.pab.pab_mgr import (
    enable_pab,
    get_mlp_output,
    if_broadcast_cross,
    if_broadcast_mlp,
    if_broadcast_spatial,
    if_broadcast_temporal,
    save_mlp_output,
)
from videosys.models.transformers.eff_attn import EfficientAttention, XFormersAttnProcessor
from videosys.utils.utils import batch_func
from xformers.ops.fmha.attn_bias import BlockDiagonalMask


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU,) if USE_PEFT_BACKEND else (GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        block_idx: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = EfficientAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            processor=XFormersAttnProcessor(),
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = EfficientAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                processor=XFormersAttnProcessor(),
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # pab
        self.cross_last = None
        self.cross_count = 0
        self.spatial_last = None
        self.spatial_count = 0
        self.block_idx = block_idx
        self.mlp_count = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def set_cross_last(self, last_out: torch.Tensor):
        self.cross_last = last_out

    def set_spatial_last(self, last_out: torch.Tensor):
        self.spatial_last = last_out


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        org_timestep: Optional[torch.LongTensor] = None,
        all_timesteps=None,
        mask_dict: Optional[Dict[str, torch.Tensor]] = None,
        attn_bias_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
        
        # prepare filtered hidden states
        B, M, C = hidden_states.shape
        indices1 = mask_dict['spatial']['indices1']
        indices2 = indices1.squeeze()
        actual_indices = mask_dict['spatial']['actual_indices']
        filtered_hidden_states = self.get_filtered_tensor(hidden_states, indices1)
        # filtered_hidden_states = hidden_states.reshape(-1, C)
        # filtered_hidden_states = torch.index_select(filtered_hidden_states, 0, indices1.squeeze())
        # filtered_hidden_states = filtered_hidden_states.reshape(1, -1, C)


        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            shift_msa1 = self.get_filtered_tensor(shift_msa.expand(-1, M, -1), indices1)
            scale_msa1 = self.get_filtered_tensor(scale_msa.expand(-1, M, -1), indices1)
            gate_msa1 = self.get_filtered_tensor(gate_msa.expand(-1, M, -1), indices1)
            shift_mlp1 = self.get_filtered_tensor(shift_mlp.expand(-1, M, -1), indices1)
            scale_mlp1 = self.get_filtered_tensor(scale_mlp.expand(-1, M, -1), indices1)
            gate_mlp1 = self.get_filtered_tensor(gate_mlp.expand(-1, M, -1), indices1)
            norm_hidden_states = self.norm1(hidden_states) # this branch
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
            
            norm_filtered_hidden_states = self.norm1(filtered_hidden_states) # this branch
            norm_filtered_hidden_states = norm_filtered_hidden_states * (1 + scale_msa1) + shift_msa1
            
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output, filtered_attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            mask_dict=mask_dict,
            attn_bias_dict=attn_bias_dict,
            mode="spatial",
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output # this branch
            
            filtered_attn_output = gate_msa1 * filtered_attn_output


        hidden_states = attn_output + hidden_states
        filtered_hidden_states = filtered_attn_output + filtered_hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            broadcast_cross, self.cross_count = if_broadcast_cross(int(org_timestep[0]), self.cross_count)
            if broadcast_cross:
                hidden_states = hidden_states + self.cross_last
            else:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states # this branch
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output, filtered_attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    mask_dict=mask_dict,
                    attn_bias_dict=attn_bias_dict,
                    mode="spatial",
                    **cross_attention_kwargs,
                )

                hidden_states = attn_output + hidden_states
                filtered_hidden_states = filtered_attn_output + filtered_hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️

        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single": # this branch
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            
            norm_filtered_hidden_states = self.norm2(filtered_hidden_states)
            norm_filtered_hidden_states = norm_filtered_hidden_states * (1 + scale_mlp1) + shift_mlp1

        ff_output = self.ff(norm_hidden_states)
        ff_output_filtered = self.ff(norm_filtered_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output
            ff_output_filtered = gate_mlp1 * ff_output_filtered

        hidden_states = ff_output + hidden_states
        filtered_hidden_states = ff_output_filtered + filtered_hidden_states
        
        filtered_hidden_states = self.token_reuse(hidden_states, filtered_hidden_states, indices1.squeeze(), actual_indices, 'spatial')
        
        
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return filtered_hidden_states
        # return hidden_states
    
    

    def token_reuse(self, x_in, x_out, indices, actual_indices, mode):
        out = torch.zeros_like(x_in)
        out = out.reshape(-1, out.shape[-1])
        out.index_put_((indices,), x_out.squeeze())
        out = out.reshape(x_in.shape[0], x_in.shape[1], -1) 
        actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        if mode == 'spatial':
            out = out.permute(1, 0, 2)
            out = out.gather(1, actual_indices).permute(1, 0, 2)
        else:
            out = out.gather(1, actual_indices)
        return out

    def get_filtered_tensor(self, x, indices):
        x = x.reshape(-1, x.shape[-1])
        x = torch.index_select(x, 0, indices.squeeze())
        x = x.reshape(1, -1, x.shape[-1])
        return x


    def forward_org(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        org_timestep: Optional[torch.LongTensor] = None,
        all_timesteps=None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if enable_pab():
            broadcast_spatial, self.spatial_count = if_broadcast_spatial(int(org_timestep[0]), self.spatial_count)

        if enable_pab() and broadcast_spatial:
            attn_output = self.spatial_last
            assert self.use_ada_layer_norm_single
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
        else:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states) # this branch
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output # this branch

            if enable_pab():
                self.set_spatial_last(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            broadcast_cross, self.cross_count = if_broadcast_cross(int(org_timestep[0]), self.cross_count)
            if broadcast_cross:
                hidden_states = hidden_states + self.cross_last
            else:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states # this branch
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )

                if enable_pab():
                    self.set_cross_last(attn_output)

                hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if enable_pab():
            broadcast_mlp, self.mlp_count, broadcast_next, broadcast_range = if_broadcast_mlp(
                int(org_timestep[0]),
                self.mlp_count,
                self.block_idx,
                all_timesteps.tolist(),
                is_temporal=False,
            )

        if enable_pab() and broadcast_mlp:
            ff_output = get_mlp_output(
                broadcast_range,
                timestep=int(org_timestep[0]),
                block_idx=self.block_idx,
                is_temporal=False,
            )
        else:
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single": # this branch
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            if enable_pab() and broadcast_next:
                # spatial
                save_mlp_output(
                    timestep=int(org_timestep[0]),
                    block_idx=self.block_idx,
                    ff_output=ff_output,
                    is_temporal=False,
                )

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock_(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        block_idx: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)  # go here

        self.attn1 = EfficientAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            processor=XFormersAttnProcessor(),
        )

        # # 2. Cross-Attn
        # if cross_attention_dim is not None or double_self_attention:
        #     # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
        #     # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
        #     # the second cross attention block.
        #     self.norm2 = (
        #         AdaLayerNorm(dim, num_embeds_ada_norm)
        #         if self.use_ada_layer_norm
        #         else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        #     )
        #     self.attn2 = Attention(
        #         query_dim=dim,
        #         cross_attention_dim=cross_attention_dim if not double_self_attention else None,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )  # is self-attn if encoder_hidden_states is none
        # else:
        #     self.norm2 = None
        #     self.attn2 = None

        # 3. Feed-forward
        # if not self.use_ada_layer_norm_single:
        #     self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # pab
        self.last_out = None
        self.mlp_count = 0
        self.block_idx = block_idx
        self.count = 0

        # parallel
        self.parallel_manager: ParallelManager = None

    def set_last_out(self, last_out: torch.Tensor):
        self.last_out = last_out

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        org_timestep: Optional[torch.LongTensor] = None,
        all_timesteps=None,
        mask_dict: Optional[Dict[str, torch.Tensor]] = None,
        attn_bias_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # prepare filtered hidden states
        B, M, C = hidden_states.shape
        indices1 = mask_dict['temporal']['indices1']
        indices2 = indices1.squeeze()
        actual_indices = mask_dict['temporal']['actual_indices']
        filtered_hidden_states = self.get_filtered_tensor(hidden_states, indices1)
        

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:  # go here
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            
            shift_msa1 = self.get_filtered_tensor(shift_msa.expand(-1, M, -1), indices1)
            scale_msa1 = self.get_filtered_tensor(scale_msa.expand(-1, M, -1), indices1)
            gate_msa1 = self.get_filtered_tensor(gate_msa.expand(-1, M, -1), indices1)
            shift_mlp1 = self.get_filtered_tensor(shift_mlp.expand(-1, M, -1), indices1)
            scale_mlp1 = self.get_filtered_tensor(scale_mlp.expand(-1, M, -1), indices1)
            gate_mlp1 = self.get_filtered_tensor(gate_mlp.expand(-1, M, -1), indices1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)
            
            norm_filtered_hidden_states = self.norm1(filtered_hidden_states)
            norm_filtered_hidden_states = norm_filtered_hidden_states * (1 + scale_msa1) + shift_msa1
            
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        if self.parallel_manager.sp_size > 1:
            norm_hidden_states = self.dynamic_switch(norm_hidden_states, to_spatial_shard=True)

        attn_output, filtered_attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            mask_dict=mask_dict,
            attn_bias_dict=attn_bias_dict,
            mode="temporal",
            **cross_attention_kwargs,
        )

        if self.parallel_manager.sp_size > 1:
            attn_output = self.dynamic_switch(attn_output, to_spatial_shard=False)

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output
            filtered_attn_output = gate_msa1 * filtered_attn_output

        hidden_states = attn_output + hidden_states
        filtered_hidden_states = filtered_attn_output + filtered_hidden_states

        if enable_pab():
            broadcast_mlp, self.mlp_count, broadcast_next, broadcast_range = if_broadcast_mlp(
                int(org_timestep[0]),
                self.mlp_count,
                self.block_idx,
                all_timesteps.tolist(),
                is_temporal=True,
            )

        if enable_pab() and broadcast_mlp:
            ff_output = get_mlp_output(
                broadcast_range,
                timestep=int(org_timestep[0]),
                block_idx=self.block_idx,
                is_temporal=True,
            )
        else:
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                # norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = self.norm3(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
                
                norm_filtered_hidden_states = self.norm3(filtered_hidden_states)
                norm_filtered_hidden_states = norm_filtered_hidden_states * (1 + scale_mlp1) + shift_mlp1

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                    raise ValueError(
                        f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                    )

                num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                ff_output = torch.cat(
                    [
                        self.ff(hid_slice, scale=lora_scale)
                        for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                    ],
                    dim=self._chunk_dim,
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)
                ff_output_filtered = self.ff(norm_filtered_hidden_states, scale=lora_scale)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output
                ff_output_filtered = gate_mlp1 * ff_output_filtered

            if enable_pab() and broadcast_next:
                save_mlp_output(
                    timestep=int(org_timestep[0]),
                    block_idx=self.block_idx,
                    ff_output=ff_output,
                    is_temporal=True,
                )

        hidden_states = ff_output + hidden_states
        filtered_hidden_states = ff_output_filtered + filtered_hidden_states
        filtered_hidden_states = self.token_reuse(hidden_states, filtered_hidden_states, indices2, actual_indices, 'temporal')
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return filtered_hidden_states
        # return hidden_states
    
    
    def token_reuse(self, x_in, x_out, indices, actual_indices, mode):
        out = torch.zeros_like(x_in)
        out = out.reshape(-1, out.shape[-1])
        out.index_put_((indices,), x_out.squeeze())
        out = out.reshape(x_in.shape[0], x_in.shape[1], -1) 
        actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        if mode == 'spatial':
            out = out.permute(1, 0, 2)
            out = out.gather(1, actual_indices).permute(1, 0, 2)
        else:
            out = out.gather(1, actual_indices)
        return out

    def get_filtered_tensor(self, x, indices):
        x = x.reshape(-1, x.shape[-1])
        x = torch.index_select(x, 0, indices.squeeze())
        x = x.reshape(1, -1, x.shape[-1])
        return x
        
    def forward_org(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        org_timestep: Optional[torch.LongTensor] = None,
        all_timesteps=None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if enable_pab():
            broadcast_temporal, self.count = if_broadcast_temporal(int(org_timestep[0]), self.count)

        if enable_pab() and broadcast_temporal:
            attn_output = self.last_out
            assert self.use_ada_layer_norm_single
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
        else:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_single:  # go here
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                # norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            if self.parallel_manager.sp_size > 1:
                norm_hidden_states = self.dynamic_switch(norm_hidden_states, to_spatial_shard=True)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if self.parallel_manager.sp_size > 1:
                attn_output = self.dynamic_switch(attn_output, to_spatial_shard=False)

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            if enable_pab():
                self.last_out = attn_output

        hidden_states = attn_output + hidden_states

        if enable_pab():
            broadcast_mlp, self.mlp_count, broadcast_next, broadcast_range = if_broadcast_mlp(
                int(org_timestep[0]),
                self.mlp_count,
                self.block_idx,
                all_timesteps.tolist(),
                is_temporal=True,
            )

        if enable_pab() and broadcast_mlp:
            ff_output = get_mlp_output(
                broadcast_range,
                timestep=int(org_timestep[0]),
                block_idx=self.block_idx,
                is_temporal=True,
            )
        else:
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                # norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = self.norm3(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                    raise ValueError(
                        f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                    )

                num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                ff_output = torch.cat(
                    [
                        self.ff(hid_slice, scale=lora_scale)
                        for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                    ],
                    dim=self._chunk_dim,
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output

            if enable_pab() and broadcast_next:
                save_mlp_output(
                    timestep=int(org_timestep[0]),
                    block_idx=self.block_idx,
                    ff_output=ff_output,
                    is_temporal=True,
                )

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def dynamic_switch(self, x, to_spatial_shard: bool):
        if to_spatial_shard:
            scatter_dim, gather_dim = 0, 1
            scatter_pad = get_pad("spatial")
            gather_pad = get_pad("temporal")
        else:
            scatter_dim, gather_dim = 1, 0
            scatter_pad = get_pad("temporal")
            gather_pad = get_pad("spatial")
        x = all_to_all_with_pad(
            x,
            self.parallel_manager.sp_group,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            scatter_pad=scatter_pad,
            gather_pad=gather_pad,
        )
        return x


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        batch_size: int = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(
            timestep, batch_size=batch_size, hidden_dtype=hidden_dtype, resolution=None, aspect_ratio=None
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class LatteT2V(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        video_length: int = 16,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length

        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = linear_cls(in_channels, inner_dim)
            else:
                self.proj_in = conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size
            self.width = sample_size

            self.patch_size = patch_size
            interpolation_scale = self.config.sample_size // 64  # => 64 (= 512 pixart) has interpolation scale 1
            interpolation_scale = max(interpolation_scale, 1)
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    block_idx=d,
                )
                for d in range(num_layers)
            ]
        )

        # Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock_(  # one attention
                    inner_dim,
                    num_attention_heads,  # num_attention_heads
                    attention_head_dim,  # attention_head_dim 72
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=False,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    block_idx=d,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        elif self.is_input_patches and norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            self.use_additional_conditions = self.config.sample_size == 128  # False, 128 -> 1024
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False

        # define temporal positional embedding
        temp_pos_embed = self.get_1d_sincos_temp_embed(inner_dim, video_length)  # 1152 hidden size
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

        # parallel
        self.parallel_manager = None

    def enable_parallel(self, dp_size, sp_size, enable_cp):
        # update cfg parallel
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2
        else:
            cp_size = 1

        self.parallel_manager = ParallelManager(dp_size, cp_size, sp_size)

        for _, module in self.named_modules():
            if hasattr(module, "parallel_manager"):
                module.parallel_manager = self.parallel_manager

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        all_timesteps=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, frame, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # 0. Split batch for data parallelism
        if self.parallel_manager.cp_size > 1:
            (
                hidden_states,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs,
                class_labels,
                attention_mask,
                encoder_attention_mask,
            ) = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
                hidden_states,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs,
                class_labels,
                attention_mask,
                encoder_attention_mask,
            )

        input_batch_size, c, frame, h, w = hidden_states.shape
        
        # filter toekn percentage
        keep_idxs = self.batched_find_idxs_to_keep(hidden_states, threshold=0.5, tubelet_size=1, patch_size=2)
        print('------------------')
        total_tokens = keep_idxs.numel()
        filtered_tokens = (keep_idxs == 0).sum().item()
        filtered_percentage = 100.0 * filtered_tokens / total_tokens
        print('timestep:', timestep)
        print(f"Mask Filtering: {filtered_percentage:.2f}% tokens filtered")    
        
        # prepare mask and attn_bias
        attn_bias_dict = {}
        attn_bias_dict['self'] = {}
        attn_bias_dict['cross'] = {}
        attn_bias_dict['self']['spatial'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, h * w // 4)
        attn_bias_dict['self']['temporal'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, frame, mode='temporal')
        # attn_bias_dict['cross']['spatial'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, encoder_hidden_states.shape[0])
        # attn_bias_dict['cross']['temporal'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, encoder_hidden_states.shape[0], mode='temporal')
        
        mask_dict = {}
        mask_dict['mask'] = keep_idxs
        mask_dict['spatial'] = {}
        mask_dict['temporal'] = {}
        mask_dict = self.compute_mask_dict_spatial(mask_dict, h // 2, w // 2)
        mask_dict = self.compute_mask_dict_temporal(mask_dict, h // 2, w// 2)
        
        
            
        frame = frame - use_image_num
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()
        org_timestep = timestep

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = repeat(encoder_attention_mask, "b 1 l -> (b f) 1 l", f=frame).contiguous()
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(
                encoder_attention_mask_video, "b 1 l -> b (1 f) l", f=frame
            ).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)
            encoder_attention_mask = rearrange(encoder_attention_mask, "b n l -> (b n) l").contiguous().unsqueeze(1)

        # Retrieve lora scale.
        cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            num_patches = height * width

            hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # batch_size = hidden_states.shape[0]
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(
                    encoder_hidden_states_video, "b 1 t d -> b (1 f) t d", f=frame
                ).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
                encoder_hidden_states_spatial = rearrange(encoder_hidden_states, "b f t d -> (b f) t d").contiguous()
            else:
                encoder_hidden_states_spatial = repeat(
                    encoder_hidden_states, "b t d -> (b f) t d", f=frame
                ).contiguous()

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
        timestep_temp = repeat(timestep, "b d -> (b p) d", p=num_patches).contiguous()

        if self.parallel_manager.sp_size > 1:
            set_pad("temporal", frame + use_image_num, self.parallel_manager.sp_group)
            set_pad("spatial", num_patches, self.parallel_manager.sp_group)
            hidden_states = self.split_from_second_dim(hidden_states, input_batch_size)
            encoder_hidden_states_spatial = self.split_from_second_dim(encoder_hidden_states_spatial, input_batch_size)
            timestep_spatial = self.split_from_second_dim(timestep_spatial, input_batch_size)
            temp_pos_embed = split_sequence(
                self.temp_pos_embed, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal")
            )
        else:
            temp_pos_embed = self.temp_pos_embed
        attn_bias_dict['cross']['spatial'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, encoder_hidden_states.shape[1])
        attn_bias_dict['cross']['temporal'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, encoder_hidden_states.shape[1], mode='temporal')
        for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                    mask_dict=mask_dict,
                    attn_bias_dict=attn_bias_dict,
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                    if use_image_num != 0:  # image-video joitn training
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        if i == 0:
                            hidden_states_video = hidden_states_video + temp_pos_embed

                        hidden_states_video = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            use_reentrant=False,
                            mask_dict=mask_dict,
                            attn_bias_dict=attn_bias_dict,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

                    else:
                        if i == 0:
                            hidden_states = hidden_states + temp_pos_embed

                        hidden_states = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            use_reentrant=False,
                        )

                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()
            else: # this branch
                hidden_states = spatial_block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    None,
                    org_timestep,
                    all_timesteps=all_timesteps,
                    mask_dict=mask_dict,
                    attn_bias_dict=attn_bias_dict,
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                    if use_image_num != 0 and self.training:
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        hidden_states_video = temp_block(
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            org_timestep,
                            mask_dict=mask_dict,
                            attn_bias_dict=attn_bias_dict,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

                    else:
                        if i == 0 and frame > 1:
                            hidden_states = hidden_states + temp_pos_embed
                        hidden_states = temp_block(
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            org_timestep,
                            all_timesteps=all_timesteps,
                            mask_dict=mask_dict,
                            attn_bias_dict=attn_bias_dict,
                        )

                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

        if self.parallel_manager.sp_size > 1:
            hidden_states = self.gather_from_second_dim(hidden_states, input_batch_size)

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                embedded_timestep = repeat(embedded_timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            output = rearrange(output, "(b f) c h w -> b c f h w", b=input_batch_size).contiguous()

        # 3. Gather batch for data parallelism
        if self.parallel_manager.cp_size > 1:
            output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
    
    def create_block_diagonal_attention_mask(self, mask, kv_seqlen, mode='spatial'):
        """
        将 mask 和 kv_seqlen 转换为 BlockDiagonalMask, 用于高效的注意力计算。
        
        Args:
            mask (torch.Tensor): 输入的掩码，标记哪些 token 应该被忽略。
            kv_seqlen (torch.Tensor): 键/值的序列长度。
            heads (int): 注意力头的数量。

        Returns:
            BlockDiagonalPaddedKeysMask: 转换后的注意力掩码，用于高效的计算。
        """
        # 计算 q_seqlen: 通过 mask 来提取有效的查询 token 数量
        mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        if mode == 'spatial':
            mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)')
        else:
            mask = rearrange(mask, 'b 1 t h w -> (b h w) (t)')
        
        q_seqlen = mask.sum(dim=-1)  # 计算每个批次中有效的查询 token 数量
        q_seqlen = q_seqlen.tolist()
        
        kv_seqlen = [kv_seqlen] * len(q_seqlen)  # 重复 kv_seqlen 次

        # 生成 BlockDiagonalPaddedKeysMask
        attn_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen,  
            kv_seqlen=kv_seqlen,  # 键/值的序列长度
        )
        
        return attn_bias, q_seqlen, kv_seqlen    
    
    def compute_mask_dict_spatial(self, mask_dict, cur_h, cur_w):
        mask = mask_dict['mask']
        indices = []
        _mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        indices1 = torch.nonzero(_mask.reshape(1, -1).squeeze(0))
        _mask = rearrange(_mask, 'b 1 t h w -> (b t) (h w)')
        # for i in range(_mask.size(0)):
        #     index_per_batch = torch.where(_mask[i].bool())[0]
        #     indices.append(index_per_batch)
        mask_dict['spatial']['indices'] = indices
        mask_dict['spatial']['indices1'] = indices1
        mask_bool = _mask.bool()
        mask_bool = mask_bool.T
        device = mask.device
        batch_size, seq_len = mask_bool.shape
        # print('------------------')
        # time_stamp = time.time()
        arange_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # print('time for arange_indices:', time.time()-time_stamp)
        # time_stamp = time.time()
        nonzero_indices = torch.nonzero(mask_bool, as_tuple=True)
        valid_indices = torch.zeros_like(arange_indices)
        valid_indices[nonzero_indices[0], torch.cumsum(mask_bool.int(), dim=1)[mask_bool] - 1] = arange_indices[mask_bool]
        cumsum_mask = torch.cumsum(mask_bool.int(), dim=1)
        # print('time for cumsum_mask:', time.time()-time_stamp)
        # time_stamp = time.time()
        nearest_indices = torch.clip(cumsum_mask - 1, min=0)
        # print('time for nearest_indices:', time.time()-time_stamp)
        # time_stamp = time.time()
        actual_indices = valid_indices.gather(1, nearest_indices)
        mask_dict['spatial']['actual_indices'] = actual_indices
        return mask_dict
        
    def compute_mask_dict_temporal(self, mask_dict, cur_h, cur_w):
        mask = mask_dict['mask']
        indices = []
        _mask = torch.round(mask).to(torch.int)
        _mask = rearrange(_mask, 'b 1 t h w -> (b h w) (t)')
        indices1 = torch.nonzero(_mask.reshape(1, -1).squeeze(0))
        mask_dict['temporal']['indices'] = indices
        mask_dict['temporal']['indices1'] = indices1
        mask_bool = _mask.bool()
        device = mask.device
        batch_size, seq_len = mask_bool.shape
        arange_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        nonzero_indices = torch.nonzero(mask_bool, as_tuple=True)
        valid_indices = torch.zeros_like(arange_indices)
        valid_indices[nonzero_indices[0], torch.cumsum(mask_bool.int(), dim=1)[mask_bool] - 1] = arange_indices[mask_bool]
        cumsum_mask = torch.cumsum(mask_bool.int(), dim=1)
        nearest_indices = torch.clip(cumsum_mask - 1, min=0)
        actual_indices = valid_indices.gather(1, nearest_indices)
        mask_dict['temporal']['actual_indices'] = actual_indices
        return mask_dict    
    
    def batched_find_idxs_to_keep(self, 
                            x: torch.Tensor, 
                            threshold: int=2, 
                            tubelet_size: int=2,
                            patch_size: int=16) -> torch.Tensor:
        """
        Find the static tokens in a video tensor, and return a mask
        that selects tokens that are not repeated.

        Args:
        - x (torch.Tensor): A tensor of shape [B, C, T, H, W].
        - threshold (int): The mean intensity threshold for considering
                a token as static.
        - tubelet_size (int): The temporal length of a token.
        Returns:
        - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] 
            that selects tokens that are not repeated.

        """
        # Ensure input has the format [B, C, T, H, W]
        assert len(x.shape) == 5, "Input must be a 5D tensor"
        #ipdb.set_trace()
        # Convert to float32 if not already
        x = x.type(torch.float32)
        
        # Calculate differences between frames with a step of tubelet_size, ensuring batch dimension is preserved
        # Compare "front" of first token to "back" of second token
        diffs = x[:, :, (2*tubelet_size-1)::tubelet_size] - x[:, :, :-tubelet_size:tubelet_size]
        # Ensure nwe track negative movement.
        diffs = torch.abs(diffs)
        
        # Apply average pooling over spatial dimensions while keeping the batch dimension intact
        avg_pool_blocks = F.avg_pool3d(diffs, (1, patch_size, patch_size))
        # Compute the mean along the channel dimension, preserving the batch dimension
        avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
        # Create a dummy first frame for each item in the batch
        first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
        # first_frame = torch.zeros_like(avg_pool_blocks[:, :, 0:1])
        # Concatenate the dummy first frame with the rest of the frames, preserving the batch dimension
        avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2)
        # Determine indices to keep based on the threshold, ensuring the operation is applied across the batch
        # Update mask: 0 for high similarity, 1 for low similarity
        keep_idxs = avg_pool_blocks.squeeze(1) > threshold  
        keep_idxs = keep_idxs.unsqueeze(1)
        keep_idxs = keep_idxs.float()
        # Flatten out everything but the batch dimension
        # keep_idxs = keep_idxs.flatten(1)
        #ipdb.set_trace()
        return keep_idxs

    def compute_similarity_mask(self, latent, threshold=0.95):
        """
        Compute frame-wise similarity for latent and generate mask.

        Args:
        - latent (torch.Tensor): Latent tensor of shape [n, c, t, h, w].
        - threshold (float): Similarity threshold to determine whether to skip computation.

        Returns:
        - mask (torch.Tensor): Mask tensor of shape [n, 1, t, h, w],
        where mask = 0 means skip computation, mask = 1 means recompute.
        """
        n, c, t, h, w = latent.shape
        mask = torch.ones((n, 1, t, h, w), device=latent.device)  # Initialize mask with all 1s

        for frame_idx in range(1, t):  # Start from the second frame
            curr_frame = latent[:, :, frame_idx, :, :]  # Current frame [n, c, h, w]
            prev_frame = latent[:, :, frame_idx - 1, :, :]  # Previous frame [n, c, h, w]

            # Compute token-wise cosine similarity
            dot_product = (curr_frame * prev_frame).sum(dim=1, keepdim=True)  # [n, 1, h, w]
            norm_curr = curr_frame.norm(dim=1, keepdim=True)
            norm_prev = prev_frame.norm(dim=1, keepdim=True)
            similarity = dot_product / (norm_curr * norm_prev + 1e-8)  # Avoid division by zero

            # Update mask: 0 for high similarity, 1 for low similarity
            mask[:, :, frame_idx, :, :] = (similarity <= threshold).float()
        # mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        return mask
    
    def resize_mask(self, mask, target_h, target_w):
        """
        Resize the mask to match the new spatial dimensions of x.

        Args:
        - mask (torch.Tensor): Input mask of shape [b, 1, t, h, w].
        - target_h (int): Target height.
        - target_w (int): Target width.

        Returns:
        - resized_mask (torch.Tensor): Resized mask of shape [b, 1, t, target_h, target_w].
        """
        if mask is None:
            return mask
        batch, _, t, h, w = mask.shape

        if h == target_h and w == target_w:
            return mask  # No resizing needed

        # Reshape to [b * t, 1, h, w]
        mask = mask.view(batch * t, 1, h, w)

        # Resize to [b * t, 1, target_h, target_w]
        resized_mask = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Ensure the mask is binary (0 or 1)
        resized_mask = (resized_mask > 0.5).float()

        # Reshape back to [b, 1, t, target_h, target_w]
        resized_mask = resized_mask.view(batch, 1, t, target_h, target_w)

        return resized_mask
    def get_1d_sincos_temp_embed(self, embed_dim, length):
        pos = torch.arange(0, length).unsqueeze(1)
        return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

    def split_from_second_dim(self, x, batch_size):
        x = x.view(batch_size, -1, *x.shape[1:])
        x = split_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal"))
        x = x.reshape(-1, *x.shape[2:])
        return x

    def gather_from_second_dim(self, x, batch_size):
        x = x.view(batch_size, -1, *x.shape[1:])
        x = gather_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
        x = x.reshape(-1, *x.shape[2:])
        return x
