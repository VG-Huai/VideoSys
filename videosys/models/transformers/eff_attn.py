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
from videosys.utils.utils import batch_func
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import xformers
from typing import Callable, List, Optional
from einops import rearrange
from diffusers.utils import deprecate, logging

class EfficientAttention(Attention):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs
    ):
        # 你可以在这里打印输入的 shape，方便 debug
        # print(f"Custom Efficient Attention - hidden_states shape: {hidden_states.shape}")

        # 执行原始 Attention 逻辑
        output = super().forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs
        )

        # # 在输出后添加一些自定义操作
        # output = output * 1.1  # 例如：增强注意力的影响力

        return output
    
class XFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        mask_dict: Optional[dict] = None,
        attn_bias_dict: Optional[dict] = None,
        mode: Optional[str] = "spatial",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        heads_num = attn.heads
        
        B, M, C = hidden_states.shape
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        # eff mode
        if mask_dict is not None:
            # indices = mask['indices']
            indices1 = mask_dict[mode]['indices1']
            indices2 = indices1.squeeze()
            actual_indices = mask_dict[mode]['actual_indices']
            mask = mask_dict['mask']
        if mask is not None:
            # time_stamp = time.time()
            mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
            mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)') 
        if encoder_hidden_states is not None:
            attention_mask = attn_bias_dict['cross'][mode]
        else:
            attention_mask = attn_bias_dict['self'][mode]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # query = query.view(B, M, heads_num, C // heads_num)
        # key = key.view(B, key_tokens, heads_num, C // heads_num)
        # value = value.view(B, key_tokens, heads_num, C // heads_num)
        
        cat_q = query.reshape(-1, C)
        cat_q = torch.index_select(cat_q, 0, indices1.squeeze()) 
        cat_q = cat_q.reshape(1, -1, heads_num, C // heads_num)
        
        cat_k = key.reshape(1, -1, heads_num, C // heads_num)
        cat_v = value.reshape(1, -1, heads_num, C // heads_num)
        
        out_with_bias = xformers.ops.memory_efficient_attention(
            cat_q, cat_k, cat_v, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        out_with_bias = out_with_bias.reshape(1, out_with_bias.shape[1], C) # B M H K ---> B M HK
        out_with_bias = out_with_bias.to(query.dtype)
        out = torch.zeros_like(hidden_states)
        out = out.reshape(-1, C)
        out.index_put_((indices2,), out_with_bias.squeeze())
        
        # token reuse, to be 
        if mask is not None:
            actual_indices1 = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
            if mode == 'spatial':
                # x: (B T) S C
                out = out.reshape(mask.shape[0], mask.shape[1], -1)  # b * t, h * w, c
                out = out.permute(1, 0, 2)
                out = out.gather(1, actual_indices1).permute(1, 0, 2)

            else:
                # x: (B S) T C
                out = out.reshape(B, M, C)  # b * t, h * w, c
                out = out.gather(1, actual_indices1)

        hidden_states = out
        hidden_states = hidden_states.reshape(B, M, C)
        
        assert (hidden_states != self.token_reuse(hidden_states, out_with_bias, indices2, actual_indices, mode)).sum() == 0

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # here is a demo for linear proj  mismatch
        # out1 = out.permute(1, 0, 2)
        # aa = attn.to_out[0](out)
        # bb = attn.to_out[0](out1).permute(1, 0, 2)
        # cc = attn.to_out[0](out_with_bias)
        # cc = self.token_reuse(hidden_states, cc, indices2, actual_indices, mode)
        # (aa != bb).sum()
        # tensor(2094130, device='cuda:0')
        # (aa != cc).sum()
        # tensor(2094130, device='cuda:0')
        # (bb != cc).sum()
        # tensor(0, device='cuda:0')
        # torch.allclose(aa.reshape_as(bb), bb, atol=1e-4, rtol=1e-3)
        # True
        filtered_hidden_states = attn.to_out[0](out_with_bias)
        filtered_hidden_states = attn.to_out[1](filtered_hidden_states)
        # filtered_hidden_states = self.token_reuse(hidden_states, filtered_hidden_states, indices2, actual_indices, mode)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        filtered_hidden_states = filtered_hidden_states / attn.rescale_output_factor

        return hidden_states, filtered_hidden_states
    
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
    def max_diff(self, A, B, name_A, name_B):
        diff = (A - B).abs().max().item()
        print(f"Max error between {name_A} and {name_B}: {diff:.6f}")