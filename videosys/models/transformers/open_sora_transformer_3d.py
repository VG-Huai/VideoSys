# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


from collections.abc import Iterable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from transformers import PretrainedConfig, PreTrainedModel

from videosys.core.dcp.recompute import auto_recompute
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
from videosys.models.modules.activations import approx_gelu
from videosys.models.modules.attentions import OpenSoraAttention, OpenSoraMultiHeadCrossAttention
from videosys.models.modules.embeddings import (
    OpenSoraCaptionEmbedder,
    OpenSoraPatchEmbed3D,
    OpenSoraPositionEmbedding2D,
    SizeEmbedder,
    TimestepEmbedder,
)
from videosys.utils.utils import batch_func
from xformers.ops.fmha.attn_bias import BlockDiagonalMask


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        block_idx=None,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = OpenSoraAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = OpenSoraMultiHeadCrossAttention(
            hidden_size, num_heads, enable_flash_attn=enable_flash_attn, temporal=temporal
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # parallel
        self.parallel_manager: ParallelManager = None

        self.grad_checkpointing = True

        # pab
        self.block_idx = block_idx
        self.attn_count = 0
        self.last_attn = None
        self.cross_count = 0
        self.last_cross = None
        self.mlp_count = 0

        self.inp_comm_time = 0
        self.oup_comm_time = 0

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        timestep=None,
        mask_dict = None,
        attn_bias_dict = None,
        all_timesteps=None,
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        if enable_pab():
            if self.temporal:
                broadcast_attn, self.attn_count = if_broadcast_temporal(int(timestep[0]), self.attn_count)
            else:
                broadcast_attn, self.attn_count = if_broadcast_spatial(int(timestep[0]), self.attn_count)

        if enable_pab() and broadcast_attn:
            x_m_s = self.last_attn
        else:
            # modulate (attention)
            normed1_x = self.norm1(x)
            x_m = t2i_modulate(normed1_x, shift_msa, scale_msa)
            if x_mask is not None:
                x_m_zero = t2i_modulate(normed1_x, shift_msa_zero, scale_msa_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # attention
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m, mask_dict=mask_dict, attn_bias_dict=attn_bias_dict, mode='temporal')
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                if self.parallel_manager.sp_size > 1:
                    is_image = T == 1
                    x_m, S, T = self.dynamic_switch(x_m, S, T, to_spatial_shard=False, is_image=is_image)

                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m, mask_dict=mask_dict, attn_bias_dict=attn_bias_dict, mode='spatial')
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
                if self.parallel_manager.sp_size > 1:
                    x_m, S, T = self.dynamic_switch(x_m, S, T, to_spatial_shard=True, is_image=is_image)

            # modulate (attention)
            x_m_s = gate_msa * x_m
            if x_mask is not None:
                x_m_s_zero = gate_msa_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            if enable_pab():
                self.last_attn = x_m_s

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        if enable_pab():
            broadcast_cross, self.cross_count = if_broadcast_cross(int(timestep[0]), self.cross_count)

        if enable_pab() and broadcast_cross:
            x = x + self.last_cross
        else:
            # x_cross = self.cross_attn(x, y, mask, mask_dict, attn_bias_dict)
            x_cross = self.cross_attn(x, y, mask)
            if enable_pab():
                self.last_cross = x_cross
            x = x + x_cross

        if enable_pab():
            broadcast_mlp, self.mlp_count, broadcast_next, skip_range = if_broadcast_mlp(
                int(timestep[0]),
                self.mlp_count,
                self.block_idx,
                all_timesteps,
                is_temporal=self.temporal,
            )

        if enable_pab() and broadcast_mlp:
            x_m_s = get_mlp_output(
                skip_range,
                timestep=int(timestep[0]),
                block_idx=self.block_idx,
                is_temporal=self.temporal,
            )
        else:
            # modulate (MLP)
            normed2_x = self.norm2(x)
            x_m = t2i_modulate(normed2_x, shift_mlp, scale_mlp)
            if x_mask is not None:
                x_m_zero = t2i_modulate(normed2_x, shift_mlp_zero, scale_mlp_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # MLP
            x_m = self.mlp(x_m)

            # modulate (MLP)
            x_m_s = gate_mlp * x_m
            if x_mask is not None:
                x_m_s_zero = gate_mlp_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            if enable_pab() and broadcast_next:
                save_mlp_output(
                    timestep=int(timestep[0]),
                    block_idx=self.block_idx,
                    ff_output=x_m_s,
                    is_temporal=self.temporal,
                )

        # residual
        x = x + self.drop_path(x_m_s)

        return x

    def dynamic_switch(self, x, s, t, to_spatial_shard: bool, is_image: bool = False):
        if to_spatial_shard:
            scatter_dim, gather_dim = 2, 1
            scatter_pad = get_pad("spatial")
            gather_pad = get_pad("temporal")
            if is_image:
                gather_dim = 0
                gather_pad = get_pad("batch")
        else:
            scatter_dim, gather_dim = 1, 2
            scatter_pad = get_pad("temporal")
            gather_pad = get_pad("spatial")
            if is_image:
                scatter_dim = 0
                scatter_pad = get_pad("batch")

        x = rearrange(x, "b (t s) d -> b t s d", t=t, s=s)
        x = all_to_all_with_pad(
            x,
            self.parallel_manager.sp_group,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            scatter_pad=scatter_pad,
            gather_pad=gather_pad,
        )
        new_s, new_t = x.shape[2], x.shape[1]
        x = rearrange(x, "b t s d -> b (t s) d")
        return x, new_s, new_t


class STDiT3Config(PretrainedConfig):
    model_type = "STDiT3"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        only_train_temporal=False,
        freeze_y_embedder=False,
        skip_y_embedder=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.only_train_temporal = only_train_temporal
        self.freeze_y_embedder = freeze_y_embedder
        self.skip_y_embedder = skip_y_embedder
        super().__init__(**kwargs)


class STDiT3(PreTrainedModel):
    config_class = STDiT3Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = OpenSoraPositionEmbedding2D(config.hidden_size)

        from rotary_embedding_torch import RotaryEmbedding

        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = OpenSoraPatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = OpenSoraCaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    block_idx=i,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                    block_idx=i,
                )
                for i in range(config.depth)
            ]
        )
        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()
        if config.only_train_temporal:
            for param in self.parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False

        # parallel
        self.parallel_manager: ParallelManager = None
        self.spatial_time = []
        self.temporal_time = []
        self.inp_time = []
        self.oup_time = []

    def enable_parallel(self, dp_size=None, sp_size=None, enable_cp=None, parallel_mgr=None):
        if parallel_mgr is not None:
            self.parallel_manager = parallel_mgr
        else:
            # update cfg parallel
            if enable_cp and sp_size % 2 == 0:
                sp_size = sp_size // 2
                cp_size = 2
            else:
                cp_size = 1

            self.parallel_manager = ParallelManager(dp_size, cp_size, sp_size)

        for name, module in self.named_modules():
            if "spatial_blocks" in name or "temporal_blocks" in name:
                if hasattr(module, "parallel_manager"):
                    module.parallel_manager = self.parallel_manager

    def enable_grad_checkpointing(self):
        for block in self.spatial_blocks:
            block.grad_checkpointing = True

        for block in self.temporal_blocks:
            block.grad_checkpointing = True

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def forward(
        self, x, timestep, y, all_timesteps=None, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs
    ):
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        # keep_idxs = self.compute_similarity_mask(x, threshold=0.95)
        # keep_idxs = None
        # keep_idxs = self.batched_find_idxs_to_keep(x, threshold=0.5, tubelet_size=1, patch_size=1)
        keep_idxs = self.batched_find_idxs_to_keep(x, threshold=0.7, tubelet_size=1, patch_size=2)
        # keep_idxs = self.batched_find_idxs_to_keep(x, threshold=1, tubelet_size=1, patch_size=2)
        print('------------------')
        total_tokens = keep_idxs.numel()
        filtered_tokens = (keep_idxs == 0).sum().item()
        filtered_percentage = 100.0 * filtered_tokens / total_tokens
        print('timestep:', timestep)
        print(f"Mask Filtering: {filtered_percentage:.2f}% tokens filtered")
        
        # === Split batch ===
        if self.parallel_manager.cp_size > 1:
            assert not self.training, "Batch split is not supported in training"
            set_pad("batch", x.shape[0], self.parallel_manager.cp_group)
            x, timestep, y, x_mask, fps = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0, pad=get_pad("batch")),
                x,
                timestep,
                y,
                x_mask,
                fps,
            )
            mask = split_sequence(mask, self.parallel_manager.cp_group, dim=0, pad=get_pad("batch"), pad_val=1)

        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        attn_bias_dict = {}
        attn_bias_dict['self'] = {}
        attn_bias_dict['cross'] = {}
        attn_bias_dict['self']['spatial'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, H * W)
        attn_bias_dict['self']['temporal'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, T, mode='temporal')
        attn_bias_dict['cross']['spatial'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, y_lens[0])
        attn_bias_dict['cross']['temporal'], _, _ = self.create_block_diagonal_attention_mask(keep_idxs, y_lens[0], mode='temporal')
        
        mask_dict = {}
        mask_dict['mask'] = keep_idxs
        mask_dict['spatial'] = {}
        mask_dict['temporal'] = {}
        mask_dict = self.compute_mask_dict_spatial(mask_dict, H, W)
        mask_dict = self.compute_mask_dict_temporal(mask_dict, H, W)
        
        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        if self.parallel_manager.sp_size > 1:
            set_pad("temporal", T, self.parallel_manager.sp_group)
            set_pad("spatial", S, self.parallel_manager.sp_group)
            set_pad("batch", x.shape[0], self.parallel_manager.sp_group)
            x = split_sequence(x, self.parallel_manager.sp_group, dim=2, grad_scale="down", pad=get_pad("spatial"))
            T, S = x.shape[1], x.shape[2]

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        valid_depth = kwargs.get("valid_depth", self.depth)
        for depth in range(valid_depth):
            spatial_block = self.spatial_blocks[depth]
            temporal_block = self.temporal_blocks[depth]
            x = auto_recompute(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, timestep, mask_dict, attn_bias_dict)
            x = auto_recompute(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, timestep, mask_dict, attn_bias_dict)

        if self.parallel_manager.sp_size > 1:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_sequence(x, self.parallel_manager.sp_group, dim=2, grad_scale="up", pad=get_pad("spatial"))
            T, S = x.shape[1], x.shape[2]
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # === Gather Output ===
        if self.parallel_manager.cp_size > 1:
            x = gather_sequence(x, self.parallel_manager.cp_group, dim=0, pad=get_pad("batch"))

        # cast to float32 for better accuracy
        x = x.to(torch.float32)

        return x
    
    def org_forward(
        self, x, timestep, y, all_timesteps=None, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs
    ):
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        # keep_idxs = self.compute_similarity_mask(x, threshold=0.95)
        # keep_idxs = None
        keep_idxs = self.batched_find_idxs_to_keep(x, threshold=0.5, tubelet_size=1, patch_size=1)
        # keep_idxs = self.batched_find_idxs_to_keep(x, threshold=0.3, tubelet_size=1, patch_size=1)
        print('------------------')
        total_tokens = keep_idxs.numel()
        filtered_tokens = (keep_idxs == 0).sum().item()
        filtered_percentage = 100.0 * filtered_tokens / total_tokens
        print('timestep:', timestep)
        print(f"Mask Filtering: {filtered_percentage:.2f}% tokens filtered")
        # === Split batch ===
        if self.parallel_manager.cp_size > 1:
            assert not self.training, "Batch split is not supported in training"
            set_pad("batch", x.shape[0], self.parallel_manager.cp_group)
            x, timestep, y, x_mask, fps = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0, pad=get_pad("batch")),
                x,
                timestep,
                y,
                x_mask,
                fps,
            )
            mask = split_sequence(mask, self.parallel_manager.cp_group, dim=0, pad=get_pad("batch"), pad_val=1)

        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        if self.parallel_manager.sp_size > 1:
            set_pad("temporal", T, self.parallel_manager.sp_group)
            set_pad("spatial", S, self.parallel_manager.sp_group)
            set_pad("batch", x.shape[0], self.parallel_manager.sp_group)
            x = split_sequence(x, self.parallel_manager.sp_group, dim=2, grad_scale="down", pad=get_pad("spatial"))
            T, S = x.shape[1], x.shape[2]

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        valid_depth = kwargs.get("valid_depth", self.depth)
        for depth in range(valid_depth):
            spatial_block = self.spatial_blocks[depth]
            temporal_block = self.temporal_blocks[depth]
            x = auto_recompute(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, timestep)
            x = auto_recompute(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, timestep)

        if self.parallel_manager.sp_size > 1:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_sequence(x, self.parallel_manager.sp_group, dim=2, grad_scale="up", pad=get_pad("spatial"))
            T, S = x.shape[1], x.shape[2]
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # === Gather Output ===
        if self.parallel_manager.cp_size > 1:
            x = gather_sequence(x, self.parallel_manager.cp_group, dim=0, pad=get_pad("batch"))

        # cast to float32 for better accuracy
        x = x.to(torch.float32)

        return x
    
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

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x
    
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
    


def STDiT3_XL_2(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        model = STDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = STDiT3(config)
    return model
