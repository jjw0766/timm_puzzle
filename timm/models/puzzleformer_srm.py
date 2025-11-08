import copy
import logging
import math
import os
import numpy as np
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

from timm.models.naflexvit import NaFlexVit
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.data import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
)
from timm.layers import (
    AttentionPoolLatent,
    Attention,
    PatchEmbed,
    Mlp,
    SwiGLUPacked,
    SwiGLU,
    LayerNorm,
    RmsNorm,
    DropPath,
    calculate_drop_path_rates,
    PatchDropout,
    trunc_normal_,
    lecun_normal_,
    resample_patch_embed,
    resample_abs_pos_embed,
    get_act_layer,
    get_norm_layer,
    maybe_add_mask,
    LayerType,
    LayerScale,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply, checkpoint, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['PuzzleTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)

def get_adjacency_matrix(n:int, shuffle_order:List[int], rotate_order:List[int]): # 패치에 대하여 연결된 패치 찾기
    label_map_dict = {
        (0,0):1, # i 패치의 down과 j 패치의 down 연결되어 있을 때
        (0,1):2, # i 패치의 down과 j 패치의 left 연결되어 있을 때
        (0,2):3, # i 패치의 down과 j 패치의 up 연결되어 있을 때
        (0,3):4, # i 패치의 down과 j 패치의 right 연결되어 있을 때
        (1,0):5, # i 패치의 left과 j 패치의 down 연결되어 있을 때
        (1,1):6, # i 패치의 left과 j 패치의 left 연결되어 있을 때
        (1,2):7, # i 패치의 left과 j 패치의 up 연결되어 있을 때
        (1,3):8, # i 패치의 left과 j 패치의 right 연결되어 있을 때
        (2,0):9, # i 패치의 up과 j 패치의 down 연결되어 있을 때
        (2,1):10, # i 패치의 up과 j 패치의 left 연결되어 있을 때
        (2,2):11, # i 패치의 up과 j 패치의 up 연결되어 있을 때
        (2,3):12, # i 패치의 up과 j 패치의 right 연결되어 있을 때
        (3,0):13, # i 패치의 right과 j 패치의 down 연결되어 있을 때
        (3,1):14, # i 패치의 right과 j 패치의 left 연결되어 있을 때
        (3,2):15, # i 패치의 right과 j 패치의 up 연결되어 있을 때
        (3,3):16 # i 패치의 right과 j 패치의 right 연결되어 있을 때
    }
    sqrt_n = int(n**0.5)
    shuffle_order_matrix = [shuffle_order[sqrt_n*i:sqrt_n*(i+1)]for i in range(sqrt_n)]
    adj_matrix = np.zeros((n,n), dtype=int)
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            o = shuffle_order_matrix[i][j]
            i_o, j_o = divmod(o,sqrt_n)
            for i_add,j_add in [(-1,0), (1,0), (0,1), (0,-1)]:
                i_compare, j_compare = i_o+i_add, j_o+j_add
                if i_compare<0 or i_compare>=sqrt_n or j_compare<0 or j_compare>=sqrt_n : continue
                i_, j_ = i*sqrt_n+j, shuffle_order.index(i_compare*sqrt_n+j_compare)
                i_rotate, j_rotate = rotate_order[i_], rotate_order[j_]
                if (i_add,j_add) == (-1,0):
                    adj_matrix[i_][j_] = label_map_dict[((i_rotate+2)%4,j_rotate)] # 상
                    adj_matrix[j_][i_] = label_map_dict[(j_rotate, (i_rotate+2)%4)] # 하
                elif (i_add,j_add) == (1,0):
                    adj_matrix[i_][j_] = label_map_dict[(i_rotate, (j_rotate+2)%4)] # 하
                    adj_matrix[j_][i_] = label_map_dict[((j_rotate+2)%4, i_rotate)] # 상
                elif  (i_add,j_add) == (0,-1):
                    adj_matrix[i_][j_] = label_map_dict[((i_rotate+1)%4, (j_rotate+3)%4)] # 좌
                    adj_matrix[j_][i_] = label_map_dict[((j_rotate+3)%4, (i_rotate+1)%4)] # 우
                elif (i_add,j_add) == (0,1):
                    adj_matrix[i_][j_] = label_map_dict[((i_rotate+3)%4, (j_rotate+1)%4)] # 우
                    adj_matrix[j_][i_] = label_map_dict[((j_rotate+1)%4, (i_rotate+3)%4)] # 좌
    return adj_matrix

def connect_to_piece_type(connect):
    piece_types = []
    for connect_row in connect:
        n_connect = np.bincount(connect_row)[1:].sum()
        if n_connect == 2:
            piece_types.append(1)
        elif n_connect == 3:
            piece_types.append(2)
        elif n_connect == 4:
            piece_types.append(3)
        else:
            piece_types.append(0)
    return piece_types


class ConnectClassifier(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            kernel_size: int = 1,
            num_classes: int = 17,
            has_class_token: bool = True,
            qk_norm: bool = False,
            scale_norm: bool = False,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_class_token = has_class_token
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.k = int(kernel_size**0.5)
        self.num_classes = num_classes

        self.query = nn.Linear(dim, dim * num_classes, **dd)
        self.key = nn.Linear(dim, dim * num_classes, **dd)
        # self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        # self.conv = nn.AvgPool2d(kernel_size, kernel_size)
        self.conv = nn.Conv2d(self.num_heads * self.num_classes, self.num_heads * self.num_classes, kernel_size, kernel_size)
        self.clf = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.num_heads * num_classes, num_classes),
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_class_token:
            x = x[:, 1:, :]
        B, N, C = x.shape
        n = int(N**0.5)
        x = x.reshape(B, n, n, C)
        x = x.reshape(B, n//self.k, self.k, n//self.k, self.k, C).permute(0,1,3,2,4,5).reshape(B, (n//self.k)*(n//self.k), self.k*self.k, C)
        x = x.reshape(B, N, C)
        q = self.query(x).reshape(B, N, self.num_heads * self.num_classes, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads * self.num_classes, self.head_dim).permute(0, 2, 1, 3)
        # q, k = self.q_norm(q), self.k_norm(k)
        # q = q * self.scale

        attn = q @ k.transpose(-2, -1) # B, num_heads, N, N
        x = self.conv(attn)
        x = x.permute(0,2,3,1)
        x = self.clf(x)
        return x
    
class PieceClassifier(nn.Module):
    def __init__(
            self,
            dim: int,
            kernel_size: int = 1,
            num_classes: int = 256,
            has_class_token: bool = True,
            device=None,
            dtype=None
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.has_class_token = has_class_token

        self.conv = nn.Conv2d(dim, dim, kernel_size, kernel_size)
        self.clf = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_class_token:
            x = x[:, 1:, :]
        B, N, C = x.shape
        n = int(N**0.5)
        x = x.transpose(1,2)
        x = x.reshape(B, C, n, n)
        x = self.conv(x)
        x = x.flatten(2).transpose(1,2)
        x = self.clf(x)
        return x

class PieceDecoder(nn.Module):
    def __init__(
            self,
            dim: int,
            grid_size: int,
            kernel_size: int = 1,
            has_class_token: bool = True,
            device=None,
            dtype=None
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.has_class_token = has_class_token
        self.grid_size = grid_size
        self.conv = nn.Conv2d(dim, self.grid_size * self.grid_size * 3, kernel_size, kernel_size)


    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_class_token:
            x = x[:, 1:, :]
        B, N, C = x.shape
        n = int(N**0.5)
        x = x.transpose(1,2)
        x = x.reshape(B, C, n, n)
        x = self.conv(x)
        x = x.reshape(B, 3, self.grid_size, self.grid_size, n, n).permute(0,1,4,2,5,3).reshape(B, 3, n*self.grid_size, n*self.grid_size)
        return x



class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        self.norm1 = norm_layer(dim, **dd)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            **dd
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            **dd,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x


class PuzzleTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 256,
            patch_size: Union[int, Tuple[int, int]] = 16,
            piece_sizes: Union[List[int], Tuple[int, ...]] = (16, 32, 64, 128),
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            pool_include_prefix: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.img_size = img_size
        self.patch_size = patch_size
        self.piece_sizes = piece_sizes
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.grad_checkpointing = False

        embed_args = {}

        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
            **dd,
        )
        self.piece_pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim, **dd) * .02)

        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim, **dd)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim, **dd) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim, **dd) if pre_norm else nn.Identity()

        dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_attn_norm=scale_attn_norm,
                scale_mlp_norm=scale_mlp_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                **dd,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim, **dd) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim, **dd) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        self.connect_classifiers = nn.ModuleDict()
        self.shuffle_classifiers = nn.ModuleDict()
        self.rotate_classifiers = nn.ModuleDict()
        self.piece_decoders = nn.ModuleDict()
        for piece_size in piece_sizes:
            # kernel_size = (piece_size // patch_size) ** 2
            self.connect_classifiers[str(piece_size)] = ConnectClassifier(
                dim=embed_dim,
                num_heads=num_heads,
                kernel_size=(piece_size // patch_size) ** 2,
                num_classes=17,
                has_class_token=self.has_class_token,
                norm_layer=norm_layer,
                **dd,
            )
            self.shuffle_classifiers[str(piece_size)] = PieceClassifier(
                dim=embed_dim,
                kernel_size=(piece_size // patch_size),
                num_classes=(img_size // piece_size) ** 2,
                has_class_token=self.has_class_token,
                **dd,
            )
            self.rotate_classifiers[str(piece_size)] = PieceClassifier(
                dim=embed_dim,
                kernel_size=(piece_size // patch_size),
                num_classes=4,
                has_class_token=self.has_class_token,
                **dd,
            )
            self.piece_decoders[str(piece_size)] = PieceDecoder(
                dim=embed_dim,
                grid_size=piece_size,
                kernel_size=embed_dim,
                has_class_token=self.has_class_token,
                **dd,
            )

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self) -> None:
        """Apply weight initialization fix (scaling w/ layer index)."""
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        """Initialize model weights.

        Args:
            mode: Weight initialization mode ('jax', 'jax_nlhb', 'moco', or '').
        """
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        if self.piece_pos_embed is not None:
            trunc_normal_(self.piece_pos_embed, std=.02)

        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for a single module (compatibility method)."""
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        """Load pretrained weights.

        Args:
            checkpoint_path: Path to checkpoint.
            prefix: Prefix for state dict keys.
        """
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Set of parameters that should not use weight decay."""
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Union[str, List]]:
        """Create regex patterns for parameter grouping.

        Args:
            coarse: Use coarse grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, 'set_grad_checkpointing'):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the input image resolution and patch size.

        Args:
            img_size: New input resolution, if None current resolution is used.
            patch_size: New patch size, if None existing patch size is used.
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=self.patch_embed.grid_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    verbose=True,
                ))

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)
    
    def _piece_pos_embed(self, x: torch.Tensor, piece_size: int) -> torch.Tensor:
        """Apply piece-wise positional embedding to input."""
        B, N, D = x.shape
        n = int(math.sqrt(N))
        k = piece_size // self.patch_size
        x = x.reshape(B, n, n, D)
        piece_pos_embed = self.piece_pos_embed.reshape(1, n//k, k, n//k, k, D).permute(0,1,3,2,4,5).reshape(1, n//k, n//k, k*k, D).mean(dim=3) # 1, n//k, n//k, D
        piece_pos_embed = piece_pos_embed.repeat_interleave(k, dim=1).repeat_interleave(k, dim=2) # 1, n, n, D
        x = x + piece_pos_embed
        x = x.reshape(B, N, D)
        return x

    
    def forward_features_pretrain(self, x: torch.Tensor, piece_size: int = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        if piece_size is not None:
            x = self._piece_pos_embed(x, piece_size)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        return x
    
    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        for piece_size in self.piece_sizes:
            x = self._piece_pos_embed(x, piece_size)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        """Apply pooling to feature tokens.

        Args:
            x: Feature tensor.
            pool_type: Pooling type override.

        Returns:
            Pooled features.
        """
        if self.attn_pool is not None:
            if not self.pool_include_prefix:
                x = x[:, self.num_prefix_tokens:]
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier.

        Returns:
            Output tensor.
        """
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x

    def forward_pretrain(self, x: torch.Tensor, piece_size: int = None, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x = self.forward_features_pretrain(x, piece_size, attn_mask=attn_mask)
        logits_connect = self.connect_classifiers[str(piece_size)](x)
        logits_shuffle = self.shuffle_classifiers[str(piece_size)](x)
        logits_rotate = self.rotate_classifiers[str(piece_size)](x)
        decoded = self.piece_decoders[str(piece_size)](x)
        connects_pred = logits_connect.argmax(dim=-1).detach().cpu().numpy()
        shuffle_pred = logits_shuffle.argmax(dim=-1).detach().cpu().numpy()
        rotate_pred = logits_rotate.argmax(dim=-1).detach().cpu().numpy()
        return {
            'logits_connects': logits_connect.detach(),
            'logits_shuffle': logits_shuffle,
            'logits_rotate': logits_rotate,
            'decoded': decoded,
            'connects_pred': connects_pred,
            'shuffle_pred': shuffle_pred,
            'rotate_pred': rotate_pred,
        }



def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ViT weight initialization, original timm impl (for reproducibility).

    Args:
        module: Module to initialize.
        name: Module name for context.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()





def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    if 'jax' in mode:
        raise ValueError(f'{mode} not supported')
    elif 'moco' in mode:
        raise ValueError(f'{mode} not supported')
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb, gs_new, gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )




def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: PuzzleTransformer,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if "encoder" in state_dict:
        # IJEPA, vit in an 'encoder' submodule
        state_dict = state_dict['encoder']
        prefix = 'module.'
    elif 'visual.trunk.pos_embed' in state_dict or 'visual.trunk.blocks.0.norm1.weight' in state_dict:
        # OpenCLIP model with timm vision encoder
        prefix = 'visual.trunk.'
        if 'visual.head.proj.weight' in state_dict and isinstance(model.head, nn.Linear):
            # remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
            out_dict['head.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
    elif 'module.visual.trunk.pos_embed' in state_dict:
        prefix = 'module.visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict

@torch.no_grad()
def _load_weights(model: PuzzleTransformer, checkpoint_path: str, prefix: str = '', load_bfloat16: bool = False) -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np
    if load_bfloat16:
        try:
            import jax.numpy as jnp
            import ml_dtypes
        except ImportError as e:
            raise ValueError(e)
        
    def _n2p(_w, t=True, idx=None):
        if idx is not None:
            _w = _w[idx]

        if load_bfloat16:
            _w = _w.view(ml_dtypes.bfloat16).astype(jnp.float32)
            _w = np.array(_w)

        if _w.ndim == 4 and _w.shape[0] == _w.shape[1] == _w.shape[2] == 1:
            _w = _w.flatten()
        if t:
            if _w.ndim == 4:
                _w = _w.transpose([3, 2, 0, 1])
            elif _w.ndim == 3:
                _w = _w.transpose([2, 0, 1])
            elif _w.ndim == 2:
                _w = _w.transpose([1, 0])

        _w = torch.from_numpy(_w)
        return _w

    if load_bfloat16:
        w = jnp.load(checkpoint_path)
    else:
        w = np.load(checkpoint_path)

    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        if f'{prefix}Transformer/encoderblock/LayerNorm_0/scale' in w:
            block_prefix = f'{prefix}Transformer/encoderblock/'
            idx = i
        else:
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            idx = None
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'], idx=idx))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'], idx=idx))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'], idx=idx))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'], idx=idx))
            
def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs,
    }

default_cfgs = {

}

_quick_gelu_cfgs = [n for n, c in default_cfgs.items() if c.get('notes', ()) and 'quickgelu' in c['notes'][0]]
for n in _quick_gelu_cfgs:
    # generate quickgelu default cfgs based on contents of notes field
    c = copy.deepcopy(default_cfgs[n])
    if c['hf_hub_id'] == 'timm/':
        c['hf_hub_id'] = 'timm/' + n  # need to use non-quickgelu model name for hub id
    default_cfgs[n.replace('_clip_', '_clip_quickgelu_')] = c
default_cfgs = generate_default_cfgs(default_cfgs)


# Global flag to use NaFlexVit instead of PuzzleTransformer
_USE_NAFLEX_DEFAULT = os.environ.get('TIMM_USE_NAFLEXVIT', 'false').lower() == 'true'

def _create_vision_transformer(
        variant: str,
        pretrained: bool = False,
        use_naflex: Optional[bool] = None,
        **kwargs,
) -> Union[PuzzleTransformer, NaFlexVit]:
    # Check if we should use NaFlexVit instead
    if use_naflex is None:
        use_naflex = _USE_NAFLEX_DEFAULT
    if use_naflex:
        # Import here to avoid circular imports
        from .naflexvit import _create_naflexvit_from_classic
        return _create_naflexvit_from_classic(variant, pretrained, **kwargs)

    out_indices = kwargs.pop('out_indices', 3)
    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = kwargs.pop('pretrained_strict', True)
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        PuzzleTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )

@register_model
def puzzleformer_srm_tiny_patch16_256(pretrained: bool = False, **kwargs) -> PuzzleTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, piece_sizes=(16, 32, 64, 128), embed_dim=192, depth=12, num_heads=3, class_token=True)
    model = _create_vision_transformer('puzzleformer_srm_tiny_patch16_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def puzzleformer_srm_small_patch16_256(pretrained: bool = False, **kwargs) -> PuzzleTransformer:
    """ ViT-Small (Vit-S/16)
    """
    model_args = dict(patch_size=16, piece_sizes=(16, 32, 64, 128), embed_dim=384, depth=12, num_heads=6, class_token=True)
    model = _create_vision_transformer('puzzleformer_srm_small_patch16_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def puzzleformer_srm_base_patch16_256(pretrained: bool = False, **kwargs) -> PuzzleTransformer:
    """ ViT-Base (Vit-B/16)
    """
    model_args = dict(patch_size=16, piece_sizes=(16, 32, 64, 128), embed_dim=768, depth=12, num_heads=12, class_token=True)
    model = _create_vision_transformer('puzzleformer_srm_base_patch16_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
