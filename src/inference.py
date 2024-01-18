""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference_kernels.kernel_selector import forward_pass_quantized_linear
from src.utils import _dequantize_weight, ellipsis, get_int_dtype, unpack_int_data


class FinalizedQuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        num_out_groups = out_features // out_group_size
        num_in_groups = in_features // in_group_size
        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = 2**nbits_per_codebook

        # CODES & CODEBOOKS
        self.codebooks = nn.Parameter(
            torch.empty((num_codebooks, self.codebook_size, out_group_size, in_group_size), **factory_kwargs),
            requires_grad=True,
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes = nn.Parameter(
            torch.empty(
                (num_out_groups, num_in_groups, num_codebooks), device=device, dtype=get_int_dtype(nbits_per_codebook)
            ),
            requires_grad=False,
        )  #  [num_out_groups, num_in_groups, num_codebooks]

        # SCALES
        self.scales = nn.Parameter(
            torch.empty((num_out_groups, 1, 1, 1), **factory_kwargs), requires_grad=True
        )  #  [num_out_groups, num_in_groups, 1, 1] if scale_nbits > 0 else [num_out_groups, 1, 1, 1]

        # BIAS
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return forward_pass_quantized_linear(input, self.codes, self.codebooks, self.scales, self.bias)
