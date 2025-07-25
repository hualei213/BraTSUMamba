# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat


try:
    from unet.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn_no_out_proj = None

try:
    from unet.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from unet.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        nslices_small=64,
        nslices_big=16
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.nslices_small = nslices_small
        self.nslices_big = nslices_big

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        assert bimamba_type == "v3"

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        # assert bimamba_type == "v3"
        # spatial
        A_s = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_log = torch.log(A_s)  # Keep A_b_log in fp32
        self.A_s_log = nn.Parameter(A_s_log)
        self.A_s_log._no_weight_decay = True 

        self.conv1d_s = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_s = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_s._no_weight_decay = True

        #切片间反向特征交互
        A_s_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_b_log = torch.log(A_s_b)  # Keep A_s_b_log in fp32
        self.A_s_b_log = nn.Parameter(A_s_b_log)
        self.A_s_b_log._no_weight_decay = True

        self.conv1d_s_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_s_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_s_b._no_weight_decay = True


        #big slice
        A_s_big = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_big_log = torch.log(A_s_big)  # Keep A_b_log in fp32
        self.A_s_big_log = nn.Parameter(A_s_big_log)
        self.A_s_big_log._no_weight_decay = True

        self.conv1d_s_big = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s_big = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s_big = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_s_big = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_s_big._no_weight_decay = True

        # 大切片间反向特征交互
        A_s_big_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_big_b_log = torch.log(A_s_big_b)  # Keep A_s_b_log in fp32
        self.A_s_big_b_log = nn.Parameter(A_s_big_b_log)
        self.A_s_big_b_log._no_weight_decay = True

        self.conv1d_s_big_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s_big_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s_big_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_s_big_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_s_big_b._no_weight_decay = True


        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v3":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )


                #small slice
                A_s = -torch.exp(self.A_s_log.float())

                xz_s = xz.chunk(self.nslices_small, dim=-1)
                xz_s = torch.stack(xz_s,dim=-1)
                xz_s = xz_s.flatten(-2)
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
                out_s = out_s.reshape(batch,self.d_inner,seqlen//self.nslices_small,self.nslices_small).permute(0,1,3,2).flatten(-2)

                #小切片间的反向特征交互
                A_s_b = -torch.exp(self.A_s_b_log.float())
                # xz_s_b = xz.chunk(self.nslices, dim=-1)
                #
                # # xz_s_b = [slice.flip([-1]) for slice in xz_s_b]
                #
                # xz_s_b = torch.stack(xz_s_b, dim=-1)
                # xz_s_b = xz_s_b.flatten(-2)
                xz_s_b = xz_s.flip([-1])
                out_s_b = mamba_inner_fn_no_out_proj(
                    xz_s_b,
                    self.conv1d_s_b.weight,
                    self.conv1d_s_b.bias,
                    self.x_proj_s_b.weight,
                    self.dt_proj_s_b.weight,
                    A_s_b,
                    None,
                    None,
                    self.D_s_b.float(),
                    delta_bias=self.dt_proj_s_b.bias.float(),
                    delta_softplus=True,
                )
                out_s_b = out_s_b.reshape(batch, self.d_inner, seqlen // self.nslices_small, self.nslices_small).permute(0, 1, 3, 2).flatten(-2)


                # big slice
                A_s_big = -torch.exp(self.A_s_big_log.float())

                xz_s_big = xz.chunk(self.nslices_big, dim=-1)
                xz_s_big = torch.stack(xz_s_big, dim=-1)
                xz_s_big = xz_s_big.flatten(-2)
                out_s_big = mamba_inner_fn_no_out_proj(
                    xz_s_big,
                    self.conv1d_s_big.weight,
                    self.conv1d_s_big.bias,
                    self.x_proj_s_big.weight,
                    self.dt_proj_s_big.weight,
                    A_s_big,
                    None,
                    None,
                    self.D_s_big.float(),
                    delta_bias=self.dt_proj_s_big.bias.float(),
                    delta_softplus=True,
                )
                out_s_big = out_s_big.reshape(batch, self.d_inner, seqlen // self.nslices_big, self.nslices_big).permute(0, 1, 3, 2).flatten(-2)

                # 大切片间的反向特征交互
                A_s_big_b = -torch.exp(self.A_s_big_b_log.float())
                # xz_s_b = xz.chunk(self.nslices, dim=-1)
                #
                # # xz_s_b = [slice.flip([-1]) for slice in xz_s_b]
                #
                # xz_s_b = torch.stack(xz_s_b, dim=-1)
                # xz_s_b = xz_s_b.flatten(-2)
                xz_s_big_b = xz_s_big.flip([-1])
                out_s_big_b = mamba_inner_fn_no_out_proj(
                    xz_s_big_b,
                    self.conv1d_s_big_b.weight,
                    self.conv1d_s_big_b.bias,
                    self.x_proj_s_big_b.weight,
                    self.dt_proj_s_big_b.weight,
                    A_s_big_b,
                    None,
                    None,
                    self.D_s_big_b.float(),
                    delta_bias=self.dt_proj_s_big_b.bias.float(),
                    delta_softplus=True,
                )
                out_s_big_b = out_s_big_b.reshape(batch, self.d_inner, seqlen // self.nslices_big,
                                          self.nslices_big).permute(0, 1, 3, 2).flatten(-2)


                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                # out = F.linear(rearrange(out + out_b.flip([-1]) + out_s+out_s_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                out = F.linear(rearrange(out + out_b.flip([-1]) + out_s+out_s_b.flip([-1])+out_s_big + out_s_big_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

        return out


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
