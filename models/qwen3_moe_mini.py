# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    num_experts: int = 64
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    hidden_size: int = 1024
    moe_intermediate_size: int = 768
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 32768
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    # Gradient checkpointing
    gradient_checkpointing: bool = False
    checkpointing_start_index: int = 0


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([freqs, freqs], dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to the query and key tensors.
    """
    cos = freqs_cis.cos().unsqueeze(0).unsqueeze(2).float()
    sin = freqs_cis.sin().unsqueeze(0).unsqueeze(2).float()
    xq_out = (xq.float() * cos) + (rotate_half(xq.float()) * sin)
    xk_out = (xk.float() * cos) + (rotate_half(xk.float()) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True, enable_gqa=True
        )
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()


class Qwen3MoeExperts(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.num_experts = model_args.num_experts
        self.hidden_dim = model_args.hidden_size
        self.intermediate_size = model_args.moe_intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_size),
            requires_grad=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, expert_idx=None):
        gate_proj_out = torch.matmul(
            hidden_states, self.gate_proj[expert_idx].transpose(0, 1)
        )
        up_proj_out = torch.matmul(
            hidden_states, self.up_proj[expert_idx].transpose(0, 1)
        )

        out = self.act_fn(gate_proj_out) * up_proj_out
        out = torch.matmul(out, self.down_proj[expert_idx].transpose(0, 1))
        return out

    def init_weights(self, init_std: float):
        for param in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.trunc_normal_(param, mean=0.0, std=init_std)

    def reset_parameters(self):
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.num_experts = model_args.num_experts
        self.top_k = model_args.num_experts_per_tok
        self.norm_topk_prob = model_args.norm_topk_prob

        # gating
        self.gate = nn.Linear(
            model_args.hidden_size, model_args.num_experts, bias=False
        )

        self.experts = Qwen3MoeExperts(model_args)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.experts(current_state, expert_idx)
                * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        self.experts.init_weights(init_std)

    def reset_parameters(self):
        self.gate.reset_parameters()
        self.experts.reset_parameters()


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LN Architecture and MoE Feed-Forward Network"""

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers
        self.depth_init = model_args.depth_init

        # Attention, MoE FFN and normalization layers
        self.attention = Attention(model_args)
        self.mlp = Qwen3MoeSparseMoeBlock(model_args)
        self.attention_norm = RMSNorm(model_args.dim, model_args.norm_eps)
        self.ffn_norm = RMSNorm(model_args.dim, model_args.norm_eps)

        # Weight initialization std based on depth (depth-wise initialization)
        self.init_std = 0.02 / (
            2 * (self.layer_id + 1) ** 0.5
            if self.depth_init
            else (2 * self.num_layers) ** 0.5
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Attention sub-layer with residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        # MoE FFN sub-layer with residual connection (take first output of MLP)
        out = h + self.mlp(self.ffn_norm(h))[0]
        return out

    def init_weights(self) -> None:
        """Initialize normalization and sub-layer weights"""
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(self.init_std)
        self.mlp.init_weights(self.init_std)

    def reset_parameters(self) -> None:
        """Reset all block parameters"""
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.reset_parameters()
        self.mlp.reset_parameters()


class Qwen3MoeMini(nn.Module):
    """Qwen3 MoE Mini Transformer Model with RoPE and Sparse Expert Layers"""

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        assert model_args.vocab_size > 0, (
            "vocab_size must be set before model initialization!"
        )
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        # Token embedding and precomputed RoPE frequencies
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                model_args.dim // model_args.n_heads,
                model_args.max_seq_len * 2,  # Double max seq len for generation safety
            ),
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(i, model_args) for i in range(model_args.n_layers)]
        )

        # Final normalization and output projection
        self.norm = RMSNorm(model_args.dim, model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # Initialize all model weights
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize all model parameters"""
        # Token embedding initialization
        nn.init.normal_(self.tok_embeddings.weight)
        # Initialize transformer layers
        for layer in self.layers:
            layer.init_weights()
        # Final normalization and output projection
        self.norm.reset_parameters()
        final_std = self.model_args.dim**-0.5
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_std,
            a=-3 * final_std,
            b=3 * final_std,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full model forward pass"""
        bsz, seq_len = tokens.shape
        # Token embedding lookup
        h = self.tok_embeddings(tokens)
        # Get RoPE frequencies for current sequence length and move to device
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)

        # Forward pass through transformer layers with optional gradient checkpointing
        for i, layer in enumerate(self.layers):
            if (
                self.model_args.gradient_checkpointing
                and i >= self.model_args.checkpointing_start_index
            ):
                h = torch.utils.checkpoint.checkpoint(layer, h, freqs_cis)
            else:
                h = layer(h, freqs_cis)

        # Final normalization and output logits
        h = self.norm(h)
        return self.output(h)

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Qwen3MoeMini":
        """Create model instance from ModelArgs configuration"""
        return cls(model_args)

    def reset_parameters(self) -> None:
        """Reset all model parameters to initial state"""
        self.tok_embeddings.reset_parameters()
        self.norm.reset_parameters()
        self.output.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


if __name__ == "__main__":
    qwen3_moe_mini_config = ModelArgs(
        dim=1024,
        n_layers=2,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=3200,
    )
    net = Qwen3MoeMini.from_model_args(qwen3_moe_mini_config).cuda().bfloat16()

    x = torch.randint(0, 3200, (20, 128)).cuda()
    y = net(x)
    print(y.shape)
    y.sum().backward()
