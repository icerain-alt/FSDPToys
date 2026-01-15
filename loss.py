# Referenced from https://github.com/InternLM/xtuner/blob/main/xtuner/v1/loss/chunk_loss.py

from typing import Callable

import torch
import torch.nn.functional as F


def ce_loss_fun(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    labels = labels.contiguous().view(-1)
    hidden_states = hidden_states.contiguous().view(-1, hidden_states.size(-1))
    logits = F.linear(hidden_states, weight).float()
    return F.cross_entropy(logits, labels, reduction=reduction)


class ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        loss_forward: Callable,
        chunk_labels: torch.Tensor,
        chunk_size: int,
    ):
        device = hidden_states.device
        batch, seq_len, hidden_size = hidden_states.shape
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)

        for i in range(len(hidden_states_chunks)):
            hidden_states_chunk = hidden_states_chunks[i]
            grad_inputs_chunk = grad_inputs_chunks[i]

            (chunk_grad_input, chunk_grad_weight), chunk_loss = (
                torch.func.grad_and_value(loss_forward, argnums=(0, 1), has_aux=False)(
                    hidden_states_chunk, head_weight, chunk_labels[i]
                )
            )

            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss.div_(batch * seq_len)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None


def chunk_loss_fun(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
    loss_forward: Callable = ce_loss_fun,
    chunk_size: int = 1024,
) -> torch.Tensor:
    return ChunkLoss.apply(
        hidden_states,
        head_weight,
        loss_forward,
        torch.split(labels, chunk_size, dim=1),
        chunk_size,
    )


if __name__ == "__main__":
    torch.manual_seed(21)
    hidden_states = torch.randn(3, 4000, 128, device="cpu")
    head_weight = torch.randn(256, 128, device="cpu")
    labels = torch.randint(0, 100, (3, 4000), device="cpu")
    loss_gt = ce_loss_fun(hidden_states, head_weight, labels, reduction="mean")
    loss_chunk = chunk_loss_fun(hidden_states, head_weight, labels)
    # Check if the chunk loss is close to the ground truth loss
    assert torch.allclose(loss_chunk, loss_gt, atol=1e-5)
