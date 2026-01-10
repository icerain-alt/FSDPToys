import time
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T

from models.llama3 import Transformer, ModelArgs
from utils import format_metrics_to_gb, print_model_info, seed_all


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch llama CUDA Example")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Total training epochs"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of data loading workers"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=421, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--seq_len", type=int, default=128, help="Input sequence length"
    )
    parser.add_argument(
        "--checkpointing_start_index",
        type=int,
        default=0,
        help="Checkpointing start from which layer",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (default: False)",
    )

    args = parser.parse_args()

    seed_all(args.seed, mode=False)

    return args


def train_one_epoch(model, loader, optimizer, epoch, args):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, _) in enumerate(loader):
        t0 = time.time()
        inputs = inputs.reshape(-1, args.seq_len).cuda()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = outputs.mean()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Calculate metrics
        total_loss += loss.item()

        print(
            f"Epoch: {epoch} | Batch: {batch_idx}/{len(loader)} | Elapsed Time: {time.time() - t0:.3f} s | Loss: {loss.item():.4f} | "
            f"Mem_alloc: {format_metrics_to_gb(torch.cuda.memory_allocated())} GB Mem_reserve: {format_metrics_to_gb(torch.cuda.memory_reserved())} GB"
        )


def main():
    args = get_args()

    # Prepare dataset
    train_set = datasets.FakeData(
        size=10000,
        image_size=(1, args.seq_len),
        num_classes=10,
        transform=T.Compose([T.ToTensor(), lambda x: (x * 256).int()]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    simple_llama2_config = ModelArgs(n_layers=2, vocab_size=10000)

    model = Transformer.from_model_args(simple_llama2_config).cuda()

    print_model_info(model)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    # Training loop
    for epoch in range(args.num_epochs):
        train_one_epoch(model, train_loader, optimizer, epoch, args)


if __name__ == "__main__":
    main()
