# core/training/loop.py
import torch
from core.distributed.sync import sync_and_validate_loss


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    epoch,
    dist_ctx,
):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        # Explicit DDP correctness check
        if dist_ctx.world_size > 1:
            loss = sync_and_validate_loss(loss)

        total_loss += loss.item()

        if dist_ctx.is_rank_zero and step % 100 == 0:
            print(f"[Epoch {epoch} | Step {step}] loss={loss.item():.4f}")

    return {"loss": total_loss / len(dataloader)}