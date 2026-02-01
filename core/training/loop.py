import torch
from core.training.metrics import MetricTracker


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    epoch,
    log_every=10,
):
    model.train()
    metrics = MetricTracker()

    for step, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # ---- forward ----
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # ---- backward ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ---- step ----
        optimizer.step()

        # ---- metrics ----
        batch_size = inputs.size(0)
        metrics.update(loss.item(), batch_size)

        if step % log_every == 0:
            avg = metrics.compute()
            print(
                f"[epoch {epoch} step {step}] "
                f"loss={avg['loss']:.4f}"
            )

    return metrics.compute()
