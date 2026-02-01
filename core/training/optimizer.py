import torch


def build_optimizer(model, config):
    opt_cfg = config.optim

    if opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
