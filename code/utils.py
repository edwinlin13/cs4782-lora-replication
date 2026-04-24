import torch
import json
import os
import time


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percentage": 100.0 * trainable / total if total > 0 else 0.0,
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """Load model checkpoint. Returns the epoch and loss."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


# TODO: maybe add a load_metrics function too
def save_metrics(metrics, path):
    """Save metrics dict as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


class TrainingLogger:
    """Logs training metrics per epoch."""

    def __init__(self):
        self.history = {"train_loss": [], "val_loss": [], "epoch_time": []}
        self._epoch_start = None

    def start_epoch(self):
        self._epoch_start = time.time()

    def end_epoch(self, train_loss, val_loss=None):
        elapsed = time.time() - self._epoch_start
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["epoch_time"].append(elapsed)
        return elapsed

    def get_history(self):
        return self.history
