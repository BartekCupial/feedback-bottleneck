import numpy as np
import torch


def get_ndim(x):
    if torch.is_tensor(x):
        return x.ndim
    elif isinstance(x, np.ndarray):
        return x.ndim
    else:
        return 0  # most likely python: float, int, etc.


class MetricsTracker:
    def __init__(self, writer):
        self.writer = writer

    def log_metrics(self, metrics, step, prefix="train"):
        """
        Log metrics to the writer.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Current training step.
            prefix (str): Prefix for the TensorBoard tag (e.g., 'train', 'eval').
        """
        for key, value in metrics.items():
            # Skip strings (e.g., "N/A"), None, or other non-loggable types
            if isinstance(value, (str, type(None))):
                continue

            # handle matplotlib figures
            if hasattr(value, "figure") or (hasattr(value, "canvas") and hasattr(value, "savefig")):
                self.writer.add_figure(f"{prefix}/{key}", value, step)
                continue

            # convert 0-d torch tensors or numpy scalars to native Python types
            if torch.is_tensor(value) and value.numel() == 1:
                value = value.item()
            elif isinstance(value, (np.ndarray, np.generic)) and value.ndim == 0:
                value = value.item()

            # scalars
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

            # multi-dimensional numeric arrays - log as histogram
            elif get_ndim(value) > 0:
                self.writer.add_histogram(f"{prefix}/{key}", value, step)

                # Also log the mean as a scalar for convenience
                if torch.is_tensor(value):
                    mean_value = value.float().mean().item()
                elif isinstance(value, np.ndarray):
                    mean_value = float(value.mean())
                else:
                    mean_value = value  # fallback

                self.writer.add_scalar(f"{prefix}/{key}_mean", mean_value, step)
