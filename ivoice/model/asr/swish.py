import torch

class Swish(torch.nn.Module):
  """Construct an Swish object."""
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Return Swish activation function."""
    return x * torch.sigmoid(x)
