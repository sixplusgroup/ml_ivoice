import torch
import torch.nn.functional as F
from typeguard import check_argument_types

class CTC(torch.nn.Module):
  def __init__(
      self,
      output_dim: int,
      input_dim: int,
      dropout_rate: float = 0.0,
      reduction: bool = True
  ) -> None:
    assert check_argument_types()
    super().__init__()
    self.dropout_rate = dropout_rate
    self.ctc_lo = torch.nn.Linear(input_dim, output_dim)
    reduction_type = "sum" if reduction else "none"
    self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

  def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
              ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
    """Calculate CTC loss.

    Args:
        hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        hlens: batch of lengths of hidden state sequences (B)
        ys_pad: batch of padded character id sequence tensor (B, Lmax)
        ys_lens: batch of lengths of character sequence (B)
    """
    # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
    ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
    # ys_hat: (B, L, D) -> (L, B, D)
    ys_hat = ys_hat.transpose(0, 1)
    ys_hat = ys_hat.log_softmax(2)
    loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
    # Batch-size average
    loss = loss / ys_hat.size(1)
    return loss
  
  def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
  
  def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
    return torch.argmax(self.ctc_lo(hs_pad), dim=2)


