from transformers import AutoFeatureExtractor
from torch import Tensor


class WaLMvFeatureExtractor():
  """基于Micorsoft的WavLM进行特征提取
  """

  def __init__(self, model_name, device) -> None:
    self.model = AutoFeatureExtractor.from_pretrained(model_name)
    self.device = device

  def __call__(
      self,
      waveform: Tensor,
      sample_rate: int
  ):
    return self.model(
      waveform.squeeze(0),
      return_tensors='pt',
      sampling_rate=sample_rate
    ).input_values.to(self.device)
