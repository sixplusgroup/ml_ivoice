from transformers import AutoModelForAudioXVector
import torch

class SpeakerRepresentation:
  """ 基于Microsoft的WavLM的说话人特征提取模型
  
  详情请见
    https://huggingface.co/spaces/microsoft/wavlm-speaker-verification
  WavLM相关请见论文
    https://arxiv.org/pdf/2110.13900.pdf
  """

  def __init__(
      self,
      x_vector,
      device
  ):
    self.x_vector = AutoModelForAudioXVector.from_pretrained(x_vector).to(device)

  def __call__(self, features):
    embedding = self.x_vector(features).embeddings
    return torch.nn.functional.normalize(embedding, dim=-1).cpu().detach().numpy()

