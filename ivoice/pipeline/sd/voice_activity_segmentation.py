from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation


class VoiceActivitySegmentation:
  """基于pyannote的segmentation模型实现分段，在这里我并没有对重叠部分进行处理
  
  模型见
    https://huggingface.co/pyannote/segmentation
  论文见
    https://arxiv.org/pdf/2104.04045.pdf
  """

  def __init__(
      self,
      segmentation='pyannote/segmentation',
      params=None
  ) -> None:
    self.segmentation = VoiceActivityDetection(segmentation=segmentation)
    if params is None:
      self.segmentation.instantiate(self.get_default_params())
    else:
      self.segmentation.instantiate(params)

  def __call__(self, audio_file) -> Annotation:
    return self.segmentation(audio_file)

  def get_default_params(self):
    return {
      "onset": 0.5,
      "offset": 0.5,
      "min_duration_on": 0.0,
      "min_duration_off": 0.0,
    }
