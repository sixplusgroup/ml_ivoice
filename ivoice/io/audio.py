import math
import os
from pyannote.core.segment import Segment
import torchaudio
from pathlib import Path
from typing import Text, Tuple, Union
from torch import Tensor
from torchaudio.backend.common import AudioMetaData
import torch.nn.functional as F

AudioFile = Union[Text, Path]

class AudioIO:
  """对音频输入进行处理
  """
  def __init__(self, sample_rate, monophony = True) -> None:
    self.sample_rate = sample_rate
    self.monoPhony = monophony

  @classmethod
  def validate_file(cls, file: AudioFile):
    if os.path.exists(file):
      return {
        "audio": str(file),
        "uri": Path(file).stem
      }
    raise ValueError('该文件不存在')

  def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
    # downmix
    if self.monoPhony and waveform.shape[0] > 1:
      waveform = waveform.mean(dim=0, keepdim=True)
    
    # resample
    if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
      resample = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
      waveform = resample(waveform)
      sample_rate = self.sample_rate
    
    return waveform, sample_rate
  
  def get_duration(self, file: AudioFile):
    file = self.validate_file(file)
    file_info: AudioMetaData = torchaudio.info(file["audio"])
    return file_info.num_frames / file_info.sample_rate
  
  def __call__(self, file: AudioFile) -> Tuple[Tensor, int]:
    file = self.validate_file(file)
    waveform, sample_rate = torchaudio.load(file["audio"])
    return self.downmix_and_resample(waveform, sample_rate)

  def crop(
    self,
    file: AudioFile,
    segment: Segment = None,
    duration = None,
    mode = "raise"
  ):
    """
    mode: Optional["raise", "pad"]
      当范围超出边界时，raise会报错，而pad会将不足部分填充为0
    """
    file = self.validate_file(file)
    info: AudioMetaData = torchaudio.info(file["audio"])
    sample_rate = info.sample_rate
    frames = info.num_frames

    if segment is None and duration is None:
      waveform, _ = torchaudio.load(
        file["audio"],
        frame_offset=0
      )
      return self.downmix_and_resample(waveform, sample_rate)

    start_frame = math.floor(segment.start * sample_rate)

    if duration is not None:
      num_frames = math.floor(duration * sample_rate)
      end_frame = start_frame + num_frames
    else:
      end_frame = math.floor(segment.end * sample_rate)
      num_frames = end_frame - start_frame
    
    if mode == 'pad':
      pad_start = -min(0, start_frame)
      pad_end = max(end_frame, frames) - frames
      start_frame = max(0, start_frame)
      end_frame = min(end_frame, frames)
      num_frames = end_frame - start_frame
    else:
      if num_frames > frames or end_frame > frames or start_frame < 0:
        raise ValueError('请求范围不合法')

    waveform, _  = torchaudio.load(
      file["audio"],
      frame_offset=start_frame,
      num_frames=num_frames
    )

    if mode == 'pad':
      waveform = F.pad(waveform, (pad_start, pad_end))
    
    return self.downmix_and_resample(waveform, sample_rate)