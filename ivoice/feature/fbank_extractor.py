import torch
import torchaudio.compliance.kaldi as kaldi


class FBankFeatureExtractor:
  """提取fbank特征
  """

  def __init__(self):
    pass

  def get_feature(
      self,
      waveform: torch.Tensor,
      num_mel_bins: int = 80,
      frame_length: float = 25.0,
      frame_shift: float = 10.0,
      dither: float = 0.0,
      energy_floor: float = 1.0,
      sample_rate: float = 16000.0,
  ):
    waveform = waveform * (1 << 15)
    fbank_feature = kaldi.fbank(waveform=waveform,
                                num_mel_bins=num_mel_bins,
                                frame_length=frame_length,
                                frame_shift=frame_shift,
                                dither=dither,
                                energy_floor=energy_floor,
                                sample_frequency=sample_rate
                                )
    fbank_feature_length = torch.tensor([fbank_feature.size(0)], dtype=torch.int32)

    return fbank_feature, fbank_feature_length
