from typing import List, TypedDict
import numpy as np
import torch

from pyannote.core import Segment
from pyannote.core.annotation import Annotation

from ivoice.feature.wavlm_extractor import WaLMvFeatureExtractor
from ivoice.io.audio import AudioIO
from ivoice.pipeline.sd.clustering import SpectralClustering
from ivoice.pipeline.sd.speaker_representation import SpeakerRepresentation
from ivoice.pipeline.sd.voice_activity_segmentation import VoiceActivitySegmentation

cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


class SegmentLabel(TypedDict, total=False):
  segment: Segment
  label: str
  word: str


class SpeakerDiarization():
  """speaker diarization的流程，缝合了多个模型
  """

  def __init__(
      self,
      segmentation='pyannote/segmentation',
      feature_extractor='microsoft/wavlm-base-plus-sv',
      speaker_representation='microsoft/wavlm-base-plus-sv',
      speaker_representation_dimensions=512,
      affinity_type='cos',
      sample_rate=16000,
      num_speakers=2
  ):
    super().__init__()
    # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.device = torch.device('cpu')
    self.segmentation = VoiceActivitySegmentation(segmentation)
    self.feature_extractor = WaLMvFeatureExtractor(feature_extractor, self.device)
    self.speaker_representation = SpeakerRepresentation(speaker_representation, self.device)
    self.speaker_representation_dimensions = speaker_representation_dimensions
    self.clustering = SpectralClustering(affinity_type, num_speakers)
    self.audio_io = AudioIO(sample_rate)

  def __call__(self, audio_file, n_neighbors=4, p_val=0.4) -> None:
    # step 1: do segmentation
    segmentation_annotation: Annotation = self.segmentation(audio_file)
    valid_segments = self.validate_segment_duration(segmentation_annotation)

    # step 2: do segmentation representation calculation
    # embedding.shape (segmentation, embedding_dimension)
    embeddings = np.empty(shape=[0, self.speaker_representation_dimensions], dtype=np.float64)
    for segment in valid_segments:
      waveform, _ = self.audio_io.crop(audio_file, segment)
      features = self.feature_extractor(
        waveform.squeeze(0),
        sample_rate=self.audio_io.sample_rate
      )
      embedding = self.speaker_representation(features)
      embeddings = np.concatenate((embeddings, embedding), axis=0)

    # step 3: do spectral clustering
    labels = self.clustering(embeddings, n_neighbors, p_val)
    result = []
    for segment, label in zip(valid_segments, labels):
      composed: SegmentLabel = {
        'segment': segment,
        'label': label,
      }
      result.append(composed)
    return self.merge_same_label_segment(result)

  def validate_segment_duration(self, segments: Annotation, min_duration: float = 0.35):
    """去除一些duration较小的片段，避免计算embedding时报错

    Args:
        segments (Annotation): 分好的片段
        min_duration (float, optional): 最小持续时间. Defaults to 0.35.

    Returns:
        [List[Segment]]
    """
    valid_segments = []
    for segment in segments.itersegments():
      if segment.duration >= min_duration:
        valid_segments.append(segment)

    return valid_segments

  def merge_same_label_segment(self, segments: List[SegmentLabel], min_gap: float = 0.4):
    """将相同label并且间距小于min_gap的segment合并

    Args:
        segments (List[SegmentLabel]): 完成diarization的结果
        min_gap (float, optional): 最小间距. Defaults to 0.3.

    Returns:
        [List[SegmentLabel]]: 合并后的结果
    """
    assert len(segments) > 1
    new_segments = []
    first_segment = segments[0]
    end_index = 1
    while end_index < len(segments):
      second_segment = segments[end_index]
      if first_segment['label'] == second_segment['label'] \
        and second_segment['segment'].start - first_segment['segment'].end < min_gap:
        # 此时合并
        new_segment = {
          'segment': Segment(first_segment['segment'].start, second_segment['segment'].end),
          'label': first_segment['label']
        }
        first_segment = new_segment
      else:
        new_segments.append(first_segment)
        first_segment = second_segment
      end_index += 1

    # 判断最后一块是否被合并了
    if second_segment['segment'].end != first_segment['segment'].end:
      new_segments.append(second_segment)
    else:
      new_segments.append(first_segment)

    return new_segments
