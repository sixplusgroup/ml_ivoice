import torch
import yaml
import logging
import sys
from ivoice.io.audio import AudioIO
from ivoice.model.asr.asr_model import init_asr_model
from ivoice.feature.fbank_extractor import FBankFeatureExtractor
from ivoice.util.config import override_config
from ivoice.util.checkpoint import load_checkpoint
from pyannote.core.segment import Segment


# TODO注释
class SpeechRecognition:
  """ speech recognition的一系列流程
  参考了wenet中定义的asr_model
    https://github.com/wenet-e2e/wenet
    
  预训练文件太大，就没有放进来，下载地址为
    https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20211025_conformer_exp.tar.gz
  """

  def __init__(
      self,
      config_path,
      checkpoint_path,
      word_dict_path,
      sample_rate=16000,
      gpu=-1,
      batch_size=1,
      mode='attention_rescoring',
      override_config_items=[],
  ):
    if mode in ['ctc_prefix_beam_search', 'attention_rescoring'] and batch_size > 1:
      logging.fatal(
        'decoding mode {} must running with batch_size == 1'.format(
          mode
        )
      )
      sys.exit(1)
    self.decode_mode = mode
    with open(config_path, 'r') as config_file:
      config = yaml.load(config_file, Loader=yaml.FullLoader)

    if len(override_config_items) > 0:
      config = override_config(config, override_config_items)

    use_cuda = gpu >= 0 and torch.cuda.is_available()

    self.device = torch.device('cuda' if use_cuda else 'cpu')
    self.model = init_asr_model(config)
    load_checkpoint(self.model, checkpoint_path, use_cuda)

    word_dict = {}
    with open(word_dict_path, 'r', encoding='UTF-8') as word_dict_file:
      for line in word_dict_file:
        pairs = line.strip().split()
        assert len(pairs) == 2
        word_dict[int(pairs[1])] = pairs[0]
    eos = len(word_dict) - 1

    self.word_dict = word_dict
    self.eos = eos

    self.audio_io = AudioIO(sample_rate)
    self.feature_extractor = FBankFeatureExtractor()

  def __call__(
      self,
      file_path: str,
      segment: Segment = None,
      beam_size: int = 10,
      decoding_chunk_size: int = -1,
      ctc_weight: float = 0.5,
      num_decoding_left_chunk: int = -1,
      simulate_streaming: bool = False,
      reverse_weight: float = 0.0,
  ):
    self.model.to(self.device)
    self.model.eval()
    waveform, _ = self.audio_io.crop(file_path, segment=segment)

    fbank_feature, fbank_feature_length = self.feature_extractor.get_feature(waveform)
    # fbank_feature.shape (frames, dimension)
    # need to transform to (batch_size, frames, dimension)
    fbank_feature = fbank_feature.unsqueeze(dim=0)

    fbank_feature = fbank_feature.to(self.device)
    fbank_feature_length = fbank_feature_length.to(self.device)
    # TODO attention_rescoring 的效果最好，后续把其它方法去掉
    if self.decode_mode == 'attention':
      hyps, _ = self.model.recognize(
        fbank_feature,
        fbank_feature_length,
        beam_size=beam_size,
        decoding_chunk_size=decoding_chunk_size,
        num_decoding_left_chunks=num_decoding_left_chunk,
        simulate_streaming=simulate_streaming
      )
      hyps = [hyp.tolist() for hyp in hyps]
    elif self.decode_mode == 'ctc_greedy_search':
      hyps, _ = self.model.ctc_greedy_search(
        fbank_feature,
        fbank_feature_length,
        decoding_chunk_size=decoding_chunk_size,
        num_decoding_left_chunks=num_decoding_left_chunk,
        simulate_streaming=simulate_streaming
      )
      hyps = [hyps]
    elif self.decode_mode == 'attention_rescoring':
      assert fbank_feature.size(0) == 1
      hyp, _ = self.model.attention_rescoring(
        fbank_feature,
        fbank_feature_length,
        beam_size=beam_size,
        decoding_chunk_size=decoding_chunk_size,
        num_decoding_left_chunks=num_decoding_left_chunk,
        ctc_weight=ctc_weight,
        simulate_streaming=simulate_streaming,
        reverse_weight=reverse_weight
      )
      hyps = [hyp]
    content = ''
    for w in hyps[0]:
      if w == self.eos:
        break
      else:
        content += self.word_dict[w]
      logging.info(content)
    return content
