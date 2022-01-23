from ivoice.pipeline.asr.speech_recognition import SpeechRecognition
from ivoice.pipeline.asr.punctuation_restoration import PunctuationRestoration
from ivoice.pipeline.sd.speaker_diarization import SpeakerDiarization
from ivoice.util.path import get_project_dir
from ivoice.util.constant import DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP
import os

def transcribe(audio_file):
  # 以下三个文件以及global_cmvn需要从
  #   https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20211025_conformer_exp.tar.gz
  # 下载，由于比较大，就不放上去了
  config_path = os.path.join(get_project_dir(), 'pretrained/wenet/train.yaml')
  checkpoint_path = os.path.join(get_project_dir(), 'pretrained/wenet/final.pt')
  word_dict_path = os.path.join(get_project_dir(), 'pretrained/wenet/words.txt')

  asr = SpeechRecognition(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    word_dict_path=word_dict_path
  )
  sd = SpeakerDiarization()
  pr = PunctuationRestoration(tag2punctuator=DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP)

  segments_label_composed = sd(audio_file, p_val=0.4)
  for composed in segments_label_composed:
    segment = composed['segment']
    word = asr(audio_file, segment)
    word_with_punc, _ = pr.punctuation([word])

    composed['word'] = word_with_punc[0]

  return segments_label_composed
