from ivoice.pipeline.asr.speech_recognition import SpeechRecognition
from ivoice.pipeline.sd.speaker_diarization import SpeakerDiarization
from ivoice.util.path import get_project_dir

import os


def transcibe(audio_file):
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
  segments_label_composed = sd(audio_file, p_val=0.4)

  with open('result/temp.txt', 'w', encoding='UTF-8') as f:
    for composed in segments_label_composed:
      segment = composed['segment']
      label = composed['label']
      word = asr(audio_file, segment)
      f.write('{} {} {}\n'.format(segment, label, word))
      composed['word'] = word

  return segments_label_composed


result = transcibe('../../../samples/siri.wav')
