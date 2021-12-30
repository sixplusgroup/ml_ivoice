from pyannote.core import Segment

from ivoice.pipeline.asr.speech_recognition import SpeechRecognition
from ivoice.util.path import get_project_dir

import os

config_path = os.path.join(get_project_dir(), 'pretrained/wenet/train.yaml')
checkpoint_path = os.path.join(get_project_dir(), 'pretrained/wenet/final.pt')
word_dict_path = os.path.join(get_project_dir(), 'pretrained/wenet/words.txt')

asr = SpeechRecognition(
  config_path=config_path,
  checkpoint_path=checkpoint_path,
  word_dict_path=word_dict_path
)

result = asr('../../samples/siri.wav', segment=Segment(21.287, 22.300))