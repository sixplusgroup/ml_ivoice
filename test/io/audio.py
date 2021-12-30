from pyannote.core.segment import Segment
import sys
sys.path.append('../..')
from ivoice.io.audio import AudioIO

audio_io = AudioIO(sample_rate=16000)

segment = Segment(5, 6.1)

waveform, _ = audio_io.crop('../../samples/eng.wav', segment)

print(waveform)