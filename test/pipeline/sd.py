from pyannote.audio.pipelines import SpeakerDiarization

import sys
sys.path.append('c:\\Users\\secri\\Documents\\GitHub\\ml_ivoice')

from ivoice.pipeline.sd.speaker_diarization import SpeakerDiarization
from pyannote.core.segment import Segment

sd = SpeakerDiarization()

waveform, _ = sd.audio_io.crop('../../samples/siri.wav', Segment(2, 2.35))
features = sd.feature_extractor(
    waveform.squeeze(0),
    sample_rate=sd.audio_io.sample_rate
)

embedding = sd.speaker_representation(features)
