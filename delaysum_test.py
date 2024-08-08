# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from beamformer import delaysum as ds
from beamformer import util

# linear array
MIC_ANGLE_VECTOR = np.array([0,180])
MIC_DIAMETER = 0.161
FFT_LENGTH = 1024
FFT_SHIFT = 512
SAMPLING_FREQUENCY = 48000
ENHANCED_WAV_NAME = './output/enhanced_2ch_delaysum_1mix.wav'
LOOK_DIRECTION = 45

def multi_channel_read(prefix=r'./input/1mix/SNR_0/45F/hal_in_pure_24_4ch_48k_{}+noise1.wav',
                       channel_index_vector=np.array([1, 2])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
    wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
    print("shape:",wav_multi.shape)
    wav_multi[:, 0] = wav
    for i in range(1, len(channel_index_vector)):
        wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
    return wav_multi

multi_channels_data = multi_channel_read()

complex_spectrum, _ = util.get_3dim_spectrum_from_data(multi_channels_data, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)

ds_beamformer = ds.delaysum(MIC_ANGLE_VECTOR, MIC_DIAMETER, sampling_frequency=SAMPLING_FREQUENCY, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

beamformer = ds_beamformer.get_sterring_vector(LOOK_DIRECTION)

enhanced_speech = ds_beamformer.apply_beamformer(beamformer, complex_spectrum)

sf.write(ENHANCED_WAV_NAME, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)
