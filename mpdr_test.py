# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from beamformer import util
from beamformer import mpdr

SAMPLING_FREQUENCY = 48000
FFT_LENGTH = 1024
FFT_SHIFT = 512
ENHANCED_WAV_NAME = './output/enhanced_2ch_mpdr_1mix.wav.wav'
# circular array
MIC_ANGLE_VECTOR = np.array([0,180])
LOOK_DIRECTION = 45
MIC_DIAMETER = 0.161

def multi_channel_read(prefix=r'./input/1mix/SNR_0/45F/hal_in_pure_24_4ch_48k_{}+noise1.wav',
                       channel_index_vector=np.array([1, 2])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
    wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(1, len(channel_index_vector)):
        wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
    return wav_multi

multi_channels_data = multi_channel_read()

complex_spectrum, _ = util.get_3dim_spectrum_from_data(multi_channels_data, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)

mpdr_beamformer = mpdr.minimum_power_distortioless_response(MIC_ANGLE_VECTOR, MIC_DIAMETER, sampling_frequency=SAMPLING_FREQUENCY, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

steering_vector = mpdr_beamformer.get_sterring_vector(LOOK_DIRECTION)

spatial_correlation_matrix = mpdr_beamformer.get_spatial_correlation_matrix(multi_channels_data)

beamformer = mpdr_beamformer.get_mpdr_beamformer(steering_vector, spatial_correlation_matrix)

enhanced_speech = mpdr_beamformer.apply_beamformer(beamformer, complex_spectrum)

sf.write(ENHANCED_WAV_NAME, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)

