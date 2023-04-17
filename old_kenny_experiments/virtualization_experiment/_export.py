from scipy.io import wavfile as wav
import numpy as np

raw_2_level = np.load('raw_2_level.npy')
raw_2_level *= 50
wav.write('mic_0_2_level.wav', 44100, raw_2_level)