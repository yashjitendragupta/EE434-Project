import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav

mic_locations = np.array([
    [ 0.0000,  0.0000], # mic 1
    [ 0.0300,  0.0000],
    [  0.0150, 0.0259],
    [ -0.0150, 0.0259],
    [-0.0300,  0.0000],
    [ -0.0150,-0.0259],
    [ 0.0150, -0.0259], # mic 7
])
target_directions = np.array([
    [-0.7071, 0.7071],
    [ 0.9805,-0.1961],
    [-0.7071, 0.7071],
])

fs = 44100
c = 343
delays = mic_locations @ target_directions.T
sample_shift = -8-delays*(fs/c) # sample_shift[i][j] is the sample shift for mic i and target j; -8 for causality

print(sample_shift)

wavs = []
for i in range(7):
    _, w = wav.read(f'./Beamforming simulation experiment/outputs/mic_{i+1}_voices.wav')
    w = np.pad(w, (0, 16), mode='constant', constant_values=0) # we'll do shifting using np.roll, so we need padding
    wavs.append(w)

for i in range(3):
    w = np.zeros(len(wavs[0]), dtype=np.float32)
    for j in range(7):
        w += np.roll(wavs[j], -int(sample_shift[j, i]+0.5)) # adding 0.5 to round to nearest integer
    w /= 7
    w = w.astype(np.int16)
    wav.write(f'./beamforming_sim_experiment_2/out/shift_{i+1}.wav', fs, w)