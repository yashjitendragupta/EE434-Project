import numpy as np
from matplotlib import pyplot as plt

c = 343
frequencies = [700, 1400, 2800, 5600]
incoming_angles = np.linspace(0, 2*np.pi, 1000)
X = np.array([ # estimated positions of the microphones
    [0, 0],
    [0, 0.045],
    [0.039, 0.023],
    [0.039, -0.023],
    [0, -0.045],
    [-0.039, -0.023],
    [-0.039, 0.023],
])
target_dir = np.array([1, 0])

def W(k, target_k):
    weights = (1/len(X))*np.exp(1j*X@target_k)
    # window_factor = 1/6 # based on the geometry, all axially-symmetric windows reduce to a single factor (1/6 is rectangular)
    # weights[0] *= len(X)*window_factor # assuming that the first microphone is the center microphone...
    # weights[1:] *= len(X)*(1-window_factor)/6  #  cancel out the previous factor of (1/len(X)) then set new factors
    return weights.dot(np.exp(-1j*X@k))

incoming_w_over_freq = []
for freq in frequencies:
    target_k = (2*np.pi*freq/c)*target_dir
    incoming_k = [np.array([np.cos(theta), np.sin(theta)])*(2*np.pi*freq)/c for theta in incoming_angles]
    incoming_w = np.array([W(k, target_k) for k in incoming_k])
    incoming_w_over_freq.append(incoming_w)

plt.figure(1)
for freq, incoming_w in zip(frequencies, incoming_w_over_freq):
    plt.polar(incoming_angles, 20*np.log10(np.abs(incoming_w)), label=f'{freq} Hz')
plt.ylim([-24, 0])
plt.legend()
plt.title('UMA-8 Theoretical Array Pattern over Frequency')

plt.show()