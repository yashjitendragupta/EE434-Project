import numpy as np
from scipy.io import wavfile as wav
from scipy.fft import fft

class virtualizer:
    def __init__(self,direction):
        self.buffer = np.zeros(1024)

        # take 60% azimuth for if it's the left or right
        # 0% for middle
        if(direction == 'l' or direction == 'r'):
            hrtf_file = "../HRTF/elev0/H0e060a.wav"
        else:
            hrtf_file = "../HRTF/elev0/H0e000a.wav"

        hrtf_samplerate, hrtf = wav.read(hrtf_file)

        # flip left and right channels of hrtf for left side
        if(direction == 'l'):
            dummy = hrtf[:, 0].copy()
            hrtf[:, 0] = hrtf[:, 1]
            hrtf[:, 1] = dummy

        N_hrtf = len(hrtf[:, 0])

        # zero pad HRTF to two times the buffer

        hrtf_0 = np.concatenate([hrtf[:, 0], np.zeros(2048-N_hrtf)])
        hrtf_1 = np.concatenate([hrtf[:, 1], np.zeros(2048-N_hrtf)])

        # store the FFT of the HRTFs because we don't actually need the sample domain 
        self.HRTF = [None, None]
        self.HRTF[0] = np.fft.fft(hrtf_0)
        self.HRTF[1] = np.fft.fft(hrtf_1)







    def virtualize(self, new_buffer):
        
        # concatenate old buffer and new buffer
        self.buffer = np.concatenate([self.buffer, new_buffer])

        # take that concatenated buffer, take the fft

        buffer_fft = np.fft.fft(self.buffer)

        # filter using fft of HRTFS per channel
        left_channel_fft = self.HRTF[0] * buffer_fft
        right_channel_fft = self.HRTF[1] * buffer_fft

        # take iffts
        left_channel = np.fft.ifft(left_channel_fft)
        right_channel = np.fft.ifft(right_channel_fft)

        # take last half corresponding to new buffer
        left_channel = left_channel[1024:]
        right_channel = right_channel[1024:]

        # store new buffer for next time
        self.buffer = self.buffer[1024:]
        
        return (left_channel, right_channel)

