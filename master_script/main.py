import numpy as np
from time import sleep

from audio import dsp_pipeline
from class_defs import virtualizer

GAIN = 50
FS = 44100
C = 343
MIC_POSITIONS = np.array([ # estimated positions of the microphones
    [0, 0],
    [0, 0.045],
    [0.039, 0.023],
    [0.039, -0.023],
    [0, -0.045],
    [-0.039, -0.023],
    [-0.039, 0.023],
])

class beamformer:
    def __init__(self, angle): # angle is in degrees
        self.buffer = np.zeros((16384, 7))

        target_dir = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        delay = (FS/C)*MIC_POSITIONS@target_dir+8
        delay = np.round(delay+0.5).astype(int) # round to nearest integer
        self.delay = delay
    def beamform(self, new_buffer):
        working_buffer = np.concatenate([self.buffer, new_buffer])

        for i, delay in enumerate(self.delay):
            working_buffer[:, i] = np.roll(working_buffer[:, i], delay)
        output_buffer = np.average(working_buffer[16384:], axis=1)

        self.buffer = new_buffer
        return output_buffer

class sequential_block:
    def __init__(self):
        self.routes = []
    def add_route(self, angle, pos):
        self.routes.append((beamformer(angle), virtualizer(pos)))
    def routine(self, raw_samples):
        output = np.zeros((16384, 2), dtype=np.float32)
        for bf, virt in self.routes:
            bf_output = bf.beamform(raw_samples[:, :7])
            # other processing can go here
            virt_output_left, virt_output_right = virt.virtualize(bf_output)
            virt_output_left = np.real(virt_output_left).astype(np.float32)
            virt_output_right = np.real(virt_output_right).astype(np.float32)
            output[:, 0] += virt_output_left
            output[:, 1] += virt_output_right
        return output

block = sequential_block()
block.add_route(0, 'l')
block.add_route(120, 'c')
block.add_route(240, 'r')

pipe = dsp_pipeline(block.routine)
pipe.start()

print('DSP pipeline started...')
print('Press Ctrl+C to stop...')

try:
    while True:
        pipe.process()
except KeyboardInterrupt:
    pipe.stop()