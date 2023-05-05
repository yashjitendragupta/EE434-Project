import numpy as np
from time import time, sleep
import subprocess
from threading import Thread

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
        self.buffer = np.zeros((4096, 7))
        self.update_angle(angle)
    def update_angle(self, angle):
        target_dir = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        delay = (FS/C)*MIC_POSITIONS@target_dir+8
        delay = np.round(delay+0.5).astype(int)
        self.delay = delay
    def beamform(self, new_buffer):
        working_buffer = np.concatenate([self.buffer, new_buffer])

        for i, delay in enumerate(self.delay):
            working_buffer[:, i] = np.roll(working_buffer[:, i], delay)
        output_buffer = np.average(working_buffer[4096:], axis=1)

        self.buffer = new_buffer
        return output_buffer

class sequential_block:
    def __init__(self):
        self.routes = []
    def add_route(self, angle, pos):
        self.routes.append((beamformer(angle), virtualizer(pos)))
    def routine(self, raw_samples):
        output = np.zeros((4096, 2), dtype=np.float32)
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

print() # the call to pyaudio tends to mess up the terminal, so we'll separate our own messages from that

print('Starting lidar process...')

lidar_process = subprocess.Popen(
    ['env', 'PYTHONUNBUFFERED=1', 'python3', 'lidar.py'], 
    bufsize=1, # line buffered for minimal latency
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT # we'll capture stout, but we'll let stderr go to the terminal
)
lidar_pipe = iter(lidar_process.stdout)
init_line = next(lidar_pipe) # wait for the lidar process to start
init_line = init_line.decode('utf-8').strip() # (the line will be "PyRPlidar Info : device is connected\n")

# read and update the angles in a separate thread
angles = [0, 0, 0] # placeholder values for the angles
def lidar_reading_routine():
    global angles
    while True:
        lidar_response = next(lidar_pipe).decode('utf-8')
        lidar_response = lidar_response.split(',')
        lidar_response = [angle_str.strip() for angle_str in lidar_response]
        angles = [float(angle_str) for angle_str in lidar_response[:3]]
lidar_reading_thread = Thread(target=lidar_reading_routine, daemon=True)
lidar_reading_thread.start()

print('Lidar pipeline started with response "%s"' % init_line)
print('Waiting 30 seconds for lidar pipeline to finish initializing...')
sleep(30) # wait for the rest of the program to initialize

pipe = dsp_pipeline(block.routine)
pipe.start()

print('DSP pipeline started...')
print('Press Ctrl+C to stop...')

try:
    while True:
        pipe.process()

        for i, angle in enumerate(angles):
            block.routes[i][0].update_angle(angle)
except KeyboardInterrupt:
    pipe.stop()