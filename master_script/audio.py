import re
from time import sleep
import queue
import pyaudio
import numpy as np

# resources:
# https://github.com/TheSalarKhan/Linux-Audio-Loopback-Device
AUDIO_INTERFACE = pyaudio.PyAudio() # https://people.csail.mit.edu/hubert/pyaudio/docs/

class dsp_pipeline:
    def __init__(
            self, 
            routine, # expected signature is func(samples_in) -> samples_out where samples are length-4096
            target_input_device='micArray RAW SPK: USB Audio \(hw:[0-9]+,0\)',  # device identified using regex
            target_output_device='Loopback: PCM \(hw:[0-9]+,1\)', # listen at 'Loopback: PCM (hw:[NUMBER],0)'
            ):
        self.audio_interface = pyaudio.PyAudio()
        
        # find the input and output devices
        i_input = None
        i_output = None
        n_devices = AUDIO_INTERFACE.get_device_count()
        for i_device in range(n_devices):
            device = AUDIO_INTERFACE.get_device_info_by_index(i_device)
            if re.match(target_input_device, device['name']):
                i_input = i_device
            if re.match(target_output_device, device['name']):
                i_output = i_device

        if i_input is None:
            raise IOError(f'{target_input_device} not found')
        if i_output is None:
            raise IOError(f'{target_output_device} not found, have you run "sudo modprobe snd-aloop"?')
        
        # input_stream  --[in_queue]-->  routine  --[out_queue]-->  output_stream
        self.in_queue = queue.SimpleQueue()
        self.out_queue = queue.SimpleQueue()
        self.routine = routine
        self.input_stream = self.audio_interface.open(
            rate=44100,
            channels=8,
            format=pyaudio.paFloat32,
            input=True,
            input_device_index=i_input,
            frames_per_buffer=4096,
            start=False,
            stream_callback=self._in_callback,
        )
        self.output_stream = self.audio_interface.open(
            rate=44100,
            channels=2,
            format=pyaudio.paFloat32,
            output=True,
            output_device_index=i_output,
            frames_per_buffer=4096,
            start=False,
            stream_callback=self._out_callback,

        )
    def _in_callback(self, samples_bytes_in, n_samples, metadata, flags): # TODO: use metadata and flags
        self.in_queue.put(samples_bytes_in)
        return (None, pyaudio.paContinue)
    def _out_callback(self, samples_bytes_in, n_samples, metadata, flags):
        samples_bytes_out = self.out_queue.get()
        return (samples_bytes_out, pyaudio.paContinue)
    def start(self):
        self.input_stream.start_stream()
        self.output_stream.start_stream()
    def process(self):
        samples_bytes_in = self.in_queue.get()
        samples_np_in = np.frombuffer(samples_bytes_in, dtype=np.float32).reshape((4096, 8))
        samples_np_out = self.routine(samples_np_in).astype(np.float32) # make sure the type is what pyaudio expects
        samples_bytes_out = samples_np_out.tobytes()
        self.out_queue.put(samples_bytes_out)
    def stop(self):
        self.input_stream.close()
        self.output_stream.close()
        self.audio_interface.terminate()

if __name__ == '__main__':
    def passthrough_routine(samples_in):
        samples_out = np.zeros((4096, 2), dtype=np.float32)
        samples_out[:, 0] = samples_in[:, 0]
        samples_out[:, 1] = samples_in[:, 1]
        return samples_out
    
    pipeline = dsp_pipeline(passthrough_routine)
    pipeline.start()

    print('Test pipeline started...')
    print('Press Ctrl+C to stop...')

    try:
        while True:
            pipeline.process()
    except KeyboardInterrupt:
        pipeline.stop()