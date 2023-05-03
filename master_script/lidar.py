# In this file, constantly poll the LIDAR device
# then find the three theta positions, then write those
# thetas to some text file that the main script can read from.
# runs at the same time as the main script.
# writes -1 to file if error or does not exist
# writes between 0 to 360 for angle that does exist.

import numpy as np
from rplidar import RPLidar
from queue import Queue
from threading import Thread
from time import time

import motion

# Using the RPLidar class as a context manager ensures the serial comms are always closed, even on crashes
class RpLidarContext(RPLidar):
	def __enter__(self):
		return self
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()
		self.disconnect()

# thetas_txt = open("thetas.txt","w")

interp = motion.interpolator(memory_size=256, meters=True, radians=True)
filt = motion.low_pass_filter(360, corner_freq=0.5, sampling_freq=4) # so far, the best sampling freq observed is about 4 Hz
vel_est = motion.angular_velocity_estimator(window_size=16)

with RpLidarContext('/dev/ttyUSB0', baudrate=115200) as lidar:
	samples_queue = Queue()
	
	# the queuing rplidar uses seems flaky, so we'll run our own queue and monitior the queue's size
	def lidar_routine():
		try:
			samples_batch = []
			for samples in lidar.iter_scans(max_buf_meas=512):
				samples_batch.extend(samples)
				if len(samples_batch) >= 256:
					samples_queue.put(samples_batch) # pass samples into samples_queue
					samples_batch = []
		except Exception as e: # if an error is encountered, pass that error to the main thread
			samples_queue.put(e)

	lidar_thread = Thread(target=lidar_routine, daemon=True)
	lidar_thread.start()

	prev_ts = time()
	while True:
		samples = samples_queue.get(timeout=1) # collect samples from samples_queue
		if isinstance(samples, Exception): # if an error was received from the lidar daemon, raise it here
			raise samples
		
		# collect dt as a difference between timestamps
		current_ts = time()
		dt = current_ts - prev_ts
		prev_ts = current_ts

		# Do processing to find the three theta values
		
		angles = []
		distances = []
		for sample in samples[::2]: # downsample by 2
			angles.append(sample[1])
			distances.append(sample[2])
		angles = np.array(angles)
		distances = np.array(distances)

		interp.insert_many(angles, distances)
		angle_grid, heights = interp.generate()
		filtered_heights = filt.filter(heights)
		velocities = vel_est.estimate(filtered_heights, dt)
		labels = motion.detect_motion(velocities)
		indicator = motion.find_motion(labels)

		# for testing and visualization, represent velocities as an integer between 0 and 9
		maxpooled_velocity = np.max(np.abs(velocities).reshape(-1, 10), axis=1).reshape(-1)
		velocity_repr = (10*maxpooled_velocity).astype(np.int64)
		velocity_repr[velocity_repr < 0] = 0
		velocity_repr[velocity_repr > 9] = 9

		# report the velocity representation, dt, and queue size (queue size shouldn't keep growing!)
		print(velocity_repr, 'dt: %.2f' % dt, 'qsize: %02d' % samples_queue.qsize(), end='\r')

		# Write theta values to text file, this is defo not the right way but whatevs we'll fix it l8r
		# thetas_txt.truncate(0)
		# thetas_txt.write("-1 -1 -1")

