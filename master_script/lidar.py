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
from time import time, sleep
import math as m
import motion
sz=24
locations = np.zeros(24)
start_time = time()
cnt = 0
# Using the RPLidar class as a context manager ensures the serial comms are always closed, even on crashes
class RpLidarContext(RPLidar):
	def __enter__(self):
		return self
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()
		self.disconnect()

# thetas_txt = open("thetas.txt","w")

interp = motion.interpolator(memory_size=256, meters=True, radians=True)
filt = motion.low_pass_filter(360, corner_freq=0.5, sampling_freq=8) # so far, the best sampling freq observed is about 8 Hz
vel_est = motion.angular_velocity_estimator(window_size=16)

with RpLidarContext('/dev/ttyUSB0', baudrate=115200) as lidar:
	samples_queue = Queue()
	
	# the queuing rplidar uses seems flaky, so we'll run our own queue and monitior the queue's size
	def lidar_routine():
		global samples_queue

		try:
			samples_batch = []
			for samples in lidar.iter_scans(max_buf_meas=512):
				samples_batch.extend(samples)
				if len(samples_batch) >= 128:
					samples_queue.put(samples_batch) # pass samples into samples_queue
					samples_batch = []
		except Exception as e: # if an error is encountered, pass that error to the main thread
			samples_queue.put(e)

	frame_rate = 0
	shared_heights = np.zeros(360, dtype=np.float32) # shared between interpolation_thread and main thread
	ready_flag = False

	# since the interpolation routine primarily uses torch (which releases the GIL), we can use an asynchronous thread
	def interpolation_routine():
		global frame_rate, shared_heights, ready_flag, samples_queue
		
		prev_ts = time()
		while True:
			samples = samples_queue.get(timeout=3)
			if isinstance(samples, Exception): # TODO: raise this exception in the main thread
				raise samples
			
			samples = np.array(samples)
			samples = samples[::2] # downsample by 2
			angles = samples[:, 1]
			distances = samples[:, 2]

			interp.insert_many(angles, distances)
			angle_grid, shared_heights = interp.generate()

			ready_flag = True

			current_ts = time()
			frame_rate = 1/(current_ts-prev_ts)
			prev_ts = current_ts

	lidar_thread = Thread(target=lidar_routine, daemon=True)
	lidar_thread.start()
	interpolation_thread = Thread(target=interpolation_routine, daemon=True)
	interpolation_thread.start()

	prev_ts = time()
	while True:
		current_time = time()-start_time
		# collect dt as a difference between timestamps
		current_ts = time()
		dt = current_ts - prev_ts
		prev_ts = current_ts

		# Do processing to find the three theta values

		while not ready_flag:
			sleep(0.010)
		ready_flag = False
		
		filtered_heights = filt.filter(shared_heights)
		velocities = vel_est.estimate(filtered_heights, dt)
		labels = motion.detect_motion(velocities)
		indicator = motion.find_motion(labels)

		# for testing and visualization, represent velocities as an integer between 0 and 9
		maxpooled_velocity = np.max(np.abs(velocities).reshape(-1, 10), axis=1).reshape(-1)
		velocity_repr = (10*maxpooled_velocity).astype(np.int64)
		velocity_repr[velocity_repr < 0] = 0
		velocity_repr[velocity_repr > 9] = 9

		# report the velocity representation, dt, and queue size (queue size shouldn't keep growing!)
		print(locations, 'dt: %.2f frame_rate: %01.1f qsize: %02d' % (dt, frame_rate, samples_queue.qsize()), end='\r')
		if(current_time > 30):
			for i in range(0,360):
				k = m.floor(i * (sz/360))
				if(indicator[i] > 0):
					if locations[k] == 0:
						cnt = cnt + 1
						locations[k] = cnt
						for j in range(1,3):
							if(k-j<0):
								if(locations[(k-j)+sz] > 0):
									locations[k] = locations[(k-j)+sz]
									cnt = cnt-1
									locations[(k-j+sz)] = 0
									
							else:
								if(locations[k-j] > 0):
									locations[k] = locations[k-j]
									cnt = cnt-1
									locations[k-j] = 0
							if(k+j>sz-1):
								if(locations[(k+j)-sz] > 0):
									locations[k] = locations[(k+j)-sz]
									cnt = cnt-1
									locations[(k+j)-sz] = 0
							else:
								if(locations[k+j] > 0):
									locations[k] = locations[k+j]
									cnt = cnt-1
									locations[k+j] = 0
							
			
	    		
		# Write theta values to text file, this is defo not the right way but whatevs we'll fix it l8r
		# thetas_txt.truncate(0)
		# thetas_txt.write("-1 -1 -1")

