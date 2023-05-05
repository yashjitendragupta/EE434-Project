# In this file, constantly poll the LIDAR device
# then find the three theta positions, then write those
# thetas to some text file that the main script can read from.
# runs at the same time as the main script.
# writes -1 to file if error or does not exist
# writes between 0 to 360 for angle that does exist.

import numpy as np
from pyrplidar import PyRPlidar
from queue import Queue
from threading import Thread
from time import time, sleep
import math as m
import motion
sz=36
locations = np.zeros(36)
angles = np.zeros(4)
start_time = time()
cnt = 0

# Using the RPLidar class as a context manager ensures the serial comms are always closed, even on crashes
class RpLidarContext(PyRPlidar):
	def __init__(self, *args, **kwargs):
		self.connect(*args, **kwargs)
		self.set_motor_pwm(1023)
	def __enter__(self):
		return self
	def iter_scans(self):
		return self.start_scan_express(1)()
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()
		self.disconnect()

# thetas_txt = open("thetas.txt","w")

interp = motion.interpolator(memory_size=220, meters=True, radians=True)
filt = motion.low_pass_filter(360, corner_freq=0.5, sampling_freq=8) # so far, the best sampling freq observed is about 8 Hz
vel_est = motion.angular_velocity_estimator(window_size=16)

with RpLidarContext('/dev/ttyUSB0', baudrate=115200) as lidar:
	samples_queue = Queue()
	
	def lidar_routine():
		global samples_queue
		try:
			samples = []
			t = None
			for sample in lidar.iter_scans(): # this always yield about 200 samples
				if sample.quality != 0:
					samples.append((sample.quality, sample.angle, sample.distance))
				if len(samples) >= 440:
					samples_queue.put(samples)
					samples = []
		except Exception as e: # if an error is encountered, pass that error to the main thread
			samples_queue.put(e)

	frame_rate = 0
	heights_queue = Queue()

	# since the interpolation routine primarily uses torch (which releases the GIL), we can use an asynchronous thread
	def interpolation_routine():
		global frame_rate, samples_queue, heights_queue
		
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
			angle_grid, heights = interp.generate()

			heights_queue.put(heights)

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

		heights = heights_queue.get(timeout=9)
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
		#print(locations, 'dt: %.2f frame_rate: %01.1f qsize: %02d' % (dt, frame_rate, samples_queue.qsize()), end='\r')
		if(current_time > 30):
			for i in range(0,360):
				k = m.floor(i * (sz/360))
				if(indicator[i] > 0):
					if cnt == 3:
						if locations[k] == 0:
							for j in range(1,19):
								if(k-j<0):
									if(locations[(k-j)+sz] > 0):
										locations[k] = locations[(k-j)+sz]
										
										locations[(k-j+sz)] = 0
										break
										
								else:
									if(locations[k-j] > 0):
										locations[k] = locations[k-j]
										
										locations[k-j] = 0
										break
								if(k+j>sz-1):
									if(locations[(k+j)-sz] > 0):
										locations[k] = locations[(k+j)-sz]
										
										locations[(k+j)-sz] = 0
										break
								else:
									if(locations[k+j] > 0):
										locations[k] = locations[k+j]
										
										locations[k+j] = 0
										break
						
					else:

						if locations[k] == 0:
							cnt = cnt + 1
							locations[k] = cnt
							for j in range(1,3):
								if(k-j<0):
									if(locations[(k-j)+sz] > 0):
										locations[k] = locations[(k-j)+sz]
										cnt = cnt-1
										locations[(k-j+sz)] = 0
										break
										
								else:
									if(locations[k-j] > 0):
										locations[k] = locations[k-j]
										cnt = cnt-1
										locations[k-j] = 0
										break
								if(k+j>sz-1):
									if(locations[(k+j)-sz] > 0):
										locations[k] = locations[(k+j)-sz]
										cnt = cnt-1
										locations[(k+j)-sz] = 0
										break
								else:
									if(locations[k+j] > 0):
										locations[k] = locations[k+j]
										cnt = cnt-1
										locations[k+j] = 0
										break

			for	i in range(sz):	
				for j in range(1,4):
					if(locations[i] == j):
						angles[j] = i*(360/sz)
		print(angles[1],",",angles[2],",",angles[3])


	    		
		# Write theta values to text file, this is defo not the right way but whatevs we'll fix it l8r
		# thetas_txt.truncate(0)
		# thetas_txt.write("-1 -1 -1")

