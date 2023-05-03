# motion.py
# given the raw samples taken from the RPLiDAR...
#  [1] pass those samples into the interpolator's memory using insert_many()
#  [2] get the interpolator's output using generate()
#  [3] pre-filter the interpolator's output by passing it through filter()
#  [4] get velocity estimates by passing the filter's output into estimate()
#  [5] get motion labels by passing the velocity estimates into detect_motion()
#  [6] get avg angle of each label by passing the labels into find_motion()
# where i-th element corresponds to the average position of the i-th detected 
# motion, counting counter-clockwise from the 0-degree position

import numpy as np
import torch
from scipy.signal import butter

from matplotlib import animation
from matplotlib import pyplot as plt

WALKING_NPY_PATH = '../lidar code/walking.npy'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class interpolator:
    def __init__(self, memory_size=512, meters=True, radians=True):
        self.memory_size = memory_size
        self.angles_memory = torch.full((memory_size, ), fill_value=np.nan, device=device)
        self.heights_memory = torch.full((memory_size, ), fill_value=np.nan, device=device)

        self.meters = meters
        self.radians = radians
    
    def insert_many(self, angles, heights):
        n_samples = len(angles)

        # copy the data to the GPU then do some transforms there
        angles_gpu = torch.from_numpy(angles).float().to(device)
        heights_gpu = torch.from_numpy(heights).float().to(device)
        if self.radians:
            angles_gpu = angles_gpu*np.pi/180
        if self.meters:
            heights_gpu = heights_gpu/1000

        if n_samples > self.memory_size:
            self.angles_memory = angles_gpu[-self.memory_size:]
            self.heights_memory = heights_gpu[-self.memory_size:]
        else:
            self.angles_memory = torch.roll(self.angles_memory, n_samples)
            self.angles_memory[:n_samples] = angles_gpu
            self.heights_memory = torch.roll(self.heights_memory, n_samples)
            self.heights_memory[:n_samples] = heights_gpu
    
    def generate(self):
        grid_angles = np.arange(360, dtype=np.float32)
        if self.radians:
            grid_angles *= np.pi/180
            epsilon = 4.5
        else:
            epsilon = 0.1
        
        angles = self.angles_memory[~torch.isnan(self.angles_memory)]
        heights = self.heights_memory[~torch.isnan(self.heights_memory)]

        outer_subtraction = angles.reshape(-1, 1)-angles.reshape(1, -1)
        rbfs = torch.exp(-(epsilon*outer_subtraction)**2)
        rbfs_pseudoinverse = torch.linalg.pinv(rbfs, rcond=1e-3)
        weights = rbfs_pseudoinverse @ heights

        angles = angles.cpu().numpy()
        weights = weights.cpu().numpy()

        interpolation = np.zeros_like(grid_angles, dtype=np.float32)
        for angle, weight in zip(angles, weights):
            rbf = np.exp(-(epsilon*(grid_angles-angle))**2)
            interpolation += weight*rbf
        
        return grid_angles, interpolation


class low_pass_filter:
    def __init__(self, n_channels, corner_freq=1.25, sampling_freq=7):
        b, a = butter(2, corner_freq, btype='low', fs=sampling_freq)
        self.num = b
        self.den = a

        self.y_1 = np.zeros(n_channels)
        self.y_2 = np.zeros(n_channels)
        self.x_1 = np.zeros(n_channels)
        self.x_2 = np.zeros(n_channels)
    
    def filter(self, x_t):
        y_t = self.num[0]*x_t + self.num[1]*self.x_1 + self.num[2]*self.x_2 \
                - self.den[1]*self.y_1 - self.den[2]*self.y_2

        self.x_2 = self.x_1
        self.x_1 = x_t
        self.y_2 = self.y_1
        self.y_1 = y_t

        return y_t


# though velocity estimation by using optical flow is actually fully defined in 
#  1d, we can apply the Lucas-Kanade method anyway to get a less noisy estimated 
#  velocity by using the assumption of constant angular velocity within a 
#  neighborhood/window? It's just the first thing that looked okay
# to see the difference, try a window size of 0 because this reverts the Lucas- 
#  Kanade method back to the solution of the fully defined 1D problem
class angular_velocity_estimator:
    def __init__(self, window_size):
        self.h_prev = np.zeros(360)
        self.window_size = window_size

    def estimate(self, h, dt):
        dtheta = 2*np.pi/360
        
        dh_dt = (h-self.h_prev)/dt
        dh_dtheta = (np.roll(h, -1)-np.roll(h, 1))/(2*dtheta)

        dh_dtheta_neighbors = np.empty((360, 2*self.window_size+1), dtype=np.float32)
        dh_dt_neighbors = np.empty((360, 2*self.window_size+1), dtype=np.float32)
        for j in range(2*self.window_size+1):
            shift = j-self.window_size
            dh_dtheta_neighbors[:, j] = np.roll(dh_dtheta, shift)
            dh_dt_neighbors[:, j] = np.roll(dh_dt, shift)
        
        # calculates all estimated velocities as many dot products
        elementwise_product = np.multiply(dh_dtheta_neighbors, dh_dt_neighbors)
        v_est = -np.sum(elementwise_product, axis=1)/np.sum(dh_dtheta_neighbors**2, axis=1)
        
        self.h_prev = h
        return v_est


def detect_motion(v, upper_threshold=0.20, lower_threshold=0.025): # upper-lower thresholds to configure hysteresis
    motion_counter = 0
    motion_detected = False
    labels = np.empty_like(v, dtype=np.int32)
    for i, v_theta in enumerate(v):
        if np.abs(v_theta) > upper_threshold:
            motion_detected = True
        elif motion_detected and np.abs(v_theta) < lower_threshold:
            motion_detected = False
            motion_counter += 1
        else:
            pass
        
        if motion_detected:
            labels[i] = motion_counter
        else:
            labels[i] = -1
    
    return labels

def find_motion(labels):
    n_labels = np.max(labels)+1 # +1 because labels start at 0
    indicator = np.zeros_like(labels)
    for i_label in range(n_labels):
        label_indices = np.argwhere(labels == i_label)
        mean_index = np.mean(label_indices)
        mean_index = int(mean_index)
        indicator[mean_index] = i_label+1 # +1 because labels start at 0
    return indicator


if __name__ == '__main__':
    raw_data = np.load(WALKING_NPY_PATH, allow_pickle=True)
    times = raw_data[:, 1]      # list of times corresponding to ...
    samples = raw_data[:, 0]    # ... (:, 3) arrays of arbitary length (first column is meaningless)

    test_set = np.vstack([samples[0], samples[1]])
    test_angles = test_set[:, 1]
    test_heights = test_set[:, 2]

    interp = interpolator(memory_size=512, meters=True, radians=True)
    filt = low_pass_filter(360, corner_freq=0.5)
    vel_est = angular_velocity_estimator(window_size=16)

    interp.insert_many(test_angles, test_heights)
    angle_grid, heights = interp.generate()
    plt.scatter(test_angles, test_heights)
    plt.plot(np.arange(0, 360, 1), 1000*heights) # convert back to mm and degrees
    plt.show()

    angle_grid = np.linspace(0, 2*np.pi, 360, endpoint=False)
    def output_next_samples():
        for sample in samples:
            interp.insert_many(sample[:, 1], sample[:, 2])
            angle_grid, heights = interp.generate()
            filtered_heights = filt.filter(heights)
            velocities = vel_est.estimate(filtered_heights, 1/7)
            labels = detect_motion(velocities)
            indicator = find_motion(labels)
            yield heights, filtered_heights, 10*np.abs(velocities), indicator

    samples = list(output_next_samples())
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    artists = []
    artists.append(ax.plot([], [], label='interpolated_heights', marker='o', linestyle='')[0])
    artists.append(ax.plot([], [], label='filtered_heights', marker='o', linestyle='')[0])
    artists.append(ax.plot([], [], label='10*np.abs(velocities)', marker='o', linestyle='')[0])
    artists.append(ax.plot([], [], label='indicator', marker='o', linestyle='')[0])

    ax.set_ylim(0, 7)
    ax.set_xlim(0, 2*np.pi)
    ax.legend()

    def animate(i):
        for artist, sample in zip(artists, samples[i]):
            artist.set_data(angle_grid, sample)
        return artists
    
    anim = animation.FuncAnimation(fig, animate, frames=len(samples)-1, interval=100, blit=True)
    plt.show()