import numpy as np

from matplotlib import animation
from matplotlib import pyplot as plt

MEMORY_SIZE = 512
RBF_EPSILON = 4.5
RBF_RCOND = 1e-3
LUCAS_KANADE_WINDOW_SIZE = 24

raw_data = np.load('walking.npy', allow_pickle=True)

# using Gaussian RBF interpolation for now
# epsilon is RBF width, letting rcond -> 0 is more accurate  but less stable
class gaussian_rbf_interpolator: 
    def __init__(self, memory_size, epsilon=3.0, rcond=1e-3):
        self.memory_size = memory_size
        self.memory = np.full((memory_size, 2), fill_value=np.nan, dtype=np.float32)
        self.epsilon = epsilon
        self.rcond = rcond
    
    def insert_many(self, samples):
        n_samples = len(samples)
        self.memory[n_samples:] = self.memory[:-n_samples]
        self.memory[:n_samples] = samples
    
    def generate_interpolation(self):
        n_valid = np.count_nonzero(~np.isnan(self.memory[:, 0]))

        observed_angles = self.memory[:n_valid, 0]*np.pi/180
        rbfs = np.empty((n_valid, n_valid))
        for i, observed_angle in enumerate(observed_angles):
            rbfs[:, i] = np.exp(-(self.epsilon*(observed_angles-observed_angle))**2)
        
        rbs_pseudoinverse = np.linalg.pinv(rbfs, rcond=self.rcond)
        observed_distances = self.memory[:n_valid, 1]
        weights = rbs_pseudoinverse @ observed_distances

        grid_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        interpolation = np.zeros_like(grid_angles, dtype=np.float32)
        for observed_angle, weight in zip(observed_angles, weights):
            rbf = np.exp(-(self.epsilon*(grid_angles-observed_angle))**2)
            interpolation += weight*rbf
        
        return interpolation

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
        h_meters = h/1000 # convert to meters
        dtheta = 2*np.pi/360
        
        dh_dt = (h_meters-self.h_prev)/dt
        dh_dtheta = (np.roll(h_meters, -1)-np.roll(h_meters, 1))/(2*dtheta)

        v_est = np.empty(360, dtype=np.float32)
        for i in range(360):
            window = np.arange(i-self.window_size, i+self.window_size+1)
            window[window < 0] += 360
            window[window >= 360] -= 360

            dh_dtheta_local = dh_dtheta[window]
            dh_dt_local = dh_dt[window]

            # in 1D, the pseudoinverse reduces to a row vector
            pseudoinverse = dh_dtheta_local.T/np.sum(dh_dtheta_local**2)

            v_est[i] = -pseudoinverse @ dh_dt_local
        
        self.h_prev = h_meters
        return v_est

times = raw_data[:, 1]
samples = raw_data[:, 0]
samples = [sample[:, 1:] for sample in samples]
average_interval = np.mean(np.diff(times))

print(f'Loaded {len(times)} frames with an average interval of {average_interval}')

interp = gaussian_rbf_interpolator(MEMORY_SIZE, epsilon=RBF_EPSILON, rcond=RBF_RCOND)
vel_est = angular_velocity_estimator(window_size=LUCAS_KANADE_WINDOW_SIZE)

interp.insert_many(samples[0])
plt.plot(range(360), interp.generate_interpolation())
plt.scatter(samples[0][:, 0], samples[0][:, 1])
plt.title('Initial interpolation vs actual samples')
plt.show()

initial_interpolation = interp.generate_interpolation()
vel_est.estimate(initial_interpolation, average_interval)
initial_vel = vel_est.estimate(initial_interpolation, average_interval)

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
line, = ax.plot(np.arange(360)/180*np.pi, 4000*np.abs(initial_vel), 'ro') # 4000 is just for visualization
line2, = ax.plot(samples[0][:, 0]/180*np.pi, samples[0][:, 1], 'bo')
line3, = ax.plot(np.arange(360)/180*np.pi, initial_interpolation, 'go')

def animate(i):
    interp.insert_many(samples[i])
    interpolation = interp.generate_interpolation()
    velocity_estimation = vel_est.estimate(interpolation, average_interval)

    line.set_data(np.arange(360)/180*np.pi, 4000*np.abs(velocity_estimation))
    line2.set_data(samples[i][:, 0]/180*np.pi, samples[i][:, 1])
    line3.set_data(np.arange(360)/180*np.pi, interpolation)
    return [line, line2, line3]
anim = animation.FuncAnimation(fig, animate, frames=len(samples), interval=1000*average_interval, blit=True)

plt.show()