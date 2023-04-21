import numpy as np

from matplotlib import animation
from matplotlib import pyplot as plt

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

class velocity_estimator:
    def __init__(self):
        self.h_prev = np.empty_like(360)
    def estimate(self, h, dt):
        dh_dt = (h-self.h_prev)/dt
        
        dx = 1/(2*np.pi)
        dh_dx = np.empty_like(h)
        dh_dx = (2*h+np.roll(h, -1)-np.roll(h, 1))/(2*dx)

        v = -dh_dt/dh_dx
        self.h_prev = h

        return v

times = raw_data[:, 1]
samples = raw_data[:, 0]
samples = [sample[:, 1:] for sample in samples]
average_interval = np.mean(np.diff(times))

print(f'Loaded {len(times)} frames with an average interval of {average_interval}')

interp = gaussian_rbf_interpolator(512, 3.0)
vel_est = velocity_estimator()

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
line, = ax.plot(np.arange(360)/180*np.pi, 10000*np.abs(initial_vel), 'ro')
line2, = ax.plot(samples[0][:, 0]/180*np.pi, samples[0][:, 1], 'bo')
line3, = ax.plot(np.arange(360)/180*np.pi, initial_interpolation, 'go')

def animate(i):
    interp.insert_many(samples[i])
    interpolation = interp.generate_interpolation()

    line.set_data(np.arange(360)/180*np.pi, 10000*np.abs(vel_est.estimate(interpolation, average_interval)))
    line2.set_data(samples[i][:, 0]/180*np.pi, samples[i][:, 1])
    line3.set_data(np.arange(360)/180*np.pi, interpolation)
    return [line, line2, line3]
anim = animation.FuncAnimation(fig, animate, frames=len(samples), interval=1000*average_interval, blit=True)

plt.show()