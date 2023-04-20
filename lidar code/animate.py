import numpy as np

from matplotlib import animation
from matplotlib import pyplot as plt

raw_data = np.load('walking.npy', allow_pickle=True)

class interpolator:
    def __init__(self, memory_size=2048):
        self.memory_size = memory_size
        self.last_samples = np.empty((memory_size, 2), dtype=np.float32) # TODO: start this gracefully
        self.last_samples[:] = np.nan
    def insert(self, sample):
        self.last_samples = np.roll(self.last_samples, 1, axis=0)
        self.last_samples[0] = sample
    def insert_many(self, samples):
        length = len(samples)
        self.last_samples = np.roll(self.last_samples, length, axis=0)
        self.last_samples[:length] = samples
    def generate_interpolation(self):
        epsilon = 0.25 # parameter for Gaussian RBF
        interpolation = np.zeros(360)
        for i in range(self.memory_size):
            if np.isnan(self.last_samples[i, 0]):
                continue
            angle = self.last_samples[i, 0]
            distance = self.last_samples[i, 1]

            input_angle = np.arange(360)-angle
            input_angle[input_angle > 180] -= 360
            input_angle[input_angle < -180] += 360

            rbf = epsilon*np.exp(-(epsilon*(np.arange(360)-angle))**2)/12
            interpolation += rbf*distance
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

interp = interpolator()
vel_est = velocity_estimator()
interp.insert_many(samples[0])

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