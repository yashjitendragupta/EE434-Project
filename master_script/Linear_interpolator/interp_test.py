import interpolator
import numpy as np
import matplotlib.pyplot as plt



test = np.load('sample.npy')
print(len(test[:,0]))
test = test[:,1:3]
# test[:,0] = np.deg2rad(test[:,0])
print(test)
fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
ax1.scatter(np.deg2rad(test[:,0]), test[:,1])

# test = np.array([[0,0],[180, 1],[359,0]])

out = interpolator.interpolator(test)
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
ax2.scatter(np.deg2rad(range(360)),out)

print(out)

plt.show()

