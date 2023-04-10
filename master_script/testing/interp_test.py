import interpolator
import numpy as np

test = np.array([[0,0],[180, 1],[359,0]])

out = interpolator.interpolator(test)

print(out)
