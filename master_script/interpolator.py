import numpy as np
# takes in np array thetas in first col and r in second col
def interpolator(pairs):
	for i in range(360):
		left_bound = pairs[-1,:]
		for pair in pairs:
			if(pair[0] >= i):
				right_bound = pair 
				break
			left_bound = pair
		printf(left_bound, right_bound)
		


