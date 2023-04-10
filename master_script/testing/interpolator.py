import numpy as np
# takes in np array thetas in first col and r in second col
def interpolator(pairs):
	out = np.empty([360,1])
	for i in range(360):
		left_bound = pairs[-1,:]
		for pair in pairs:
			if(pair[0] >= i):
				right_bound = pair 
				break
			left_bound = pair
		# print(left_bound, right_bound)
		left_dist = i - left_bound[0]
		right_dist = right_bound[0] - i
		if(left_dist < 0):
			left_dist += 360
		if(right_dist < 0):
			right_dist += 360
		deg_range = left_dist + right_dist
		left_weight = 1 - left_dist/deg_range
		right_weight = 1 - right_dist/deg_range
		out[i] = left_weight*left_bound[1] + right_weight*right_bound[1]
		# print(i,out[i,1]) 

	return out



