# In this file, constantly poll the LIDAR device
# then find the three theta positions, then write those
# thetas to some text file that the main script can read from.
# runs at the same time as the main script.
# writes -1 to file if error or does not exist
# writes between 0 to 360 for angle that does exist.

thetas_txt = open("thetas.txt","w")

while True:

	# read array from LIDAR

	# Do processing to find the three theta values

	# Write theta values to text file, this is defo not the right way but whatevs we'll fix it l8r
	thetas_txt.truncate(0)
	thetas_txt.write("-1 -1 -1")

