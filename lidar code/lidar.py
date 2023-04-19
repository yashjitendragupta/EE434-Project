            
import numpy as np    
import matplotlib.pyplot as plt
import math as m
from rplidar import RPLidar



def get_data():    
    lidar = RPLidar("COM3", baudrate=115200)
    for scan in lidar.iter_scans(max_buf_meas=1000000):    
        break    
        lidar.stop()    
    return scan

current_data = get_data() 

#%%
arr = np.asarray(current_data)
fig = plt.figure()
arr[:,1] = arr[:,1]*2*m.pi/360
ax = fig.add_subplot(projection='polar')
plt.scatter(arr[:,1],arr[:,2])



