
#!/usr/bin/env python3
'''Records scans to a given file in the form of numpy array.
Usage example:
$ ./record_scans.py out.npy'''
import sys
import numpy as np
import time as t
from rplidar import RPLidar


PORT_NAME = '/dev/ttyUSB0'


def run():
    '''Main function'''
    lidar = RPLidar("COM3", baudrate=115200)
    arr = []
    data = []
    times = []
    start = t.perf_counter()
    try:
        print('Recording measurments... Press Crl+C to stop.')
        for scan in lidar.iter_scans():
            data.append(np.array(scan))
            times.append(np.array(t.perf_counter() - start))
    except KeyboardInterrupt:
        print('Stoping.')
    lidar.stop()
    lidar.disconnect()
    return data,times

    
    
if __name__ == '__main__':
    data,times = run()
    sample = list(zip(data,times))
    np.save("walking",sample)
    