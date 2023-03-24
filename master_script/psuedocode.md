# ideas for main script

## Getting the required data from the function libraries

### Theta Values

We will probably have to run the lidar script constantly, then write the specified theta values to a simple file which this main script can read when it's ready to beamform

### Audio buffer

The main script should basically run once all 7 of the pyaudio buffers are full of data from the microphone array. Per iteration, it will pass the thetas and buffers to a function written in a * *separate* * file