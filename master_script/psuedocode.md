# ideas for main script

## Getting the required data from the function libraries

### Theta Values

We will probably have to run the lidar script constantly, then write the specified theta values to a simple file which this main script can read when it's ready to beamform

### Audio buffer

The main script should basically run once all 7 of the pyaudio buffers are full of data from the microphone array. This can be done in the main script

## Processing all the data

### Audio Separation

Per iteration, it will pass the thetas and buffers to a function written in a *separate* file where that function will then pass back three arrays back with the same length as the original buffer.

### virtualization

These three arrays, along with the ones before them, are then passed into a function which will pass back two stereo arrays with the same length as the buffer that contain the virtualized audio.

## Output

Finally, we output to a buffer which simulates a virtual microphone in linux which can be captured by Zoom or Audacity