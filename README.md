# EE434-Project

Kenny P, Yash G, Oliver H. DSP lab project

## Setup

The following setup is untested, but it generally covers:

1. Installing numpy and scipy dependencies
2. Compiling and installing numpy and scipy from source
3. Downloading the Nvidia Jetson Nano distribution of PyTorch
4. Installing PyTorch
5. Installing other dependencies

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config
sudo apt-get install -y python3-pip python3-dev
pip3 install Cython
pip3 install numpy scipy\<=1.6

wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

pip3 install PyAudio rplidar pyrplidar
```

## Running

Starting from the project root directory, the main routine can be started by the command `cd master_script && python3 main.py`

There are also test routines in this project:

* A demo of the motion detection functions: `cd master_script && motion.py`
* A live motion tracking routine: `cd master_script && lidar.py`
* A live passthrough pipeline from the UMA-8 microphone array to the ALSA loopback device `cd master_script && audio.py`

## Project Tree

In the below project tree, the remaining unexplained but significant files and directories are:

* `HRTF`: a dataset of HRTF impulse responses
* `lidar code`: contains initial work on LiDAR, including the generated data file `walking.npy` that is used in the `motion.py` demo
* `old_kenny_experiments`: contains initial work on the UMA-8 microphone array
* `old_yash_experiments`: contains initial work in MCRoomSim


```
EE434-Project/
├── environment.yml
├── finalReport.tex
├── HRTF
│   ├── elev0
│   ├── elev-10
│   ├── elev10
│   ├── elev-20
│   ├── elev20
│   ├── elev-30
│   ├── elev30
│   ├── elev-40
│   ├── elev40
│   ├── elev50
│   ├── elev60
│   ├── elev70
│   ├── elev80
│   └── elev90
├── lidar code
│   ├── animate.py
│   ├── lidar.py
│   ├── lidarwork.py
│   └── walking.npy
├── master_script
│   ├── audio.py
│   ├── class_defs.py
│   ├── lidar.py
│   ├── Linear_interpolator
│   ├── main.py
│   ├── motion.py
│   └── psuedocode.md
├── old_kenny_experiments
│   ├── uma8_collected_patterns
│   ├── uma8_fracdelay_recovery
│   ├── uma8_recovery
│   ├── uma8_sampling
│   ├── uma8_theoretical_patterns
│   └── virtualization_experiment
├── old_yash_experiments
│   ├── beamforming_sim_experiment_2
│   ├── Beamforming simulation experiment
│   └── Crosstalk_Isolation_experiment
└── README.md
```