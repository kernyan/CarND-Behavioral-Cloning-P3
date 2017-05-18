# CarND-Behavioral-Cloning-P3
Udacity CarND Program Term1 Assignment 3 - Behavioral Cloning

This repository is about implementing a behavioral cloning neural network on a car driving simulator

Please refer to [writeup](./writeup_report.md) for further details

This repository contains the following:
1. [model.py](./model.py) - Python files which trains based on images in ./data/IMG
2. [utils.py](./utils.py) - Python files which includes all utility functions used in model.py (e.g. architecture model and image preprocessing) 
3. [drive.py](./drive.py) - Udacity provided socketio file which allows communication between simulation program (to send images to Keras prediction model) and Keras model (to relay back predicted steering angle to the simulator). We modified the `set_speed` constant to 25 instead of the default 9
4. [model.h5](./model.h5) - Saved final Keras model with weights
5. [video.mp4](./examples/video.mp4) - mp4 of final model on test track 1
6. [video_nodropout.mp4](./examples/video_nodropout.mp4) - mp4 of model without dropout on test track 1

