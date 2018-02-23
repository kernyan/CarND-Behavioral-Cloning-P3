# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

#### Executive Summary: 
We employed a modified version of NVidia's End-to-End Neural Network pipeline to train a game car simulator to drive based purely on its camera inputs. No manually crafted image features were used. We showed that our model achieved a satisfactory performance on a test simulation run. This report includes the steps we took in designing and training our behavioral cloning model.

Outline of the report:
1. Introduction
2. Description of simulation environment
3. Data collection and preprocessing
4. Model architecture
5. Training and validation
6. Testing
7. Summary
8. Appendix


[//]: # (Image References)

[image1]: ./examples/IMG1.jpg "Centre"
[image2]: ./examples/IMG2.jpg "Left"
[image3]: ./examples/IMG3.jpg "Right"
[image4]: ./examples/Histogram_sample.png "Histogram_sample_only"
[image5]: ./examples/Histogram_sample_recovery.png "Histogram_sample_and_recovery"
[image6]: ./examples/RightSide.jpg "Orginal_RightSide"
[image7]: ./examples/LeftSide.jpg "Flipped_LeftSide"
[image8]: ./examples/Histogram_final_dataset.png "Histogram_final"
[image9]: ./examples/model_architecture.png "Model_architecture"
[image10]: ./examples/MSE_Behavioral_Cloning.png "MSE_loss"

---

#### 1. Introduction

Udacity's Car Simulator is a program which allows frames of joystick/keyboard controlled car to be captured along with steering angle measurements. In addition, it also provides an 'autonomous mode' which uses prediction output from a trained model (which was fed streams of frames) to test drive the car. The communication between the prediction model and simulator occurs in realtime with the back and forth between raw images and steering prediction.

The goal of this project is to be able to train a prediction model solely by cloning the behavior of car camera images and human-controlled steering angle pairs. The prediction model is then tested in a car simulator environment. Our final model successfully completed an autonomous driving lap of track 1 while staying within the track surface the whole time.

#### 2. Description of simulation environment

Below is a sample image from the simulation track 1

Centre
![alt text][image1] 

Left
![alt text][image2] 

Right
![alt text][image3]

The three cameras are views of the environment from the car's three mounted cameras. One for centre, left, and right. The test track contains various features such as 
1. bridge
2. river
3. road of various texture/color
4. left/right curves of varying degrees

#### 3. Data collection and preprocessing

Each time step provides three captured images (from the three cameras) and its corresponding throttle, steering angle, brake, and speed. However, in this report, we only utilize steering angle in our training. The autonomous model is specified to always accelerate at the start and maintain at a constant speed throughout the track.

The training data we used are composed of two parts,
1. Udacity's sample [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) - which contains 24108 images of the car making 4 laps of track 1 while mostly maintaining the car in centre of the path.
2. Augmented recovery dataset - which contains 4380 images of the car recovering from the left and right side of the path throughout different locations of track 1.

The histograms below show the distribution of the images we have with respect to steering angles

![alt text][image4] ![alt text][image5]

We see that the Udacity sample dataset is centered around 0 with little extreme values. Our experimentations indicated that this itself was insufficient in correcting the car when it's moderately off position. We generated additional 4380 recovery image-steering pairs which is roughly 15% of the total dataset size. 

We noticed that the histogram is not symmetric in that we have different number of positive and negative measurements. This might bias our model towards predicting a steering angle with signs simply because they were more frequent in the training dataset. We addressed this by further augmenting our dataset by flipping all the images and assigning the opposite sign of steering angles. For example,

![alt text][image6] ----Flip---> ![alt text][image7]

To utilize the left and right camera images, we include those images in our dataset by introducing them as pairs of 
1. (left_camera_image,  centre steering angle + rightwards correction)
2. (right_camera_image, centre steering angle + leftwards correction)
Our experimentation led us to choose a correction angle of +/- 0.20.

The total number of images we used are:
56976 = (24108 from Udacity sample dataset + 4380 from recovery dataset) * (2 from flipping * 3 from three camera angles) 

The histogram of the final augmented dataset is

![alt text][image8]

The steps used to augment data can be found in both the functions `GetNextBatch` and `GetNextBatchFlip` in [utlis.py](./utils.py)

We used an 80/20 split on the dataset for training/validation. Lastly, we used the following two Keras layers to prepare our images for training (see `NVIDIANet` function in [utils.py](./utils.py))
1. We cropped away the top 60 and bottom 20 pixels to focus the training on only the track.
2. Normalize using a lambda layer by dividing 255 and subtracting 0.5

#### 4. Model architecture

The final model architecture used is a similar to NVidia's End-to-End model \[1\]. The differences we made were
1. Adding a cropping layer to exclude the top 60 and bottom 20 pixels
2. Using only a single layer of Dropout (after the second convolutional layer)
3. Input size of 160x320x3 instead of 66x200x3
4. Using (2,2) stride for all 5 convolutional layers instead of only the first 3 convolutional layers

![alt text][image9]

We chose the NVidia's End-to-End model as a starting point because it had worked in a real world situation. We kept much of the model architecture because it worked well on our test track without much additional tweaking. Our implementation of the model can be found in `NVIDIANet` function in [utils.py](./utils.py).

#### 5. Training and validation

##### 5.1 Generator

Given that there were a total of 56796 of 160x320x3 images, they would take around 8.7 GB just to load the images. To prevent such memory load, we implemented a generator function which processes the dataset in batches. Our function `GetNextBatch` and `GetNextBatchFlip` takes in a CSV file and a batch size and returns subsequent sets of images according to the batch size.

##### 5.2 Training steps

As we realize that Mean-Squared-Error (mse) is not a good indicator of track performance (we had lots of low mse models that would just drive in constant circles), we relied less on mse in model selection. mse was used mainly as an indicator that training and validation exhibit a downward trend.

The process we used in both parameter selection and model architecture was primarily iterative. We would tweak one thing at a time and check our model's performance. We did close to 35 iterative steps before settling on the final model. If a tweak improves our model, we kept it and proceed, otherwise we remove the tweak and try something else (or change its parameter). We provide a trimmed down log of our process

1. NVidiaNet without modification - Model drives straight with mild steering. Fails at first turn
2. Add normalization - Model drives only in clockwise circles
3. Remove normalization, add dropout - Model drives only on the right side of the curb
4. Remove dropout, change strides parameter - Model drives reasonably straight. Fails at bridge
....
35. Model successfully finished test track.

Throughout our experimentation, the important notes we learned are that
1. Augmenting with recovery data is necessary - otherwise the model will not survive sharp turns
2. Normalization - it has to be used with a large enough number of epoch, otherwise the model may have a large bias (i.e. giving predictions that are not responsive to images)
3. Dropout - similar to normalization, we noticed that by adding a dropout layer, we have to increase the number of epochs trained to several fold. Otherwise we notice that the model will exhibit large bias. However, dropout does improve the model's robustness in that the drive is smoother (cf [video without dropout](./examples/video_nodropout.mp4) and [final video](./examples/video.mp4))
4. We needed to introduce L2Norm regularizer to alleviate the model's tendency for constant clockwise or counterclockwise turns.
5. Using an Adam optimizer relieves us of the need to search for a learning rate

The final parameter set used are

| Parameters         		     |  Description	  | 
|:-------------------------------|:--------------:| 
| Left/Right camera correction   |  +/- 0.2  	  | 
| L2Norm Regularizer    	     | 0.0008	      |
| Activation function	         | elu            |
| Number of dropout layer needed | 1              |
| Dropout rate 					 | 0.5            |
| Epochs needed	      	         |  12	          |
| Strides of all 5 conv2d	     | (2,2)          |
| Normalize	image			     |	  Yes	   	  |
| Learning rate  			     |Adam Optimizer  |

To run the model, place training images in ./data/IMG and execute
```python
python model.py
```
Our final training/validation loss curve is

![alt text][image10]

#### 6. Testing

To try the model on the car simulator

```python
./linux_sim/linux_sim.x86_64 # to start the simulator
python drive.py model.h5     # to initiate realtime communication between prediction model and simulator
```
A video of the model's performance is included below

[Video](./examples/video.mp4)

The top speed in which the model still manages to drive the track well is 25.

#### 7. Summary

We reiterate our findings in the following bullets,

1. NVidia architecture is a good starting model for car behavioral cloning.
2. Recovery data and dropout layer are crucial for our model's success
3. End-to-end vehicle modeling is possible as we demonstrated. However we could benefit greatly if we implemented some form of network visualization to enable us to understand the model's steering decision. We look forward to implementing NVidia's salient object identification technique \[2\] in a follow up work. This would allow us to mask the deconvoluted activations of our model on our input images.
4. We also wanted to pursue further work for training the model by adding lighting and shadow augmentation so that our model can work on test track 2.

#### 8. Appendix

#### 8.1. References

1. Mariusz Bojarski, Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, and Karol Zieba. End to end learning for self-driving cars. April 25 2016. URL: [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)
2. Mariusz Bojarski, Philip Yeres, Anna Choromanaska, Krzysztof Choromanski, Bernhard Firner, Lawrence Jackel, Urs Muller. Explaining how a deep neural network trained with end-to-end learning steers a car. April 25 2017. URL: [https://arxiv.org/pdf/1704.07911.pdf](https://arxiv.org/pdf/1704.07911.pdf)

#### 8.2. Related files

1. [model.py](./model.py) - Python files which trains based on images in ./data/IMG
2. [utils.py](./utils.py) - Python files which includes all utility functions used in model.py (e.g. architecture model and image preprocessing) 
3. [drive.py](./drive.py) - Udacity provided socketio file which allows communication between simulation program (to send images to Keras prediction model) and Keras model (to relay back predicted steering angle to the simulator). We modified the `set_speed` constant to 25 instead of the default 9
4. [model.h5](./model.h5) - Saved final Keras model with weights
5. [video.mp4](./examples/video.mp4) - mp4 of final model on test track 1
6. [video_nodropout.mp4](./examples/video_nodropout.mp4) - mp4 of model without dropout on test track 1

