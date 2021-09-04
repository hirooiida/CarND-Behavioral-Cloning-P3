# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nvidia-cnn.png "Model Visualization"
[image2]: ./images/left_2016_12_01_13_44_32_011.jpg "Left Camera"
[image3]: ./images/center_2016_12_01_13_44_32_011.jpg "Center Camera"
[image4]: ./images/right_2016_12_01_13_44_32_011.jpg "Right Camera"
[image5]: ./images/center_2016_12_01_13_44_32_011_flipped.jpg "Flipped Image"
[image6]: ./images/history.png "Training History"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVIDIA convolutional neural network model according to the suggestion in Udacity course and [NVIDIA blog](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). The input image size is changed from original 66x220 to 90x320, which is the size after cropping of 160x320. The model can be found at line 53 to 64 in model.py. The NVIDIA model looks like this:

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

I decided not to make any changes to NVIDIA model since it is considered as a proven model. The epoch number is decided to avoid overfitting according to the learning history analysis. See "3. Creation of the Training Set & Training Process" section for more details.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

The sample data provided by Udacity is used for training.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I spent some time with LeNet model first during getting familiar with the project and moved to NVIDIA model soon. The model didn't work immediately but after increasing training data by flipping (line 43 - 44 in model.py), adding left and right camera (line 31 - 36) and converting color space from BGR to RGB made the car run through the course.

#### 2. Final Model Architecture

As mentioned earlier, the model is based on NVIDIA model except to the size of input. The output from `model.summary()` looks like this:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 18, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 8, 64)          36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               51300     
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 51        
=================================================================
Total params: 187,749
Trainable params: 187,749
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

Udacity sample dataset consists of the multiple laps of images in clockwise and anti-clock wise with center, left, right images.

![alt text][image2]
![alt text][image3]
![alt text][image4]

I increased the dataset even more by flipping the images. For example,

![alt text][image3]
![alt text][image5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the diagram below. The error loss for validation dataset represented as "val_loss" is saturated after 7th training. The error loss for training dataset represented as "loss" improves even after but it is considered as overfitting.

![alt text][image6]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
