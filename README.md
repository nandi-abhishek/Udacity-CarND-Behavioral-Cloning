
# **Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project** 

---

*My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.*

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: nVidia_model.png "nVidia CNN Model"
[image2]: my_model.png "My CNN model"
[image3]: original_hist.png "Original Data Distribution"
[image4]: augmented_hist.png "Partially augmented data"
[image5]: brightness.png "Brightness modification"
[image6]: translate.png "Random Translation"
[image7]: cropped.png "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 video of autonomous driving using my model in track 1
* track2.mp4 video of autonomous driving using my model in track 2
* bunch of images for this writeup

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) - the diagram below is a depiction of the nVidia model architecture.

![nVidia CNN Model][image1]

#### 2. Attempts to reduce overfitting in the model

I initially used dropout layers. But was not getting good performance. Then changed to  L2 regularization (lambda of 0.001) to all model layers - convolutional and fully-connected as discussed in the forum. To reduce overfitting, the trained and validated on different data sets.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 173). I have played arund with the L2 regularization lamda ana epochs. Finally, set them to 0.001 and 20 respectively.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have used my own data instead of data provided by Udacity, I used the first track and second track data. Also I have used images from all three cameras (center, left and right) to train my model.

For details about how I created the training data, see the next section. 

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The simulated car is equipped with three cameras, one to the left, one in the center and one to the right of the driver that provide images from these different view points. The first track has sharp corners, exits, entries, bridges and changing light conditions. An additional track exists with changing elevations, even sharper turns. It is thus crucial that the CNN is generalized enough to drive autonously on these varity of track condition. My model is trained using both the tracks and then it is able to drive them autonomously.

The main problem lies in the skew and bias of the data set. Shown below is a histogram of the steering angles recorded while driving in the middle of the road for a few laps. This is also the data used for training. The left-right skew is less problematic and can be eliminated by flipping images and steering angles simultaneously. However, even after balancing left and right angles most of the time the steering angle during normal driving is small or zero and thus introduces a bias towards driving straight. The most important events however are those when the car needs to turn sharply.

![Original Data Distribution][image3]

Without accounting for this bias towards zero, the car leaves the track quickly. I have used varity of ideas to mitigate the problem described in 'Creation of Training Set' section below.

#### 2. Final Model Architecture

The final model architecture (model.py lines 159-179) consisted of a convolution neural network which is similar to nVidia's model. Here is a visualization of the architecture 

![My Model][image2]

The model first uses a Keras Lambda layer for normalization . hen I have three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - as described in the Nvidia paper text - including converting from RGB to YUV color space, and 2x2 striding on the 5x5 convolutional layers. The model includes RELU layers to introduce nonlinearity (model.py line 159-171). 

Since the top portion of the image has mountain, sky etc and bottom portion has car hood I thought of cropping the image by 1/4 from top and 25 pixel from bottom. Otherwise these portion istead of helping would distract the model. Some, example cropped images are displayed below:

![Cropped Image][image7]

Then, before feeding the image to the model the image is resized to 66x200 and then passed to the model.

#### 3. Data Augementation 
Augmentation helps generate powerful models even when dataset is small. The following techniques were used to generate augmented dataset

##### Brigtness Adjustment 
Changed brightness to simulate day and night conditions. I have generated images with different brightness by first converting images to HSV, and randomly scaling up or down the V channel and converting back to the BGR channel.

![Brightness modification][image5]

##### Horizontal and vertical shifts

I have shifted the camera images horizontally to simulate the effect of car being at different positions on the road, and add an offset corresponding to the shift to the steering angle. I have added 0.004 steering angle units per pixel shift to the right, and subtracted 0.004 steering angle units per pixel shift to the left. I have also shifted the images vertically by a random number to simulate the effect of driving up or down the slope.

This modification was inspired by [Vivek's model](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9).

![Random Translation][image6]

Apart from these to remove the skewed data as described above I have used

##### Flipping

For any images with absolute turning angle greater than 0.1 added a flipped image and changed the sign of angle to  simulate driving in the opposite direction.

##### Using left and right camera images

To further add data with nonzero driving angle I have randomly added left or right camera images for images with non zero angle. Before adding those images to test set their abgles are properly adjusted with an offset (0.25 for left camera and -0.25 for right camera image). Note the code also keeps the center image for that instance.

##### Keras generator for subsampling

When working with datasets that have a large memory footprint (large quantities of image data, in particular) Keras python generators are a convenient way to load the dataset one batch at a time rather than loading it all at once. I have used Keras  fit_generator() and generator code to feed batches of processed the images into the model. The way we apply the generator with augmentation is:

Generator > Pick out a batch of random images from the original training data > Apply random augmentation > Feed to the model to train > Model is done training with that batch, destroy the data > Repeat the process


#### 4. Creation of the Training Set & Training Process

My first raw training data was gathered by driving the car as smoothly as possible right in the middle of the road for 3 in one direction in the first track. I have also captured few manual recovery events at the turns. Othehr recovery was simulated using shifts, flip, left/right camera images etc.

The model performed well in the first track autonomously but failed terribly in the second track. Then I updated my training data by 2 laps of second track and one or two manual recovery from it. With this data set the model successfully drove the car in both tracks at speed 9.

But if I increase speed to 20+ it was failing in second track. I realised that the model is still suffereing from bias towards driving straight. So, I downsampled the samples with 0 steering angle to 25%. iven below is a histogram after down smapling (but with filpping - as flipping is done inside the generator a running time)

![Partially augmented data][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

### Result

After all these augmentation and tweeking my car was driving well in both tracks - In track1 it drove with speed 25 and in track 2 with speed 20.


