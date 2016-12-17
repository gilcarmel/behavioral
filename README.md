## Steering Angle Prediction Using Behavioral Cloning

![alt text](https://github.com/gilcarmel/behavioral/blob/master/sample_images/driving.gif "Drive!")

For this project, I designed and trained a neural network that can steer a simulated car based on an image from its front-facing camera. The network was trained using _behavioral cloning_ - a technique that teaches the model to mimic an observed behavior. In this case, the network learned to mimic a human driver's steering angle driving around two different tracks.

### Data Collection
I collected training data (images and their corresponding steering angles) by driving the simulated car for a few laps down the center of the lane. I added a lot of examples of recentering the car from both the left and right shoulder. This is necessary to help the model recover when the car inevitably starts to drift (a model trained purely on center-lane driving would not know how to recover).
I used a gaming controller to drive, so that the training data contained smooth steering angles (as opposed to the discrete values of -1, 0, and 1 generated by driving with the keyboard). 

In total I collected about 6000 images from track 1 and 7000 images from track 2.

I found it hard to control the car perfectly and some bad driving examples definitely snuck in and polluted my training data. For production models, I would need an easy way to discard bad training data - perhaps through a visual editor tool. 

Here are some examples from the training set. Steering angles range from -1 (hard right turn) to 1 (hard left turn):


![alt text](https://github.com/gilcarmel/behavioral/blob/master/sample_images/center_0.0.jpg "Center-lane driving (steering angle 0)")

_Center-lane driving (requires steering angle 0)_


![alt text](https://github.com/gilcarmel/behavioral/blob/master/sample_images/recover_left_0.70.jpg "Recover from left (steering angle 0.7)")

_Recover from left (requires steering angle 0.7)_

![alt text](https://github.com/gilcarmel/behavioral/blob/master/sample_images/recover_right_-0.38.jpg "Recover from right (steering angle -0.38)")

_Recover from right (requires steering angle -0.38)_


### Preprocessing

Input images are preprocessed prior to training and inference:
* Scaling/cropping: images were scaled down to 30% scale, and cropped the top 30% (mostly sky). After visually examining the scaled/cropped images, I suspected they still conveyed enough information to make a correct steering prediction (this was confirmed in practice later). The smaller images dramatically improved network training time, allowing for more experimentation with preprocessing, architecture, and hyperparemeters. It also allowed all the training data to fit in memory, obviating the need to stream it in using a generator.
* Horizontal flipping: The data was collected driving one way around the looped track, resulting in a much higher proportion of left turns vs right turns. After noticing that the model was performing fairly well on left turns but poorly on sharp right turns, I doubled the data set by flipping each image and negating its corresponding steering angle.
* YUV conversion: My models seemed to perform better with YUV images (vs RGB). I found this surprising - I would think the convolutional layers of my model would in a way "figure out" the correct color space. I wasn't methodical in verifying the improved performance vs RGB - so I might be overstating the effect.

### Network Architecutre
After initially trying [NVIDIA's network structure](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with the original image size (320x160), I decided to try a simpler network to reduce training times. My intuition was that NVIDIA's model had to deal with a much wider variety of environments and lighting conditions than our simple simulator.

I began removing convolutional layers from the NVIDIA architecture. The smaller input image meant we did not need as many convolution layers before getting down to a short, wide, deep image (12x4, depth 36) that could be reasonably fed to a fully connected layer. I intuited that this wide short image could easily convey all the information we needed to make a single steering angle prediction in the range [-1,1]. I would love to see a visualization of this layer's activation - it might very well look like a stereo equalizer lighting up more or less according to the desired turn' sharpness. I ended up with the following architecture:

    (input 96x48x3) -->
      (crop to 96x34x3) --> 
	     (5x5 conv to 48x17x24) --> (maxpool to 24x8x24) -> relu -> dropout(0.2) -->
            (3x3 conv to 24x8x36) --> (maxpool to 12x4x36) -> relu -> dropout(0.2) -->
               (flatten) --> (fully connected to 100) --> dropout(0.2) --> relu -->
    	           (flatten) --> (fully connected to 10) --> relu -->
    	              (fully connected to 1) --> prediction:

Here is the summary as printed by Keras:

    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    cropping2d_1 (Cropping2D)        (None, 34, 96, 3)     0           cropping2d_input_1[0][0]         
    ____________________________________________________________________________________________________
    convolution2d_1 (Convolution2D)  (None, 17, 48, 24)    1824        cropping2d_1[0][0]               
    ____________________________________________________________________________________________________
    maxpooling2d_1 (MaxPooling2D)    (None, 8, 24, 24)     0           convolution2d_1[0][0]            
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 8, 24, 24)     0           maxpooling2d_1[0][0]             
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 8, 24, 24)     0           activation_1[0][0]               
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 8, 24, 36)     7812        dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    maxpooling2d_2 (MaxPooling2D)    (None, 4, 12, 36)     0           convolution2d_2[0][0]            
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 4, 12, 36)     0           maxpooling2d_2[0][0]             
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 4, 12, 36)     0           dropout_2[0][0]                  
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 1728)          0           activation_2[0][0]               
    ____________________________________________________________________________________________________
    hidden1 (Dense)                  (None, 100)           172900      flatten_1[0][0]                  
    ____________________________________________________________________________________________________
    dropout_3 (Dropout)              (None, 100)           0           hidden1[0][0]                    
    ____________________________________________________________________________________________________
    activation_3 (Activation)        (None, 100)           0           dropout_3[0][0]                  
    ____________________________________________________________________________________________________
    hidden3 (Dense)                  (None, 10)            1010        activation_3[0][0]               
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 10)            0           hidden3[0][0]                    
    ____________________________________________________________________________________________________
    output (Dense)                   (None, 1)             11          activation_4[0][0]               
    ====================================================================================================
    Total params: 183557
    ____________________________________________________________________________________________________


### Training
I used a train/test split of 95%/5% and a train/validation split of 80%/20%. The loss function was the mean square error between the prediction and the steering angle. 

The simplified network and reduced image size resulted in training time of about 3 seconds per 26k-image epoch on an AWS p2 instance, allowing me to try various combinations of hyperparameters (batch size & learning rate). The loss appeared to converge after about 100 epochs. A few things of note:
* I used the mean absolute error as the accuracy metric (to get a rough idea of a model's potential during training, before connecting it to the simulator). Track 1 trains from a mean error of about 0.7 down to about 0.4 for a successful model. Track 2 trains down from 0.20 to 0.1 - perhaps because this track feature sharper turns. Mean error seemed to correlate roughly, though not perfectly, with real performance on the track (i.e the lowest-error model was not necessarily the best performer on the track). I'd be curious to learn more about accuracy metrics for this type of prediction task. 
* I fixed the random seed, hoping to make training deterministic to help with hyperparameter tuning. This didn't seem to work - training proceeded differently each time, even with no changes to the model or data. This made hyperparameter tuning more difficult as it was hard to attribute improved performance to parameter values (vs randomness).
* Hyperparameter tuning was done very ad-hoc, until it was "good enough". For example, after trying several different learning rates in the range 0.00001 - 0.0002, I settled on 0.00005 because it performed best with the architecture at the time. I stuck with that learning rate despite tweaking the architecture several times afterwards. A more formal approach to parameter search would probably yeild better results. 

### Performance

After a lot of trial and error, I finally got a model that made it all all the way through both tracks. But it was a bit swervy, and would sometimes go on the shoulder during sharp turns:

![alt text](https://github.com/gilcarmel/behavioral/blob/master/sample_images/swerve.gif "Swerve!")

To improve performance, I made three adjustments:
* Collected more training data for sharp turns, to better distribute the steering angles in the training set 
* Increased dropout from 0.2 to 0.4, to avoid overfitting to bad data that snuck into the training set
* Increased the number of epochs to 200 after noticing that accuracy was still improving after 100 epochs.

We these edits, the car was much less erratic, swerving less and staying closer to center lane.

A final note - I noticed that my model does not do as well for graphics settings that are different than the training images (640x480, "Fastest" graphics). This indicates that the model to some extent memorized unrelated features in the training set. To fix this, I could try augmenting the training data set by adjusting various image settings (brightness, saturation, etc), and by adding images captured on different graphics settings.

Here's the [full video](https://youtu.be/BFMoccwHGjc), going around both tracks.


