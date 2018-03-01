
# coding: utf-8

# In[6]:

# coding: utf-8

import csv
import cv2
import numpy as np
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert it to hsv
    
    hsv = np.array(hsv, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2]*random_bright
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def trans_image(image, steer, trans_range):
    rows,cols = image.shape[:2] 
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr, steer_ang

def transform_image(img, angle, rot_ang) :
    rows,cols = img.shape[:2] 
    
    # Rotation
    #rot = np.random.uniform(rot_ang) - rot_ang/2
    #RM = cv2.getRotationMatrix2D((cols/2,rows/2), rot, 1)
    #img = cv2.warpAffine(img,RM,(cols,rows))

    # random shift 
    img, angle = trans_image(img, angle, 100)
    
    # Brightness 
    img = brightness(img)
    
    return img, angle

def process_image(image):
   shape = image.shape
   image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
   #print(image.shape)
   image = cv2.resize(image,(200, 66), interpolation=cv2.INTER_AREA)
   image =  cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
   return image

def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield them to the
    model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            if not validation_flag:
                img, angle = transform_image(img, angle, 15)
            img = process_image(img)
            X.append(img)
            y.append(angle)
            # flip horizontally and invert steer angle, if magnitude is > 0.1
            if  len(X) != batch_size and abs(angle) > 0.1:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)


lines = []
with open('./MyDataV5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

correction = [0, 0.25, -0.25]
image_paths = []
measurements = []
for line in lines:
    if (line[3] == 'steering') :
        continue
    measurement = float(line[3])
    #c = np.random.randint(3)
    c = 0
    if (measurement != 0) :
        if (np.random.choice(2, 1)[0]):
            c = 1
            if (measurement < 0) :
                c = 2  
    measurements.append(measurement+correction[c])
    source_path = line[c]
    filename = source_path.split('\\')[-1]
    #filename = source_path.split('/')[-1]
    image_path = './MyDataV5/IMG/'+filename
    image_paths.append(image_path)

image_paths = np.array(image_paths)
measurements = np.array(measurements)
print(len(image_paths))

# In[27]:

import matplotlib.pyplot as plt

num_bins = 23
avg_samples_per_bin = len(measurements)/num_bins
hist, bins = np.histogram(measurements, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()


# In[7]:


# split into train/test sets
image_paths_train, image_paths_valid, angles_train, angles_valid = train_test_split(image_paths, measurements,
                                                                                  test_size=0.2, random_state=42)
print('Train:', image_paths_train.shape, angles_train.shape)
print('Valid:', image_paths_valid.shape, angles_valid.shape)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66,200,3)))
#model.add(Cropping2D(cropping=((40,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation="relu"))

model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="relu"))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#print(X_train.shape)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=5)

# initialize generators
train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
val_gen = generate_training_data(image_paths_valid, angles_valid, validation_flag=True, batch_size=64)

s_per_epoch = ((int)(len(angles_train) / 64) + 1) * 64
model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=len(angles_valid), samples_per_epoch=s_per_epoch,
                    nb_epoch=5, verbose=1)

print(model.summary())

model.save('model.h5')


# In[ ]:



