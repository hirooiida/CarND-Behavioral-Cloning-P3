import csv
import cv2
import numpy as np

log_file = '/opt/carnd_p3/data/driving_log.csv'
img_dir = '/opt/carnd_p3/data/IMG/'

lines = []
with open(log_file) as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.2

for i in range(3):
    for line in lines:
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = img_dir + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
        # center camera
        if i == 0:
            measurement = float(line[3])
        # left camera
        elif i == 1:
            measurement = float(line[3]) + correction
        # right camera
        elif i == 2:
            measurement = float(line[3]) - correction
        else:
            print("Out of range")
        
        measurements.append(measurement)
        
        # Flipping
        images.append(cv2.flip(image,1))
        measurements.append(measurement * (-1.0))
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Cropping2D
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
history=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
