import cv2
import random
import numpy as np

# Putting the images in a list with their corresponding steering angles in y_list.
x_list = []
y_list = []

train_batch = 0
test_batch = 0

with open("driving_dataset/data.txt") as f:
    for line in f:
        # Putting images in x_list.
        x_list.append("driving_dataset/" + line.split()[0])
        y_list.append(float(line.split()[1]) * 3.14159265 / 180)

# Number of images
num_images = len(x_list)

# Shuffling the images, keeping their steering angles intact.
data = list(zip(x_list, y_list))
random.shuffle(data)
x_list, y_list = zip(*data)

# Splitting the dataset into training and testing in 80:20 ratio.
train_x_list = x_list[:int(len(x_list) * 0.8)]
train_y_list = y_list[:int(len(x_list) * 0.8)]

test_x_list = x_list[-int(len(x_list) * 0.2):]
test_y_list = y_list[-int(len(x_list) * 0.2):]

num_train_images = len(train_x_list)
num_test_images = len(test_x_list)


def TrainBatch(batch_size):
    # Accessing the variable from the outside the function.
    global train_batch
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Taking each image of the training set and resizing it to 200 x 66 and finally normalizing the pixel values and appending it in list.
        x_out.append(cv2.resize(cv2.imread(
            train_x_list[(train_batch + i) % num_train_images])[-150:], (200, 66)) / 255.0)
        y_out.append([train_y_list[(train_batch + i) % num_train_images]])
    train_batch += batch_size
    return x_out, y_out


def testBatch(batch_size):
    global test_batch
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Same as above.
        x_out.append(cv2.resize(cv2.imread(
            test_x_list[(test_batch + i) % num_test_images])[-150:], (200, 66)) / 255.0)
        y_out.append([test_y_list[(test_batch + i) % num_test_images]])
    test_batch += batch_size
    return x_out, y_out
