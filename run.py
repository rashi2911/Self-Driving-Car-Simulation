import os
from subprocess import call
import cv2
import model
import tensorflow as tf
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
    #to resize image 
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[
        0][0] * 180.0 / 3.14159265
    if not windows:
        call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", full_image)
    # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    # and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
        degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    # make the transformation matrix M which will be used for rotating a image. 
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
    #performs an affine transformation to an image.
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
