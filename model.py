# In machine learning, the function we are trying to solve is typically represented as Wx+b = y,
#  where we are given x (the list of input images) and y (the list of corresponding steering instructions),
# and want to find the best combination of W and b to make the equation balance.
import tensorflow as tf
import tensorflow.compat.v1 as tf  # to avoid tf.placeholder error
# It switches all global behaviors that are different between TensorFlow 1.x and 2.x to behave as intended for 1.x.
tf.disable_v2_behavior()

#  weight-variable and bias_variable : to build out the softmax layer


def weight_variable(shape):
    # The weight matrix is initialized using random values following a (truncated) normal distribution.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # converts constant value to tensor
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# performs kernel convolutions on input manipulating images to highlight some of its characteristics
# x=image itself, stride=how to manipulate image, W= set of coeff to blend image manipulations together


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


# placeholders to set x and y's dimensions
# x=image aayegi
x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
# setting up y to receive steering angle as an output
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

# first convolutional layer

W_conv1 = weight_variable([5, 5, 3, 24])  # 5x5 image size , 24 features
# 24 to help with matrix multiplication
b_conv1 = bias_variable([24])
# Since we're using ReLU neurons initialize them with a slightly positive initial bias to avoid "dead neurons".
# stddev=0.1
# input data has a shape of (batch_size, height, width, depth); depth=1: grayscale image(RGB BASICALLY)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])
# use output from prev layer
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

# third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

# fourth convolutional layer
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

# fifth convolutional layer
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

# Fully connected 1
W_fc1 = weight_variable([1152, 1164])
b_fc1 = bias_variable([1164])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
# softmax layer
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # to avoid overfitting

# FCL 2
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# FCL 3
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# FCL 4
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

# Output (readout layer: softmax; 1 feature)
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable([1])

y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5),
                2)  # scale the atan output
