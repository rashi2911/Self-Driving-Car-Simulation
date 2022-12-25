import model  # Importing model.py file.
import driving_data  # Importing the driving_data.py file.
from tensorflow.core.protobuf import saver_pb2
# The OS module in Python provides functions for creating and removing a directory (folder), fetching its contents, changing and
import os
# identifying the current directory, etc.
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.InteractiveSession()
# This is L2 Normalization which is used to calculate the error of the model. There is L1 Normalization too.
L2NormConst = 0.001

train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + \
    tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

epochs = 30
batch_size = 100

# train over the dataset about 30 times
for epoch in range(epochs):
    for i in range(int(driving_data.num_images/batch_size)):
        x_list, y_list = driving_data.TrainBatch(batch_size)
        train_step.run(feed_dict={model.x: x_list,
                       model.y_: y_list, model.keep_prob: 0.8})
        if i % 10 == 0:
            x_list, y_list = driving_data.testBatch(batch_size)
            loss_value = loss.eval(
                feed_dict={model.x: x_list, model.y_: y_list, model.keep_prob: 1.0})
            print("Epoch: %d, Step: %d, Loss: %g" %
                  (epoch, epoch * batch_size + i, loss_value))
