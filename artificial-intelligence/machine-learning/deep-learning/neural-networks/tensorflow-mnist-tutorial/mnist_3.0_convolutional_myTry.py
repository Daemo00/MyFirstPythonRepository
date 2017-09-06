# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Neural Network
# neurons with sigmoid  (y = 1 / (1 + exp(-x)))
# neurons with relu     (y = max(0, x))
# neurons with softmax  (y = exp(x) / |exp(x)|)
#
# · · · · · · · · · ·       (input data)       X [batch, 28, 28]
# \x/x\x/x\x/x\x/x\x/    -- convolutional layer, 4 channels, stride 1
#   · · · · · · · ·         W1[5, 5, 1, 4]      B1[4]       Y1[batch, 28, 28, 4]
#   \x/x\x/x\x/x\x/      -- convolutional layer, 8 channels, stride 2
#     · · · · · ·           W2[4, 4, 4, 8]      B2[8]       Y2[batch, 14, 14, 8]
#     \x/x\x/x\x/        -- convolutional layer, 12 channels, stride 2
#       · · · ·             W3[4, 4, 8, 12]     B3[12]      Y3[batch, 7, 7, 12]
#       \x/x\x/          -- fully connected layer
#         · ·               W4[7*7*12, 200]     B4[200]     Y4[batch, 200]
#         \x/            -- fully connected layer
#          ·                W5[200, 10]         B5[10]      Y5[batch, 10]

# input [100, 28*28]:  100 grayscale images of 28x28 pixels, flattened (a mini-batch equals to 100 images)
# output[100, 10]:     the probability that each image is a number, hot-encoded
#
# Fully connected model from K to L neurons is:
# Y = softmax( X * W + b)
#       X[batch, K]:    input
#       W[K, L]:        weight matrix
#       b[L]:           bias vector
#       +:              add with broadcasting: adds the vector to each line of the matrix (numpy)
#       Y[batch, L]:    output matrix
#
# Convolutional model from [m, n] neurons with stride s is:
# Y = cnv(X, W, s) + b
#       X[batch, m, n]:         input
#           m1, n1:             dimension of filter
#           i:                  input channels (layers in previous cluster)
#           o:                  output channels (layers of current cluster)
#       W[m1, n1, i, o]:        weight matrix
#       b[o]:                   bias vector
#       +:                      add with broadcasting: adds the vector to each line of the matrix (numpy)
#       Y[batch, m/s, n/s, o]:  output matrix

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
images_size = 28
answers = 10
X = tf.placeholder(tf.float32, [None, images_size, images_size, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, answers])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# Convolutional layer from [m, n] neurons, stride s:
#   - input:   X[batch, m, n]
#   - weights: W[m1, n1, i, o]
#   - biases:  B[o]
#   - output:  Y[batch, m/s, n/s]
# weights and biases
filters = [[6, 6], [5, 5], [4, 4]]
inputs = [1, 6, 12]
outputs = [6, 12, 24]
W, B = [], []
for f, i, o in zip(filters, inputs, outputs):
    W.append(tf.Variable(tf.truncated_normal([*f, i, o], stddev=0.1)))
    B.append(tf.Variable(tf.ones([o]) / 10))
W1, W2, W3 = W[0], W[1], W[2]
B1, B2, B3 = B[0], B[1], B[2]

strides = [1, 2, 2]
mul_strides = 1
Ylist = [X]
for i, s in enumerate(strides):
    Ycnv = tf.nn.conv2d(Ylist[-1], W[i], strides=[1, s, s, 1], padding='SAME')
    Ylist.append(tf.nn.relu(Ycnv + B[i]))
    mul_strides *= s
Y1, Y2, Y3 = Ylist[1], Ylist[2], Ylist[3]
# stride = 1
# Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
# stride = 2  # output is 14x14
# Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
# stride = 2  # output is 7x7
# Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

layer3_size = int(images_size / 4)
W4 = tf.Variable(tf.truncated_normal([layer3_size * layer3_size * outputs[-1], 200], stddev=0.1))
B4 = tf.Variable(tf.ones([200]) / 10)
YY3 = tf.reshape(Y3, [-1, layer3_size * layer3_size * outputs[-1]])
Y4 = tf.nn.relu(tf.matmul(YY3, W4) + B4)

W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10]) / 10)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualisation
all_weights = tf.concat(
    [tf.reshape(W1, [-1]),
     tf.reshape(W2, [-1]),
     tf.reshape(W3, [-1]),
     tf.reshape(W4, [-1]),
     tf.reshape(W5, [-1])], 0)
all_biases = tf.concat(
    [tf.reshape(B1, [-1]),
     tf.reshape(B2, [-1]),
     tf.reshape(B3, [-1]),
     tf.reshape(B4, [-1]),
     tf.reshape(B5, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, all_weights, all_biases],
                                  feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(
            c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})


# datavis.animate(training_step, iterations=10000 + 1, train_data_update_freq=10, test_data_update_freq=50,
#                 more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
for j in range(10000+1):
    training_step(j, j % 500 == 0, j % 100 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
