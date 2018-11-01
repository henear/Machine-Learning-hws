# Part 0: Setup

import tensorflow as tf
import numpy as np
import util
import os
from scipy import io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load the data we are giving you
# def load(filename, W=28, H=28):
#     data = np.fromfile(filename, dtype=np.uint8).reshape((-1, W*H*3+1))
#     images, labels = data[:, :-1].reshape((-1,H,W,3)), data[:, -1]
#     return images

# image_data, label_data = load('digits.dat')
trainDataNum = 60000
allData = sio.loadmat("digits.mat")
# image_data = allData.get("trainImages").reshape((60000, 28, 28, 1))
image_data = np.zeros((trainDataNum, 28, 28, 1))
for i in range(trainDataNum):
  a = allData.get("trainImages")[:,:,:,i]
  if i == 0:
    print(a.shape)
  image_data[i,:,:,:] = a
label_data = allData.get("trainLabels").reshape(trainDataNum)
# label_data = np.ndarray.flatten(label_data)
print("---")
print(type(image_data))
print(type(label_data))
print("~~~")
print('Input shape: ' + str(image_data.shape))
print('Labels shape: ' + str(label_data.shape))

num_classes = 10

# Lets clear the tensorflow graph, so that you don't have to restart the notebook every time you change the network
tf.reset_default_graph()
# Part 1: Define your convnet
# Make sure the total number of parameters is less than 100,000.

# Set up your input placeholder
inputs = tf.placeholder(tf.float32, (None,28,28,1), name='input')

# Whenever you deal with image data it's important to mean center it first and subtract the standard deviation
white_inputs = (inputs - np.mean(image_data)) / np.std(image_data)

# Set up your label placeholders
labels = tf.placeholder(tf.int64, (None), name='labels')

# Step 1: define the compute graph of your CNN here
#   Use 5 conv2d layers (tf.contrib.layers.conv2d) and one pooling layer tf.contrib.layers.max_pool2d or tf.contrib.layers.avg_pool2d.
#   The output of the network should be a None x 1 x 1 x 6 tensor.
#   Make sure the last conv2d does not have a ReLU: activation_fn=None
conv1 = tf.contrib.layers.conv2d(
  inputs=white_inputs, # normalized version
  num_outputs=9,      # 20 kernel matrices
  kernel_size=[4, 4],  #
  stride=2,
  padding="same",     # valid(original dimension), same 
  scope='conv1')      # name variable for input and output

conv2 = tf.contrib.layers.conv2d(
inputs=conv1,
num_outputs=12,
kernel_size=[4, 4],
stride=2,
padding="same",
scope='conv2')

conv3 = tf.contrib.layers.conv2d(
    inputs=conv2,
    num_outputs=15,
    kernel_size=[4, 4],
    stride=2,
    padding="same",
    scope='conv3')

conv4 = tf.contrib.layers.conv2d(
  inputs=conv3, # normalized version
  num_outputs=18,      # 20 kernel matrices
  kernel_size=[4, 4],  #
  stride=2,
  padding="same",     # valid(original dimension), same 
  scope='conv4')      # name variable for input and output
conv5 = tf.contrib.layers.conv2d(
    inputs=conv4,
    num_outputs=21,
    kernel_size=[4, 4],
    stride=2,
    padding="same",
    scope='conv5')

# pool1 = tf.contrib.layers.max_pool2d(inputs=conv5, kernel_size=[2, 2], stride=2, scope='pool1')
# pool1_flat = tf.reshape(pool1, [-1, np.prod(pool1.get_shape().as_list()[1:])])
dense = tf.contrib.layers.fully_connected(inputs=conv5, num_outputs=1024, activation_fn=tf.nn.relu)
h = tf.contrib.layers.fully_connected(inputs=conv5, num_outputs=10)
# The input here should be a   None x 1 x 1 x 6   tensor
output = tf.identity(tf.contrib.layers.flatten(h), name='output')

# Step 2: use a classification loss function (from assignment 3)
loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = output)
# Step 3: create an optimizer (from assignment 3)
optimizer1 = tf.train.MomentumOptimizer(0.0001, 0.9)
# Step 4: use that optimizer on your loss function (from assignment 3)
minimizer1 = optimizer1.minimize(loss1)
# Step 5: calculate some metrics
correct = tf.equal(tf.argmax(output, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]), '/', 100000 )


# Part 2: Training
# Training might take up to 20 min depending on your architecture. This time around you should get close to 100% trianing accuracy.


# Batch size
BS = 32

# Start a session
sess = tf.Session()

# Set up training
sess.run(tf.global_variables_initializer())

# This is a helper function that trains your model for several epochs un shuffled data
# train_func should take a single step in the optmimzation and return accuracy and loss
#   accuracy, loss = train_func(batch_images, batch_labels)
# HINT: train_func should call sess.run
def train(train_func):
    # An epoch is a single pass over the training data
    for epoch in range(20):
        # Let's shuffle the data every epoch
        np.random.seed(epoch)
        np.random.shuffle(image_data)
        np.random.seed(epoch)
        np.random.shuffle(label_data)
        # Go through the entire dataset once
        accs, losss = [], []
        for i in range(0, image_data.shape[0]-BS+1, BS):
            # Train a single batch
            batch_images, batch_labels = image_data[i:i+BS], label_data[i:i+BS]
            acc, loss = train_func(batch_images, batch_labels)
            accs.append(acc)
            losss.append(loss)
        print('[%3d] Accuracy: %0.3f  \t  Loss: %0.3f'%(epoch, np.mean(accs), np.mean(losss)))


# Train convnet
print('Convnet')
### Your Code Here (from assignment 3) ### 
def mytrainfunc(image_data, label_data):
    _, acc, loss = sess.run([minimizer1, accuracy, loss1], feed_dict={inputs: image_data, labels: label_data})
    return acc, loss

train(mytrainfunc)
# Part 3: Evaluation
# Show the current graph
util.show_graph(tf.get_default_graph().as_graph_def())


# Compute the valiation accuracy
# The convnet still massively overfits. We will deal with this in assignment 5.
testDataNum = 10000
image_val = np.zeros((testDataNum, 28, 28, 1))
for i in range(testDataNum):
  a = allData.get("testImages")[:,:,:,i]
  if i == 0:
    print(a.shape)
  image_val[i,:,:,:] = a
label_val = allData.get("testLabels").reshape(testDataNum)
# label_val = np.ndarray.flatten(label_val)

print('Input shape: ' + str(image_val.shape))
print('Labels shape: ' + str(label_val.shape))

val_accuracy, val_loss = sess.run([accuracy, loss1], feed_dict={inputs: image_val, labels: label_val})
print("ConvNet Easy Validation Accuracy: ", val_accuracy)


