# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
#dataset_path      = "/path/to/your/dataset/mnist/"
#test_labels_file  = "test-labels.csv"
#train_labels_file = "train-labels.csv"

test_set_size = 5

IMAGE_HEIGHT  = 240    #960
IMAGE_WIDTH   = 180    #720
NUM_CHANNELS  = 3
BATCH_SIZE    = 5

def encode_label(label):
  return int(label)

def read_label_file(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label = line.split(",")
    filepaths.append(filepath)
    labels.append(encode_label(label))
  return filepaths, labels

# reading labels and file path
train_filepaths, train_labels = read_label_file('/home/waleed/Desktop/leaf_data.csv')
#test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

# transform relative path into full path
#train_filepaths = [ dataset_path + fp for fp in train_filepaths]
#test_filepaths = [ dataset_path + fp for fp in test_filepaths]

# for this example we will create or own test partition
#all_filepaths = train_filepaths + test_filepaths
#all_labels = train_labels + test_labels

all_filepaths = train_filepaths[:]
all_labels = train_labels[:]

# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# create a partition vector
#partitions = [0] * len(all_filepaths)
#partitions[:test_set_size] = [1] * test_set_size
#random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
#train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
#train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

# create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [all_images, all_labels],
                                    shuffle=True)
#test_input_queue = tf.train.slice_input_producer(
#                                    [test_images, test_labels],
#                                    shuffle=False)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_image = tf.image.resize_images(train_image, 240, 180)
train_label = train_input_queue[1]
train_label = tf.cast( train_label, tf.int32 )
train_label = tf.one_hot( train_label, 4, 1, 0 )
train_label = tf.cast( train_label, tf.float32 )
#train_label.set_shape([4])
#print train_label
#file_content = tf.read_file(test_input_queue[0])
#test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
#test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
#test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
#test_image_batch, test_label_batch = tf.train.batch(
#                                    [test_image, test_label],
#                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
#                                    )

print "input pipeline ready"

################# MODEL BELOW

# Parameter
learning_rate = 0.001
training_iters = 1000
#batch_size = 1
display_step = 10

# Network Parameters
n_input = 240 * 180 # MNIST data input (img shape: 28*28)
n_classes = 4 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 240, 180, 3])
#y = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)



def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, 240, 180, 1])

    # Convolution Layer 1 with 2 strides
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print "pool5.shape:", conv1.get_shape()

    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print "pool5.shape:", conv1.get_shape()
    # Convolution Layer part 2

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print "pool5.shape:", conv2.get_shape()
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    print "pool5.shape:", conv2.get_shape()

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    conv2Shape = conv2.get_shape().as_list()
    print conv2Shape
    fc1 = tf.reshape(conv2, [-1, conv2Shape[1] * conv2Shape[2] * conv2Shape[3]])

    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([60*45*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
print ("pred is ",pred)
# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))   # should work without one-hot encoding and hence defining shape of y
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
print ("cost is ",cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1
    # Keep training until reach max iterations
    # start populating filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            print 'I have reached here'
            while step * BATCH_SIZE < training_iters:
                print ('iterating')
                batch_x, batch_y = sess.run([train_image_batch, train_label_batch])
                #batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                #batch_x = np.reshape(batch_x, (-1, 43200))
                print batch_y
                #batch_y = np.reshape(batch_y, (1, 4))
                #print batch_y
                print 'problem is after'
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
                print 'coming here'
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

    except tf.errors.OutOfRangeError:
        print('Done training --  epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
