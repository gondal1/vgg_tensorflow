import tensorflow as tf
import numpy as np
import os

IMAGE_HEIGHT  = 240    #960
IMAGE_WIDTH   = 180    #720
NUM_CHANNELS  = 3
BATCH_SIZE = 5
logs_path = './tensorflow_logs/new'
#=======================================================================================================
# Reading data from CSV FILE
#=======================================================================================================

# load csv content
csv_path = tf.train.string_input_producer(['/home/waleed/Desktop/leaf_data.csv'])
textReader = tf.TextLineReader()
_, csv_content = textReader.read(csv_path)
im_name, label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])

# load images, and convert labels into one_hot encoded form
im_content = tf.read_file(im_name)
image = tf.image.decode_jpeg(im_content, channels=3)
image = tf.cast(image, tf.float32) / 255. # could be unnecessary
image = tf.image.resize_images(image, 240, 180)
label = tf.one_hot(label, 4, 1, 0 )
label = tf.cast(label, tf.float32 )

# Make Batches of images with shuffling
min_after_dequeue = 10                                               # Defines how big a buffer we will randomly sample from -- bigger means better shuffling but slower start up and more memory used
capacity = min_after_dequeue + 3 * BATCH_SIZE                        # Capacity must be larger than min_after_dequeue and the amount larger determines the maximum we will prefetch.
train_image_batch, train_label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=BATCH_SIZE, capacity=capacity,
    min_after_dequeue=min_after_dequeue)


#====================================================================================================
# Model Hyperparameters, and characteristics
#=====================================================================================================
# Model Parameters
LEARNING_RATE = 0.001
TRAINING_ITERS = 500
DISPLAY_STEP = 5

# Network Parameters
N_CLASSES = 4
DROPOUT = 0.75                                                       # Dropout,probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 240, 180, 3])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
drop_out = tf.placeholder(tf.float32)                                # dropout (keep probability)

# Wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxPool(x, k=2, s=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')


# Model
def convolutionalNet(x, weights, biases, dropout):

    '''
    # 1. Convolution layer, max-pooling, stride=1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print "pool1.shape:", conv1.get_shape()
    pool1 = maxPool(conv1)
    print "pool1.shape:", pool1.get_shape()
    tf.histogram_summary("layer1", pool1)

    # 2. Second Convolutional layer, max-pooling, stride=1
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    print "conv2.shape:", conv2.get_shape()
    pool2 = maxPool(conv2)
    print "pool2.shape:", pool2.get_shape()
    tf.histogram_summary("layer2", pool2)

    # 3. Fully connected layer 1
    pool2Shape = pool2.get_shape().as_list()                           # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2, [-1, pool2Shape[1] * pool2Shape[2] * pool2Shape[3]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)                                   # Apply Dropout

    # 4. Output Layer
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    '''

    with tf.name_scope('conv1_1') as scope:
        conv1_1 = conv2d(x, weights['wc1'], biases['bc1'],2)
        tf.histogram_summary("conv1_1", conv1_1)
    with tf.name_scope('conv1_2') as scope:
        conv1_2 = conv2d(conv1_1, weights['wc2'], biases['bc2'])
        tf.histogram_summary("conv1_2", conv1_2)
    with tf.name_scope('maxPool_1') as scope:
        maxPool_1 = maxPool(conv1_2, k=3, s=2)
        tf.histogram_summary("maxPool_1", maxPool_1)


    with tf.name_scope('conv2_1') as scope:
        conv2_1 = conv2d(maxPool_1, weights['wc3'], biases['bc3'],2)
    with tf.name_scope('conv2_2') as scope:
        conv2_2 = conv2d(conv2_1, weights['wc4'], biases['bc4'])
    with tf.name_scope('conv2_3') as scope:
        conv2_3 = conv2d(conv2_2, weights['wc5'], biases['bc5'])
    with tf.name_scope('maxPool_2') as scope:
        maxPool_2 = maxPool(conv2_3, k=3, s=2)


    with tf.name_scope('conv3_1') as scope:
        conv3_1 = conv2d(maxPool_2, weights['wc6'], biases['bc6'])
    with tf.name_scope('conv3_2') as scope:
        conv3_2 = conv2d(conv3_1, weights['wc7'], biases['bc7'])
    with tf.name_scope('conv3_3') as scope:
        conv3_3 = conv2d(conv3_2, weights['wc8'], biases['bc8'])
    with tf.name_scope('maxPool_3') as scope:
        maxPool_3 = maxPool(conv3_3, k=3, s=2)


    with tf.name_scope('conv4_1') as scope:
        conv4_1 = conv2d(maxPool_3, weights['wc9'], biases['bc9'])
    with tf.name_scope('conv4_2') as scope:
        conv4_2 = conv2d(conv4_1, weights['wc10'], biases['bc10'])
    with tf.name_scope('conv4_3') as scope:
        conv4_3 = conv2d(conv4_2, weights['wc11'], biases['bc11'])
    with tf.name_scope('maxPool_4') as scope:
        maxPool_4 = maxPool(conv4_3, k=3, s=2)


    with tf.name_scope('conv5_1') as scope:
        conv5_1 = conv2d(maxPool_4, weights['wc12'], biases['bc12'])
    with tf.name_scope('conv5_2') as scope:
        conv5_2 = conv2d(conv5_1, weights['wc13'], biases['bc13'])


    with tf.name_scope('dropout1') as scope:
        dropout1 = tf.nn.dropout(conv5_2, dropout)                                   # Apply Dropout


    with tf.name_scope('fc1') as scope:
        dShape = dropout1.get_shape().as_list()                           # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(dropout1, [-1, dShape[1] * dShape[2] * dShape[3]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)


    with tf.name_scope('out') as scope:
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out




# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32]), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32]), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64]), name='wc4'),
    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64]), name='wc5'),
    'wc6': tf.Variable(tf.random_normal([3, 3, 64, 128]), name='wc6'),
    'wc7': tf.Variable(tf.random_normal([3, 3, 128, 128]), name='wc7'),
    'wc8': tf.Variable(tf.random_normal([3, 3, 128, 128]), name='wc8'),
    'wc9': tf.Variable(tf.random_normal([3, 3, 128, 256]), name='wc9'),
    'wc10': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='wc10'),
    'wc11': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='wc11'),
    'wc12': tf.Variable(tf.random_normal([3, 3, 256, 512]), name='wc12'),
    'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512]), name='wc13'),
    'wd1': tf.Variable(tf.random_normal([6144, 1024]), name='wd1'),           # Dim of wd1 should be matched with the dimensions of input recevied from the last layer
    'out': tf.Variable(tf.random_normal([1024, N_CLASSES]), name='out')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([32]), name='bc2'),
    'bc3': tf.Variable(tf.random_normal([64]), name='bc3'),
    'bc4': tf.Variable(tf.random_normal([64]), name='bc4'),
    'bc5': tf.Variable(tf.random_normal([64]), name='bc5'),
    'bc6': tf.Variable(tf.random_normal([128]), name='bc6'),
    'bc7': tf.Variable(tf.random_normal([128]), name='bc7'),
    'bc8': tf.Variable(tf.random_normal([128]), name='bc8'),
    'bc9': tf.Variable(tf.random_normal([256]), name='bc9'),
    'bc10': tf.Variable(tf.random_normal([256]), name='bc10'),
    'bc11': tf.Variable(tf.random_normal([256]), name='bc11'),
    'bc12': tf.Variable(tf.random_normal([512]), name='bc12'),
    'bc13': tf.Variable(tf.random_normal([512]), name='bc13'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
    'out': tf.Variable(tf.random_normal([N_CLASSES]), name='out')
}


with tf.name_scope('Model'):
    pred = convolutionalNet(x, weights, biases, drop_out)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    # Create a summary to monitor cost tensor
    tf.scalar_summary("loss", loss)

with tf.name_scope('SGD'):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)


# Evaluate model
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Create a summary to monitor accuracy tensor
    tf.scalar_summary("accuracy", accuracy)





# Create summaries to visualize weights
#for var in tf.trainable_variables():
#    tf.histogram_summary(var.name, var)
# Summarize all gradients
#for grad, var in grads:
#    tf.histogram_summary(var.name + '/gradient', grad)

# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

# Saver Operation to save and restore all variables, first create directory
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1
    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    # Keep training until reach max iterations
    # start populating filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print 'Starting training'
    while step * BATCH_SIZE < TRAINING_ITERS:
        batch_x, batch_y = sess.run([train_image_batch, train_label_batch])

        _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={x: batch_x, y: batch_y, drop_out: DROPOUT})


        if step % DISPLAY_STEP == 0:
            # Calculate batch loss and accuracy
            loss_value, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, drop_out: 1.})
            print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss_value) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1

        summary_writer.add_summary(summary, step)
    print("Optimization Finished!")

    saver.save(sess, ckpt_dir + "/model.ckpt")