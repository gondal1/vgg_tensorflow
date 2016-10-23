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
DISPLAY_STEP = 10

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

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([60*45*64, 1024])),           # Dim of wd1 should be matched with the dimensions of input recevied from the last layer
    'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
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
for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.histogram_summary(var.name + '/gradient', grad)

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