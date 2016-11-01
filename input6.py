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
csv_path = tf.train.string_input_producer(['/home/waleed/Desktop/leaf_example/train.csv'])
textReader = tf.TextLineReader()
_, csv_content = textReader.read(csv_path)
im_name, label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])

# load images, and convert labels into one_hot encoded form
im_content = tf.read_file(im_name)
image = tf.image.decode_jpeg(im_content, channels=3)
image = tf.cast(image, tf.float32) # could be unnecessary
#image = tf.image.resize_images(image, 240, 180)
label = tf.one_hot(label, 40, 1, 0 )
label = tf.cast(label, tf.float32 )


#========================================================================================================
# Data Augmentation
#========================================================================================================

height = IMAGE_HEIGHT
width = IMAGE_WIDTH

# Image processing for training the network. Note the many random
# distortions applied to the image.

# Crop the center portion
image = tf.image.central_crop(image, 0.8)

image = tf.image.resize_images(image, 240, 180)
# Randomly crop a [height, width] section of the image.
#image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# Randomly flip the image horizontally.
distorted_image = tf.image.random_flip_left_right(image)

# Because these operations are not commutative, consider randomizing
# the order their operation.
distorted_image = tf.image.random_brightness(distorted_image, max_delta=50)
distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=2.8)

# Subtract off the mean and divide by the variance of the pixels.
float_image = tf.image.per_image_whitening(distorted_image)



# Make Batches of images with shuffling
min_after_dequeue = 30                                               # Defines how big a buffer we will randomly sample from -- bigger means better shuffling but slower start up and more memory used
capacity = min_after_dequeue + 3 * BATCH_SIZE                        # Capacity must be larger than min_after_dequeue and the amount larger determines the maximum we will prefetch.
train_image_batch, train_label_batch = tf.train.shuffle_batch(
    [float_image, label], batch_size=BATCH_SIZE, capacity=capacity,
    min_after_dequeue=min_after_dequeue)

tf.image_summary('images', train_image_batch, max_images=10)
#====================================================================================================
# Model Hyperparameters, and characteristics
#=====================================================================================================
# Model Parameters
LEARNING_RATE = 0.0001
TRAINING_ITERS = 20000
DISPLAY_STEP = 10

# Network Parameters
N_CLASSES = 40
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

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)
# Model
def convolutionalNet(x, weights, biases, dropout):


    with tf.name_scope('conv1_1') as scope:
        conv1_1 = conv2d(x, weights['wc1'], biases['bc1'],2)
        tf.histogram_summary("conv1_1", conv1_1)

        with tf.variable_scope('visualization'):

            grid_x =8
            grid_y =4
            grid = put_kernels_on_grid (weights['wc1'], grid_y, grid_x)
            tf.image_summary('conv1/features', grid, max_images=1)
            # scale weights to [0 1], type is still float
            #x_min = tf.reduce_min(weights['wc1'])
            #x_max = tf.reduce_max(weights['wc1'])
            #kernel_0_to_1 = (weights['wc1'] - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            #kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
            # this will display random 3 filters from the 64 in conv1
            #tf.image_summary('conv1_1/filters', kernel_transposed, max_images=3)

        ## Prepare for visualization
        # Take only convolutions of first image, discard convolutions for other images.
        #V = tf.slice(conv1_1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
        #V = tf.reshape(V, (120, 90, 32))

        # Reorder so the channels are in the first dimension, x and y follow.
        #V = tf.transpose(V, (2, 0, 1))
        # Bring into shape expected by image_summary
        #V = tf.reshape(V, (-1, 240, 180, 1))

        #tf.image_summary("first_conv", V)

        #with tf.variable_scope('visualization'):
            #tf.get_variable_scope().reuse_variables()
            #weights = tf.get_variable(weights['wc1'])
            #grid_x = grid_y = 8   # to get a square grid for 64 conv1 features
            #grid = put_kernels_on_grid (weights, (grid_y, grid_x))
            #tf.image_summary('conv1/features', grid, max_images=1)

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    #grads = tf.gradients(loss, tf.trainable_variables())
    #grads = list(zip(grads, tf.trainable_variables()))
    #apply_grads = optimizer.apply_gradients(grads_and_vars=grads)


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
        #print  batch_y

        _, c, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={x: batch_x, y: batch_y, drop_out: DROPOUT})


        if step % DISPLAY_STEP == 0:
            # Calculate batch loss and accuracy
            loss_value, acc= sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, drop_out: 0.5})
            print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss_value) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
            summary_writer.add_summary(summary, step)
        step += 1

        #summary_writer.add_summary(summary, step)
    print("Optimization Finished!")

    saver.save(sess, ckpt_dir + "/model.ckpt")
    '''
    combined_summary.MergeFromString(image_summary)
    if i % 10 == 0:
        summary_writer.add_summary(combined_summary)
        combined_summary = tf.Summary()
    '''