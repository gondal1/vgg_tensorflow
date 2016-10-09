
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



#graph = tf.Graph()
#with graph.as_default():

input_image = tf.placeholder(dtype=tf.float32, shape=[None, 784])
labels = tf.placeholder(dtype= tf.float32 , shape= [None, 10])
x_image = tf.reshape(input_image, [-1, 28, 28, 1])
# MEAN COMPUTATION AND SUBTRACTION TO BE INCLUDED
# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# pool1
pool1 = tf.nn.max_pool(conv1_2,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool1')
# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)
#self.parameters += [kernel, biases]
# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# pool2
pool2 = tf.nn.max_pool(conv2_2,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool2')
# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# conv3_3
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# pool3
pool3 = tf.nn.max_pool(conv3_3,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool3')
# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)
    #self.parameters += [kernel, biases]
# conv4_3
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    #parameters += [kernel, biases]
# pool4
pool4 = tf.nn.max_pool(conv4_3,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool4')
# conv5_1
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)
    #parameters += [kernel, biases]
# conv5_2
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)
    #parameters += [kernel, biases]
# conv5_3
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)
    #parameters += [kernel, biases]
# pool5
pool5 = tf.nn.max_pool(conv5_3,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool5')
#def fc_layers(self):
# fc1
with tf.name_scope('fc1') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    fc1w = tf.Variable(tf.truncated_normal([shape, 1024], dtype=tf.float32,
                                        stddev=1e-1), name='weights')
    fc1b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),
                             trainable=True, name='biases')
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    fc1 = tf.nn.relu(fc1l)
    #parameters += [fc1w, fc1b]
# fc2
with tf.name_scope('fc2') as scope:
    fc2w = tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32,
                                        stddev=1e-1), name='weights')
    fc2b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),
                             trainable=True, name='biases')
    fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
    fc2 = tf.nn.relu(fc2l)
    #parameters += [fc2w, fc2b]
# fc3
with tf.name_scope('fc3') as scope:
    fc3w = tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32,
                                        stddev=1e-1), name='weights')
    fc3b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32),
                             trainable=True, name='biases')
    fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
y_predicted = fc3l
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predicted, labels))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(y_predicted)
#valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
#test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
correct_prediction =  tf.equal(tf.argmax(y_predicted,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print ('initialized')

    for steps in range (1000):
        batch = mnist.train.next_batch(50)

        feed_dict = {input_image : batch[0], labels : batch[1]}
        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if steps%100 == 0:

            print('Loss at step %d: %f' % (steps, l))
            train_accuracy = accuracy.eval(session=sess, feed_dict={input_image:batch[0], labels: batch[1]})
            print("step %d, training accuracy %g"%(steps, train_accuracy))
        #print('Training accuracy: %.1f%%' % accuracy(
        #  predictions, train_labels[:train_subset, :]))
        # Calling .eval() on valid_prediction is basically like calling run(), but
        # just to get that one numpy array. Note that it recomputes all its graph
        # dependencies.
        #print('Validation accuracy: %.1f%%' % accuracy(
        #valid_prediction.eval(), valid_labels))
        #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))




#     sess = tf.Session()
#     x = tf.placeholder(tf.float32, [None, 784])
#     y_ = tf.placeholder(tf.float32, shape=[None, 10])
#     #y = vgg16(x)  # for now passing x as an empty image
#     cross_entropy = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(y, y_))
#     train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
#     correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     sess.run(tf.initialize_all_variables())
#     for i in range(1000):
#         batch = mnist.train.next_batch(50)
#         if i%100 == 0:
#             train_accuracy = accuracy.eval(session=sess, feed_dict={
#                 x:batch[0], y_: batch[1], keep_prob: 1.0})
#             print("step %d, training accuracy %g"%(i, train_accuracy))
#         train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# print("test accuracy %g"%accuracy.eval(feed_dict={
#         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# print 'still working'



    # vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    #
    # img1 = imread('elephant.jpg', mode='RGB')
    # img1 = imresize(img1, (224, 224))
    #
    # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print class_names[p], prob[p]
