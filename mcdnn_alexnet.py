import scipy.io
import tensorflow as tf
import os
from pylab import *
import numpy as np
import pickle
from numpy import *
from tensorflow.contrib.data import Iterator, Dataset
import threading

WEIGHT_DECAY = 0.0001
batch_size = 128
net_data = load("bvlc_alexnet.npy").item()
data_dir = '../50_species_simple_collection/'
plant_list = ['flower', 'leaf', 'branch', 'entire']
image_size = 227
label_count = 50

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
        kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

def read_images_from_disk(input_queue):
    label = tf.cast(input_queue[1], tf.float32)
    file_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image, [227, 227], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32) * (1. / 255)
    return image, label

def read_tfrecords_file(input_queue, feature, type):
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(input_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features[type + 'image'], tf.float32)
    label = tf.cast(features[type + 'label'], tf.int32)
    label = tf.one_hot(label, label_count)
    image = tf.reshape(image, [4, 227, 227, 3])
    return image, label

def simple_shuffle_batch(source, capacity, batch_size=batch_size):
    # Create a random shuffle queue.
    queue = tf.RandomShuffleQueue(capacity=capacity,
                                min_after_dequeue=int(0.9*capacity),
                                shapes=[source[0].shape, source[1].shape], dtypes=[source[0].dtype, source[1].dtype])

    # Create an op to enqueue one item.
    enqueue = queue.enqueue(source)

    # Create a queue runner that, when started, will launch 4 threads applying
    # that enqueue op.
    num_threads = 4
    qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

    # Register the queue runner so it can be found and started by
    # `tf.train.start_queue_runners` later (the threads are not launched yet).
    tf.train.add_queue_runner(qr)

    # Create an op to dequeue a batch
    return queue.dequeue_many(batch_size)


def run_model(train_path, test_path):
    graph = tf.Graph()
    with graph.as_default():
        feature = {'mcdnntrain/image': tf.FixedLenFeature([], tf.string),
               'mcdnntrain/label': tf.FixedLenFeature([], tf.int64)}

        filename_queue = tf.train.string_input_producer(train_path, num_epochs=3)
        data_batch, label_batch = tf.train.shuffle_batch(read_tfrecords_file(filename_queue, feature, 'mcdnntrain/'), 
            batch_size=batch_size, 
            capacity=10000,
            min_after_dequeue=1000,
            allow_smaller_final_batch=True)

        feature_test = {'mcdnntest/image': tf.FixedLenFeature([], tf.string),
                'mcdnntest/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue_test = tf.train.string_input_producer(test_path, num_epochs=1)
        data_batch_test, label_batch_test = tf.train.batch(read_tfrecords_file(filename_queue_test, feature_test, 'mcdnntest/'), batch_size=10)

        #queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.float32], shapes=[[227, 227, 3], [label_count]])
        #enqueue_op = queue.enqueue_many([images, labels])
        #dequeue_op = queue.dequeue()

        #data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=500)

        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])

        conv1W1 = tf.Variable(net_data["conv1"][0])
        conv1b1 = tf.Variable(net_data["conv1"][1])
        conv2W1 = tf.Variable(net_data["conv2"][0])
        conv2b1 = tf.Variable(net_data["conv2"][1])
        conv3W1 = tf.Variable(net_data["conv3"][0])
        conv3b1 = tf.Variable(net_data["conv3"][1])
        conv4W1 = tf.Variable(net_data["conv4"][0])
        conv4b1 = tf.Variable(net_data["conv4"][1])
        conv5W1 = tf.Variable(net_data["conv5"][0])
        conv5b1 = tf.Variable(net_data["conv5"][1])
        fc6W1 = tf.Variable(net_data["fc6"][0])
        fc6b1 = tf.Variable(net_data["fc6"][1])
        fc7W1 = tf.Variable(net_data["fc7"][0])
        fc7b1 = tf.Variable(net_data["fc7"][1])
        fc8W1 = tf.Variable(net_data["fc8"][0])
        fc8b1 = tf.Variable(net_data["fc8"][1])

        conv1W2 = tf.Variable(net_data["conv1"][0])
        conv1b2 = tf.Variable(net_data["conv1"][1])
        conv2W2 = tf.Variable(net_data["conv2"][0])
        conv2b2 = tf.Variable(net_data["conv2"][1])
        conv3W2 = tf.Variable(net_data["conv3"][0])
        conv3b2 = tf.Variable(net_data["conv3"][1])
        conv4W2 = tf.Variable(net_data["conv4"][0])
        conv4b2 = tf.Variable(net_data["conv4"][1])
        conv5W2 = tf.Variable(net_data["conv5"][0])
        conv5b2 = tf.Variable(net_data["conv5"][1])
        fc6W2 = tf.Variable(net_data["fc6"][0])
        fc6b2 = tf.Variable(net_data["fc6"][1])
        fc7W2 = tf.Variable(net_data["fc7"][0])
        fc7b2 = tf.Variable(net_data["fc7"][1])
        fc8W2 = tf.Variable(net_data["fc8"][0])
        fc8b2 = tf.Variable(net_data["fc8"][1])

        conv1W3 = tf.Variable(net_data["conv1"][0])
        conv1b3 = tf.Variable(net_data["conv1"][1])
        conv2W3 = tf.Variable(net_data["conv2"][0])
        conv2b3 = tf.Variable(net_data["conv2"][1])
        conv3W3 = tf.Variable(net_data["conv3"][0])
        conv3b3 = tf.Variable(net_data["conv3"][1])
        conv4W3 = tf.Variable(net_data["conv4"][0])
        conv4b3 = tf.Variable(net_data["conv4"][1])
        conv5W3 = tf.Variable(net_data["conv5"][0])
        conv5b3 = tf.Variable(net_data["conv5"][1])
        fc6W3 = tf.Variable(net_data["fc6"][0])
        fc6b3 = tf.Variable(net_data["fc6"][1])
        fc7W3 = tf.Variable(net_data["fc7"][0])
        fc7b3 = tf.Variable(net_data["fc7"][1])
        fc8W3 = tf.Variable(net_data["fc8"][0])
        fc8b3 = tf.Variable(net_data["fc8"][1])

        fc9W = weight_variable([1000 * 4, label_count], 'W_fc9')
        fc9b = bias_variable([label_count], 'b_fc9')
        keep_prob = tf.placeholder('float')

        def enqueue(session):
            while True:
                session.run([images, labels, enqueue_op])

        def model(x, conv1W, conv1b, conv2W, conv2b, conv3W, conv3b, conv4W, conv4b, conv5W, conv5b, fc6W, fc6b, fc7W, fc7b, fc8W, fc8b):
            conv1 = tf.nn.relu(conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="VALID", group=1))
            lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=0.00002, beta=0.75, bias=1.0)
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
            lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=0.00002, beta=0.75, bias=1.0)
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
            conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
            conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
            fc6_drop = tf.nn.dropout(fc6, keep_prob)
            fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b)
            fc7_drop = tf.nn.dropout(fc7, keep_prob)
            fc8 = tf.nn.relu_layer(fc7_drop, fc8W, fc8b)
            return fc8

        data_batch0, data_batch1, data_batch2, data_batch3 = tf.split(data_batch, num_or_size_splits=4, axis=1)
        logits0 = model(tf.squeeze(data_batch0, [1]), conv1W, conv1b, conv2W, conv2b, conv3W, conv3b, conv4W, conv4b, conv5W, conv5b, fc6W, fc6b, fc7W, fc7b, fc8W, fc8b)
        logits1 = model(tf.squeeze(data_batch1, [1]), conv1W1, conv1b1, conv2W1, conv2b1, conv3W1, conv3b1, conv4W1, conv4b1, conv5W1, conv5b1, fc6W1, fc6b1, fc7W1, fc7b1, fc8W1, fc8b1)
        logits2 = model(tf.squeeze(data_batch2, [1]), conv1W2, conv1b2, conv2W2, conv2b2, conv3W2, conv3b2, conv4W2, conv4b2, conv5W2, conv5b2, fc6W2, fc6b2, fc7W2, fc7b2, fc8W2, fc8b2)
        logits3 = model(tf.squeeze(data_batch3, [1]), conv1W3, conv1b3, conv2W3, conv2b3, conv3W3, conv3b3, conv4W3, conv4b3, conv5W3, conv5b3, fc6W3, fc6b3, fc7W3, fc7b3, fc8W3, fc8b3)
        logits = tf.concat([logits0, logits1, logits2, logits3], 1)
        logits = tf.nn.xw_plus_b(logits, fc9W, fc9b)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch))
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = tf.reduce_mean(cross_entropy + WEIGHT_DECAY * regularizers)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.02, global_step, 2000, 0.65, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits),1), tf.argmax(label_batch,1))
        count_correct_prediction = tf.reduce_sum(tf.cast(correct_prediction, 'float'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        data_batch_test0, data_batch_test1, data_batch_test2, data_batch_test3 = tf.split(data_batch_test, num_or_size_splits=4, axis=1)
        logits_test0 = model(tf.squeeze(data_batch_test0, [1]), conv1W, conv1b, conv2W, conv2b, conv3W, conv3b, conv4W, conv4b, conv5W, conv5b, fc6W, fc6b, fc7W, fc7b, fc8W, fc8b)
        logits_test1 = model(tf.squeeze(data_batch_test1, [1]), conv1W1, conv1b1, conv2W1, conv2b1, conv3W1, conv3b1, conv4W1, conv4b1, conv5W1, conv5b1, fc6W1, fc6b1, fc7W1, fc7b1, fc8W1, fc8b1)
        logits_test2 = model(tf.squeeze(data_batch_test2, [1]), conv1W2, conv1b2, conv2W2, conv2b2, conv3W2, conv3b2, conv4W2, conv4b2, conv5W2, conv5b2, fc6W2, fc6b2, fc7W2, fc7b2, fc8W2, fc8b2)
        logits_test3 = model(tf.squeeze(data_batch_test3, [1]), conv1W3, conv1b3, conv2W3, conv2b3, conv3W3, conv3b3, conv4W3, conv4b3, conv5W3, conv5b3, fc6W3, fc6b3, fc7W3, fc7b3, fc8W3, fc8b3)
        logits_test = tf.concat([logits_test0, logits_test1, logits_test2, logits_test3], 1)
        logits_test = tf.nn.xw_plus_b(logits_test, fc9W, fc9b)
        test_labels = tf.argmax(label_batch_test, 1)
        prediction_vector_score = tf.nn.softmax(logits_test)
        count_correct_prediction_test = tf.reduce_sum(tf.cast(
            tf.equal(tf.argmax(prediction_vector_score,1), test_labels), 'float'))
        count_top_k_correct_prediction = [tf.reduce_sum(tf.cast(tf.nn.in_top_k(prediction_vector_score, test_labels, k=i), 'float')) for i in range(1, 50)]

        saver = tf.train.Saver()


    with tf.Session(graph=graph) as session:
        session.run(tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer()))

        #enqueue_thread = threading.Thread(target=enqueue, args=[session])
        #enqueue_thread.isDaemon()
        #enqueue_thread.start()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #saver.restore(session, './mcdnn_alexnet_plant50.ckpt')

        try:
            step = 0
            while True:
                batch_res = session.run([ train_step, loss, accuracy ],
                    feed_dict = { keep_prob: 0.5 })

                print step, batch_res[1:]
                step += 1

        except tf.errors.OutOfRangeError:
              print('Done training')
        
        saver.save(session, './mcdnn_alexnet_plant50.ckpt')

        test_ret_topk = [0.0] * 49
        test_count = 0
        try:
            while True:
                ret = session.run([count_correct_prediction_test, count_top_k_correct_prediction], 
                    feed_dict = { keep_prob: 1.0})
                test_ret_topk = [x + y for x, y in zip(test_ret_topk, ret[1])]
                test_count = test_count + 10 
        except tf.errors.OutOfRangeError:
            print('Done testing')

        print np.array(test_ret_topk) / test_count
        
        #session.run(queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        session.close()

test_path = ['flower_leaf.tfrecords']
train_path = ['mcdnn_train.tfrecords']
run_model(train_path, test_path)