from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import TensorflowUtils as utils
import datetime
from six.moves import xrange
from six.moves import cPickle as pickle
import os
import argparse


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
#tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("logs_dir", "/media/user2/DATA/HAMEDFAHIMI/LiveWire/livewire_original/deep_learning/FCN.tensorflow-master2/logs_scratch_3d/", "path to logs directory")
#tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("data_dir", "/media/user2/DATA/HAMEDFAHIMI/LiveWire/livewire_original/deep_learning/FCN.tensorflow-master2/Data_zoo/Eye_LayerParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")
tf.flags.DEFINE_string('phase', "combination", "phase original/ combination")


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 10
IMAGE_SIZE1 = 128  # 326
IMAGE_SIZE2 = 128  # 466
IMAGE_SIZE2_crop = 64
IMAGE_SIZE3_crop = 16
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='0', type=int,
                    help='The id number of the model to evaluate: i.e. 0 -->max_unpool')
args = parser.parse_args()
model_id = args.model_id;
ckpt_path = os.path.join(FLAGS.logs_dir, "model_%d" % (model_id))
if FLAGS.phase == "combination":
    ckpt_path = os.path.join(FLAGS.logs_dir, "model_%d" % (model_id), "combination")


# lgd = FLAGS.logs_dir.split('/')
# FLAGS.logs_dir = '/'.join(lgd[:-2]+[lgd[-2]+'_%s'%(str(model_id))]+[''])


def inference_deconvolution(image, keep_prob, training):
    """
    FCN with deconvolution decoder definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob: drop-out argument
    """
    print("setting up initialized conv layers ...")
    net = {};
    num_conv = 4;
    height = 7;
    width = 3;
    depth = 2;
    height1 = 7;
    width1 = 3;
    depth1 = 2;
    inpt = image;
    output_depth = 64;
    input_depth = inpt.get_shape().as_list()[4]
    W_t = tf.Variable(tf.truncated_normal([height, width, depth, input_depth, output_depth], stddev=0.02))
    tf.add_to_collection('lr_w', W_t)
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
    output = utils.conv3d_strided_batch_normalization(inpt, W_t, training, stride=1)
    net['conv_batch_normalization1'] = output
    for i in range(1, num_conv):
        output = tf.nn.avg_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        if i == (num_conv - 1):
            height_mlp = 1;
            width_mlp = 1;
            depth_mlp = 1;
            output_mlp = 8
            W_t = tf.Variable(
                tf.truncated_normal([height_mlp, width_mlp, depth_mlp, output.get_shape().as_list()[4], output_mlp],
                                    stddev=0.02))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
            tf.add_to_collection('lr_w', W_t)
            output = utils.conv3d_strided_batch_normalization(output, W_t, training, stride=1)
            output_mlp = 64
            W_t = tf.Variable(
                tf.truncated_normal([height_mlp, width_mlp, depth_mlp, output.get_shape().as_list()[4], output_mlp],
                                    stddev=0.02))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
            tf.add_to_collection('lr_w', W_t)
            output = utils.conv3d_strided_batch_normalization(output, W_t, training, stride=1)
        net['pool%s' % str(i)] = output
        inpt = output;
        output_depth = 64;
        input_depth = inpt.get_shape().as_list()[4]
        W_t = tf.Variable(tf.truncated_normal([height, width, depth, input_depth, output_depth], stddev=0.02))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
        tf.add_to_collection('lr_w', W_t)
        output = utils.conv3d_strided_batch_normalization(inpt, W_t, training, stride=1)
        net['conv_batch_normalization%s' % str(i + 1)] = output

    last_output_num = 64
    # now to upscale to actual image size
    deconv_shape1 = net["conv_batch_normalization3"].get_shape()
    W_t1 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, deconv_shape1[4].value, last_output_num], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t1))
    tf.add_to_collection('lr_w', W_t1)
    # b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
    conv_t1 = utils.conv3d_transpose_strided_batch_normalization(output, W_t1, training, output_shape=tf.shape(
        net["conv_batch_normalization3"]), stride=2)
    try:
        fuse_1 = tf.concat((conv_t1, net["conv_batch_normalization3"]), 4, name="fuse_1")
    except Exception as e:
        fuse_1 = tf.concat(4, (conv_t1, net["conv_batch_normalization3"]), name="fuse_1")
    deconv_shape2 = net["conv_batch_normalization2"].get_shape()
    W_t2 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, deconv_shape2[4].value, 2 * deconv_shape1[4].value], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t2))
    tf.add_to_collection('lr_w', W_t2)
    # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t2 = utils.conv3d_transpose_strided_batch_normalization(fuse_1, W_t2, training, output_shape=tf.shape(
        net["conv_batch_normalization2"]), stride=2)
    try:
        fuse_2 = tf.concat((conv_t2, net["conv_batch_normalization2"]), 4, name="fuse_2")
    except Exception as e:
        fuse_2 = tf.concat(4, (conv_t2, net["conv_batch_normalization2"]), name="fuse_2")

    deconv_shape3 = net["conv_batch_normalization1"].get_shape()
    W_t3 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, deconv_shape3[4].value, 2 * deconv_shape2[4].value], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t3))
    tf.add_to_collection('lr_w', W_t3)
    # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t3 = utils.conv3d_transpose_strided_batch_normalization(fuse_2, W_t3, training, output_shape=tf.shape(
        net["conv_batch_normalization1"]), stride=2)
    try:
        fuse_3 = tf.concat((conv_t3, net["conv_batch_normalization1"]), 4, name="fuse_3")
    except Exception as e:
        fuse_3 = tf.concat(4, (conv_t3, net["conv_batch_normalization1"]), name="fuse_3")

    W_t4 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape3[4].value, last_output_num], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t4))
    tf.add_to_collection('lr_w', W_t4)
    # b_t4 = tf.Variable(tf.constant(0.0, shape=[last_output_num]))
    # tf.add_to_collection('lr_b', b_t4)
    conv_t4 = utils.conv3d_strided_batch_normalization(fuse_3, W_t4, training, stride=1)

    conv_t4 = tf.nn.dropout(conv_t4, keep_prob=keep_prob)
    W_4 = tf.Variable(tf.truncated_normal([1, 1, 1, last_output_num, NUM_OF_CLASSESS], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_4))
    tf.add_to_collection('lr_w', W_4)
    b_4 = tf.Variable(tf.constant(0.0, shape=[NUM_OF_CLASSESS]))
    tf.add_to_collection('lr_b', b_4)
    conv_t4 = tf.nn.conv3d(conv_t4, filter=W_4, strides=[1, 1, 1, 1, 1], padding="SAME")
    conv_t4 = tf.nn.bias_add(conv_t4, b_4)

    weight_decay_sum = tf.add_n(tf.get_collection('weight_decay'))
    lr_w_vars = tf.get_collection('lr_w')
    lr_b_vars = tf.get_collection('lr_b')

    return conv_t4, weight_decay_sum, lr_w_vars, lr_b_vars


def inference_max_unpool(image, keep_prob, training):
    """
    FCN with max unpooling decoder definition

    """
    print("setting up initialized conv layers ...")
    net = {};
    num_conv = 4;
    height = 7;
    width = 3;
    depth = 2;
    height1 = 7;
    width1 = 3;
    depth1 = 2;
    pool_sz = [2, 2, 2]
    inpt = image;
    output_depth = 64;
    input_depth = inpt.get_shape().as_list()[4]
    W_t = tf.Variable(tf.truncated_normal([height, width, depth, input_depth, output_depth], stddev=0.02))
    tf.add_to_collection('lr_w', W_t)
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
    output = utils.conv3d_strided_batch_normalization(inpt, W_t, training, stride=1)
    net['conv_batch_normalization1'] = output
    for i in range(1, num_conv):
        print(output.get_shape())
        argmax = utils.max_pool_3d_with_argmax(output, pool_sz)
        output = tf.nn.max_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        net['pool%s' % str(i)] = output
        net['pool_argmax%s' % str(i)] = argmax
        inpt = output;
        output_depth = 64;
        input_depth = inpt.get_shape().as_list()[4]
        W_t = tf.Variable(tf.truncated_normal([height, width, depth, input_depth, output_depth], stddev=0.02))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
        tf.add_to_collection('lr_w', W_t)
        output = utils.conv3d_strided_batch_normalization(inpt, W_t, training, stride=1)
        net['conv_batch_normalization%s' % str(i + 1)] = output

    last_output_num = 64
    # now to upscale to actual image size
    unpool1 = utils.unpool_layer_batch_unraveled_indices(output, net['pool_argmax3'], pool_sz)
    try:
        fuse_1 = tf.concat((unpool1, net["conv_batch_normalization3"]), 4, name="fuse_1")
    except Exception as e:
        fuse_1 = tf.concat(4, (unpool1, net["conv_batch_normalization3"]), name="fuse_1")
    deconv_shape1 = net["conv_batch_normalization3"].get_shape()
    W_t1 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape1[4].value, last_output_num], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t1))
    tf.add_to_collection('lr_w', W_t1)
    # b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
    conv_t1 = utils.conv3d_strided_batch_normalization(fuse_1, W_t1, training, stride=1)

    unpool2 = utils.unpool_layer_batch_unraveled_indices(conv_t1, net['pool_argmax2'], pool_sz)
    try:
        fuse_2 = tf.concat((unpool2, net["conv_batch_normalization2"]), 4, name="fuse_2")
    except Exception as e:
        fuse_2 = tf.concat(4, (unpool2, net["conv_batch_normalization2"]), name="fuse_2")
    deconv_shape2 = net["conv_batch_normalization2"].get_shape()
    W_t2 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape1[4].value, deconv_shape2[4].value, ],
                            stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t2))
    tf.add_to_collection('lr_w', W_t2)
    # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t2 = utils.conv3d_strided_batch_normalization(fuse_2, W_t2, training, stride=1)

    unpool3 = utils.unpool_layer_batch_unraveled_indices(conv_t2, net['pool_argmax1'], pool_sz)
    deconv_shape3 = net["conv_batch_normalization1"].get_shape()
    try:
        fuse_3 = tf.concat((unpool3, net["conv_batch_normalization1"]), 4, name="fuse_3")
    except Exception as e:
        fuse_3 = tf.concat(4, (unpool3, net["conv_batch_normalization1"]), name="fuse_3")
    W_t3 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape2[4].value, deconv_shape3[4].value, ],
                            stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t3))
    tf.add_to_collection('lr_w', W_t3)
    # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t3 = utils.conv3d_strided_batch_normalization(fuse_3, W_t3, training, stride=1)

    #    W_t4 = tf.Variable(tf.truncated_normal([height1, width1, depth1, 2*deconv_shape3[4].value, last_output_num], stddev=0.02))
    #    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t3))
    #    tf.add_to_collection('lr_w', W_t4)
    #    #b_t4 = tf.Variable(tf.constant(0.0, shape=[last_output_num]))
    #    #tf.add_to_collection('lr_b', b_t4)
    #    conv_t4 = utils.conv3d_strided_batch_normalization(conv_t3, W_t4, training, stride=1)

    #    conv_t4 = tf.nn.dropout(conv_t4, keep_prob=keep_prob)
    W_4 = tf.Variable(tf.truncated_normal([1, 1, 1, last_output_num, NUM_OF_CLASSESS], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_4))
    tf.add_to_collection('lr_w', W_4)
    b_4 = tf.Variable(tf.constant(0.0, shape=[NUM_OF_CLASSESS]))
    tf.add_to_collection('lr_b', b_4)

    conv_t4 = tf.nn.conv3d(conv_t3, filter=W_4, strides=[1, 1, 1, 1, 1], padding="SAME")
    conv_t4 = tf.nn.bias_add(conv_t4, b_4)

    weight_decay_sum = tf.add_n(tf.get_collection('weight_decay'))
    lr_w_vars = tf.get_collection('lr_w')
    lr_b_vars = tf.get_collection('lr_b')

    return conv_t3, conv_t4, weight_decay_sum, lr_w_vars, lr_b_vars


def inference_median_unpool(image, keep_prob, training):
    """
    FCN with median unpooling decoder definition

    """
    print("setting up initialized conv layers ...")
    net = {};
    num_conv = 4;
    height = 7;
    width = 3;
    depth = 3;
    height1 = 7;
    width1 = 3;
    depth1 = 3;
    pool_sz = [2, 2, 2]
    inpt = image;
    output_depth = 64;
    input_depth = inpt.get_shape().as_list()[4]
    W_t = tf.Variable(tf.truncated_normal([height, width, depth, input_depth, output_depth], stddev=0.02))
    tf.add_to_collection('lr_w', W_t)
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
    output = utils.conv3d_strided_batch_normalization(inpt, W_t, training, stride=1)
    net['conv_batch_normalization1'] = output
    for i in range(1, num_conv):
        print(output.get_shape())
        output, argmax = utils.median_pool_3d_with_argmedian(output, pool_sz)
        print(argmax.get_shape())
        net['pool%s' % str(i)] = output
        net['pool_argmax%s' % str(i)] = argmax
        inpt = output;
        output_depth = 64;
        input_depth = inpt.get_shape().as_list()[4]
        W_t = tf.Variable(tf.truncated_normal([height, width, depth, input_depth, output_depth], stddev=0.02))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t))
        tf.add_to_collection('lr_w', W_t)
        output = utils.conv3d_strided_batch_normalization(inpt, W_t, training, stride=1)
        net['conv_batch_normalization%s' % str(i + 1)] = output

    last_output_num = 64
    # now to upscale to actual image size
    unpool1 = utils.unpool_layer_batch_unraveled_indices(output, net['pool_argmax3'], pool_sz)
    try:
        fuse_1 = tf.concat((unpool1, net["conv_batch_normalization3"]), 4, name="fuse_1")
    except Exception as e:
        fuse_1 = tf.concat(4, (unpool1, net["conv_batch_normalization3"]), name="fuse_1")
    deconv_shape1 = net["conv_batch_normalization3"].get_shape()
    W_t1 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape1[4].value, last_output_num], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t1))
    tf.add_to_collection('lr_w', W_t1)
    # b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
    conv_t1 = utils.conv3d_strided_batch_normalization(fuse_1, W_t1, training, stride=1)

    unpool2 = utils.unpool_layer_batch_unraveled_indices(conv_t1, net['pool_argmax2'], pool_sz)
    try:
        fuse_2 = tf.concat((unpool2, net["conv_batch_normalization2"]), 4, name="fuse_2")
    except Exception as e:
        fuse_2 = tf.concat(4, (unpool2, net["conv_batch_normalization2"]), name="fuse_2")
    deconv_shape2 = net["conv_batch_normalization2"].get_shape()
    W_t2 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape1[4].value, deconv_shape2[4].value, ],
                            stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t2))
    tf.add_to_collection('lr_w', W_t2)
    # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t2 = utils.conv3d_strided_batch_normalization(fuse_2, W_t2, training, stride=1)

    unpool3 = utils.unpool_layer_batch_unraveled_indices(conv_t2, net['pool_argmax1'], pool_sz)
    deconv_shape3 = net["conv_batch_normalization1"].get_shape()
    try:
        fuse_3 = tf.concat((unpool3, net["conv_batch_normalization1"]), 4, name="fuse_3")
    except Exception as e:
        fuse_3 = tf.concat(4, (unpool3, net["conv_batch_normalization1"]), name="fuse_3")
    W_t3 = tf.Variable(
        tf.truncated_normal([height1, width1, depth1, 2 * deconv_shape2[4].value, deconv_shape3[4].value, ],
                            stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t3))
    tf.add_to_collection('lr_w', W_t3)
    # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t3 = utils.conv3d_strided_batch_normalization(fuse_3, W_t3, training, stride=1)

    #    W_t4 = tf.Variable(tf.truncated_normal([height1, width1, depth1, 2*deconv_shape3[4].value, last_output_num], stddev=0.02))
    #    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_t3))
    #    tf.add_to_collection('lr_w', W_t4)
    #    #b_t4 = tf.Variable(tf.constant(0.0, shape=[last_output_num]))
    #    #tf.add_to_collection('lr_b', b_t4)
    #    conv_t4 = utils.conv3d_strided_batch_normalization(conv_t3, W_t4, training, stride=1)

    #    conv_t4 = tf.nn.dropout(conv_t4, keep_prob=keep_prob)
    W_4 = tf.Variable(tf.truncated_normal([1, 1, 1, last_output_num, NUM_OF_CLASSESS], stddev=0.02))
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_4))
    tf.add_to_collection('lr_w', W_4)
    b_4 = tf.Variable(tf.constant(0.0, shape=[NUM_OF_CLASSESS]))
    tf.add_to_collection('lr_b', b_4)

    conv_t4 = tf.nn.conv3d(conv_t3, filter=W_4, strides=[1, 1, 1, 1, 1], padding="SAME")
    conv_t4 = tf.nn.bias_add(conv_t4, b_4)

    weight_decay_sum = tf.add_n(tf.get_collection('weight_decay'))
    lr_w_vars = tf.get_collection('lr_w')
    lr_b_vars = tf.get_collection('lr_b')
    return conv_t3, conv_t4, weight_decay_sum, lr_w_vars, lr_b_vars

def main(argv=None):
    iter_start = 0
    new_training_round = True
    print(ckpt_path)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    training = tf.placeholder_with_default(True, shape=())
    print("Setting up image reader...")
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 613;  # 8*22*128/64
    NUM_EPOCHS_PER_DECAY = 30  # 350;
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY);
    LEARNING_RATE_DECAY_FACTOR = 0.1
    global_step = tf.Variable(0, trainable=False);
    starter_learning_rate1 = 0.1;
    starter_learning_rate2 = 0.2;
    learning_rate1 = tf.train.exponential_decay(starter_learning_rate1, global_step, decay_steps,
                                                LEARNING_RATE_DECAY_FACTOR, staircase=True)
    learning_rate2 = tf.train.exponential_decay(starter_learning_rate2, global_step, decay_steps,
                                                LEARNING_RATE_DECAY_FACTOR, staircase=True)
    # boundaries = [32000, 64000]; values = [0.1,0.01,0.001]
    # learning_rate1 = tf.train.piecewise_constant(global_step, boundaries, values)
    # learning_rate2 = tf.train.piecewise_constant(global_step, boundaries, [2*x for x in values])

    if FLAGS.mode == 'train':
        image_options = {'resize': False, 'resize_size1': IMAGE_SIZE1, 'resize_size2': IMAGE_SIZE2, 'random_crop': True,
                         'combination': False, 'crop_h': IMAGE_SIZE2_crop, 'crop_d': IMAGE_SIZE3_crop}
        if FLAGS.phase == "combination":
            image_options['combination'] = True
        filenames = [os.path.join(FLAGS.data_dir, 'OCT_train_weighted_cropped_cycled_3d_128.bin')]
        num_samples = 8 * 22 * IMAGE_SIZE2 / 64;
        filename_queue = tf.train.string_input_producer(
            filenames)  # Create a queue that produces the filenames to read.
        image, annotation, weight_annotation_edge = utils.ImageProducer_train_3d(filename_queue, image_options);
        min_fraction_of_examples_in_queue = 0.4;
        min_queue_examples = int(num_samples * min_fraction_of_examples_in_queue)
        image_batch, annotation_batch, weight_annotation_edge_batch = tf.train.shuffle_batch(
            [image, annotation, weight_annotation_edge], batch_size=FLAGS.batch_size, num_threads=16,
            capacity=min_queue_examples + 3 * FLAGS.batch_size, min_after_dequeue=min_queue_examples)
    else:
        image_options = {'resize': False, 'resize_size1': IMAGE_SIZE1, 'resize_size2': IMAGE_SIZE2,
                         'combination': False, 'random_crop': False, 'crop_h': IMAGE_SIZE2}
        if FLAGS.phase == "combination":
            image_options['combination'] = True
        filenames = [os.path.join(FLAGS.data_dir, 'OCT_test_weighted_cropped_cycled_3d_128_16.bin')]
        num_samples = 2 * 11 * IMAGE_SIZE2 / 64;
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)
        # Start the image processing workers
        image, annotation, weight_annotation_edge = utils.ImageProducer_test(filename_queue, image_options);
        min_fraction_of_examples_in_queue = 0.4;
        min_queue_examples = int(num_samples * min_fraction_of_examples_in_queue)
        # Read 'batch_size' images + labels from the example queue.(without shuffling)
        image_batch, annotation_batch, weight_annotation_edge_batch = tf.train.batch(
            [image, annotation, weight_annotation_edge], batch_size=FLAGS.batch_size)
    #        image_batch = tf.expand_dims(image_batch, dim=3)
    #        annotation_batch = tf.expand_dims(annotation_batch, dim=3)

    #    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE1, IMAGE_SIZE2, 3], name="input_image")
    #    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE1, IMAGE_SIZE2, 1], name="annotation")
    #    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    #    if FLAGS.mode == 'train':
    #        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    #    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    #
    #    valid_part = validation_dataset_reader.get_all_batch(FLAGS.batch_size)
    #    train_part = train_dataset_reader.get_all_batch(FLAGS.batch_size)

    print("Entering the inference part")
    if model_id in [0, 1, 2]:
        conv_t3, logits, weight_decay_sum, lr_weight_vars, lr_biases_vars = inference_max_unpool(image_batch, keep_probability,
                                                                                    training)
    if model_id in [3, 4, 5]:
        logits, weight_decay_sum, lr_weight_vars, lr_biases_vars = inference_deconvolution(image_batch, keep_probability, training)
    if model_id in [6, 7, 8]:
        conv_t3, logits, weight_decay_sum, lr_weight_vars, lr_biases_vars = inference_median_unpool(image_batch,
                                                                                               keep_probability,
                                                                                               training)
    sess = tf.Session()
    # Continue training...
    if FLAGS.phase == "combination":
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())  # Load weights from the other model
        saver = tf.train.Saver(max_to_keep=None)
        ckpt_path_original = os.sep.join(ckpt_path.split(os.sep)[:-1])
        ckpt = tf.train.get_checkpoint_state(ckpt_path_original)
        print("Original pretrained model is %s" % (ckpt_path_original))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from %s" % (ckpt.model_checkpoint_path))
        TRANSFERABLE_VARIABLES = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        temp = [vv.name for vv in TRANSFERABLE_VARIABLES]
        print(sess.run(temp[0]))
        #print('Variables is',sess.run(TRANSFERABLE_VARIABLES))
        transferable_weights = sess.run(TRANSFERABLE_VARIABLES)
        out_shape = conv_t3.get_shape()
        W_c = tf.Variable(tf.truncated_normal([1, 1, out_shape[3].value, out_shape[4].value, NUM_OF_CLASSESS], stddev=0.02))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(W_c))
        tf.add_to_collection('lr_w', W_c)
        logits = tf.nn.conv3d(conv_t3, filter=W_c, strides=[1, 1, 1, 1, 1], padding="VALID")

    pred_annotation = tf.argmax(logits, dimension=4, name="prediction")
    pred_annotation = tf.expand_dims(pred_annotation, dim=4)

    #    tf.summary.image("input_image", image, max_outputs=2)
    #    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    #    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=tf.squeeze(annotation_batch,
                                                                                     squeeze_dims=[4]),
                                                                   name="entropy")
    orig_loss = tf.reduce_mean(cross_entropy)

    # errors = tf.logical_not(tf.equal(tf.cast(pred_annotation,'int32'), annotation_batch))
    # weight_errors = tf.multiply(weight_annotation_edge_batch, tf.cast(errors,'float'))
    if model_id in [0, 3, 6]: #Standard cross entropy loss function
        loss = tf.add(orig_loss, tf.multiply(0.0001, weight_decay_sum))
    elif model_id in [1, 4, 7]: #Weighted cross entropy loss function
        loss = tf.add(
            tf.reduce_mean(tf.multiply(tf.squeeze(weight_annotation_edge_batch, squeeze_dims=[4]), cross_entropy)),
            tf.multiply(0.0001, weight_decay_sum))
    elif model_id in [2, 5, 8]: #weighted cross entropy loss function augmented with Dice overlap score
        y_hat_softmax = tf.nn.softmax(logits)
        shp = tf.shape(annotation_batch)
        zero_part = tf.zeros(shp)
        one_part = tf.ones(shp)
        class_ind = tf.equal(0, annotation_batch)
        zero_one = tf.where(class_ind, one_part, zero_part)
        for class_id in range(1, NUM_OF_CLASSESS):
            class_ind = tf.equal(class_id, annotation_batch)
            try:
                zero_one = tf.concat((zero_one, tf.where(class_ind, one_part, zero_part)), axis=4)
            except Exception as e:
                zero_one = tf.concat(4, (zero_one, tf.where(class_ind, one_part, zero_part)))
        Den = tf.add(tf.reduce_sum(tf.multiply(zero_one, zero_one)),
                     tf.reduce_sum(tf.multiply(y_hat_softmax, y_hat_softmax)))
        Num = tf.reduce_sum(tf.multiply(zero_one, y_hat_softmax))
        Dice_loss = tf.subtract(1., tf.divide(tf.multiply(2., Num), Den))
        loss = tf.add(tf.add(
            tf.reduce_mean(tf.multiply(tf.squeeze(weight_annotation_edge_batch, squeeze_dims=[4]), cross_entropy)),
            tf.multiply(0.5, Dice_loss)), tf.multiply(0.0001, weight_decay_sum))

    #    tf.summary.scalar("entropy", loss)

    #    opt1 = tf.train.MomentumOptimizer(learning_rate1,momentum = 0.9)
    #    opt2 = tf.train.MomentumOptimizer(learning_rate2,momentum = 0.9)
    #    grads = tf.gradients(loss, lr_weight_vars + lr_biases_vars)
    #    grads1 = grads[:len(lr_weight_vars)]
    #    grads2 = grads[len(lr_weight_vars):(len(lr_weight_vars)+len(lr_biases_vars))]
    #    train_op1 = opt1.apply_gradients(zip(grads1, lr_weight_vars),global_step=global_step)
    #    train_op2 = opt2.apply_gradients(zip(grads2, lr_biases_vars),global_step=global_step)
    #    train_op = tf.group(train_op1, train_op2)
    trainable_var = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate1)
    grads = optimizer.compute_gradients(loss, var_list=trainable_var)
    train_op = optimizer.apply_gradients(grads,global_step=global_step)
    print("Setting up summary op...")
    #    summary_op = tf.summary.merge_all()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())  # Load weights from the other model
    if FLAGS.phase == "combination":
        for var, weight in zip(TRANSFERABLE_VARIABLES, transferable_weights):
            var.load(weight, sess)
    print('learning rate is', sess.run(learning_rate1))
    print('global step is', sess.run(global_step))
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=None)
    #    summary_writer = tf.summary.FileWriter(ckpt_path, sess.graph)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        if FLAGS.mode == "train":
            saver.restore(sess, ckpt.model_checkpoint_path)
            iter_start = int(ckpt.model_checkpoint_path.split(os.sep)[-1].split('-')[-1].split('.')[0]) + 1
            print("Model restored from %s" % (ckpt.model_checkpoint_path))
        elif FLAGS.mode == "visualize":
            ckpt.model_checkpoint_path = os.path.join(ckpt_path, 'scratch_model.ckpt-20001')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from %s" % (ckpt.model_checkpoint_path))
    print('global step is', sess.run(global_step))
    print('learning rate is', sess.run(learning_rate1))
    if FLAGS.mode == "train":
        try:
            if not new_training_round:
                record_file = open(os.path.join(ckpt_path, 'records%s.pkl' % str(iter_start - 1)), 'rb')
                training_record = pickle.load(record_file)
                record_file.close()
                os.remove(os.path.join(ckpt_path, 'records%s.pkl' % str(iter_start - 1)))
            else:
                training_record = {'train_loss': [], 'orig_loss': [], 'iteration': [], 'Dice_loss': []}
        except Exception as e:
            print(e)
        threads = tf.train.start_queue_runners(sess)  # Start the image processing workers
        feed_dict = {keep_probability: 0.85}
        for itr in xrange(iter_start, MAX_ITERATION):
            sess.run(train_op, feed_dict=feed_dict)
            if (itr % 500 == 0 or new_training_round):
                # train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                train_loss = sess.run(loss, feed_dict=feed_dict)
                training_record['train_loss'].append(train_loss)
                training_record['iteration'].append(itr)
                training_record['orig_loss'].append(sess.run(orig_loss, feed_dict=feed_dict))
                if model_id in [2, 5, 8]:
                    training_record['Dice_loss'].append(sess.run(Dice_loss, feed_dict=feed_dict))
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                # summary_writer.add_summary(summary_str, itr)
                saver.save(sess, os.path.join(ckpt_path, "scratch_model.ckpt"), global_step=global_step)
                file1 = open(os.path.join(ckpt_path, "records%s.pkl" % str(itr)), 'wb')
                pickle.dump(training_record, file1)
                file1.close()
                new_training_round = False
    elif FLAGS.mode == "visualize":
        result = np.zeros((IMAGE_SIZE1, IMAGE_SIZE2, IMAGE_SIZE3_crop,22))
        if FLAGS.phase == "combination":
            result = np.zeros((IMAGE_SIZE1, IMAGE_SIZE2, 1, 22))
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            max_steps = int(math.ceil(num_samples / FLAGS.batch_size))
            step = 0
            while step < max_steps:
                pred = sess.run(pred_annotation, feed_dict={keep_probability: 1.0})
                pred = np.squeeze(pred, axis=4)
                print(np.shape(pred))
                print(np.shape(pred[0]))
                for itr in range(FLAGS.batch_size):
                    result[:, :, :, step * FLAGS.batch_size + itr] = pred[itr]
                    print("Saved results: %d" % itr)
                    with open(os.path.join(ckpt_path, 'results.pickle'), 'wb') as f:
                        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                step = step + 1
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        sess.close()


if __name__ == "__main__":
    tf.app.run()