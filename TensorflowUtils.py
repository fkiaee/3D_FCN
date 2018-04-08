# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import random

np.random.seed(0);tf.set_random_seed(0);

def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def ImageProducer_train_3d(filename_queue,image_options): 
    height = 128; width = 128; depth = 21
    image_bytes = height * width * depth
    annotation_bytes = image_bytes
    weight_annotation_edge_bytes = image_bytes
    record_bytes = image_bytes + annotation_bytes + weight_annotation_edge_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    image = tf.slice(record_bytes, [0], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    depth_major = tf.reshape(image,[depth,height,width,1])
    image = tf.transpose(depth_major, [1, 2, 0, 3])  
    
    annotation = tf.slice(record_bytes, [image_bytes], [annotation_bytes]);
    annotation = tf.cast(annotation, tf.int32)   
    annotation = tf.reshape(annotation,[1,image_bytes])      
    depth_major = tf.reshape(annotation,[depth,height,width,1])
    annotation = tf.transpose(depth_major, [1, 2, 0, 3])
    
    weight_annotation_edge = tf.slice(record_bytes, [image_bytes+annotation_bytes], [weight_annotation_edge_bytes]);
    weight_annotation_edge = tf.cast(weight_annotation_edge, tf.float32)   
    weight_annotation_edge = tf.reshape(weight_annotation_edge,[1,image_bytes])  
    depth_major = tf.reshape(weight_annotation_edge,[depth,height,width, 1])
    weight_annotation_edge = tf.transpose(depth_major, [1, 2, 0, 3])   

#    if image_options.get("resize", False) and image_options["resize"]:
#        new_shape = tf.stack([image_options["resize_size1"], image_options["resize_size2"]])
#        image = tf.image.resize_images(image, new_shape)
#        annotation = tf.image.resize_images(annotation, new_shape)
#        weight_annotation_edge = tf.image.resize_images(weight_annotation_edge, new_shape)
    if image_options.get("random_crop", False):
        crop_h = image_options["crop_h"]
        crop_d = image_options["crop_d"]
        offset_d = random.sample(range(depth-crop_d),1)[0]
        crop_points = range(0,width,64)
        offset_h = random.sample(crop_points,1)[0]
        translation_h = random.sample(range(-10,10),1)[0]
        if (not(offset_h==0 and translation_h<0) and not(offset_h==crop_points[-1] and translation_h>0)):
            offset_h = offset_h + translation_h
        offset_h = tf.to_int32(offset_h); offset_d = tf.to_int32(offset_d)  
        image = tf.slice(image, begin=tf.stack([0, offset_h, offset_d, 0]), size=tf.stack([-1, crop_h, crop_d, -1]))
        annotation = tf.slice(annotation, begin=tf.stack([0, offset_h, offset_d, 0]), size=tf.stack([-1, crop_h, crop_d, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, offset_h, offset_d, 0]), size=tf.stack([-1, crop_h, crop_d, -1]))
        #flip all randomly
        distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
        distort_left_right_random = distortions[0]
        distort_depth_random = distortions[1]
        try:
            mirror = tf.less(tf.stack([1.0, distort_left_right_random, distort_depth_random, 1.0]), 0.5)
            image = tf.reverse(image, mirror)        
            annotation = tf.reverse(annotation, mirror)        
            weight_annotation_edge = tf.reverse(weight_annotation_edge, mirror)        
        except Exception as e:
            mirror_cond = tf.less(distort_left_right_random, 0.5)
            mirror_depth_cond = tf.less(distort_depth_random, 0.5)
            image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
            annotation = tf.cond(mirror_cond, lambda: tf.reverse(annotation, [1]), lambda: annotation)
            weight_annotation_edge = tf.cond(mirror_cond, lambda: tf.reverse(weight_annotation_edge, [1]), lambda: weight_annotation_edge)
            image = tf.cond(mirror_depth_cond, lambda: tf.reverse(image, [2]), lambda: image)
            annotation = tf.cond(mirror_depth_cond, lambda: tf.reverse(annotation, [2]), lambda: annotation)
            weight_annotation_edge = tf.cond(mirror_depth_cond, lambda: tf.reverse(weight_annotation_edge, [2]), lambda: weight_annotation_edge)
    if image_options.get("combination", False):
        middle = crop_d/2;
        annotation = tf.slice(annotation, begin=tf.stack([0, 0, middle, 0]), size=tf.stack([-1, -1, 1, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, 0, middle, 0]), size=tf.stack([-1, -1,1, -1]))

    return image, annotation, weight_annotation_edge  

def ImageProducer_train_3d_biosig(filename_queue,image_options): 
    height = 128; width = 128; depth = 19
    image_bytes = height * width * depth
    annotation_bytes = image_bytes
    weight_annotation_edge_bytes = image_bytes
    record_bytes = image_bytes + annotation_bytes + weight_annotation_edge_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    image = tf.slice(record_bytes, [0], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    depth_major = tf.reshape(image,[depth,height,width,1])
    image = tf.transpose(depth_major, [1, 2, 0, 3])  
    
    annotation = tf.slice(record_bytes, [image_bytes], [annotation_bytes]);
    annotation = tf.cast(annotation, tf.int32)   
    annotation = tf.reshape(annotation,[1,image_bytes])      
    depth_major = tf.reshape(annotation,[depth,height,width,1])
    annotation = tf.transpose(depth_major, [1, 2, 0, 3])
    
    weight_annotation_edge = tf.slice(record_bytes, [image_bytes+annotation_bytes], [weight_annotation_edge_bytes]);
    weight_annotation_edge = tf.cast(weight_annotation_edge, tf.float32)   
    weight_annotation_edge = tf.reshape(weight_annotation_edge,[1,image_bytes])  
    depth_major = tf.reshape(weight_annotation_edge,[depth,height,width, 1])
    weight_annotation_edge = tf.transpose(depth_major, [1, 2, 0, 3])   

#    if image_options.get("resize", False) and image_options["resize"]:
#        new_shape = tf.stack([image_options["resize_size1"], image_options["resize_size2"]])
#        image = tf.image.resize_images(image, new_shape)
#        annotation = tf.image.resize_images(annotation, new_shape)
#        weight_annotation_edge = tf.image.resize_images(weight_annotation_edge, new_shape)
    if image_options.get("random_crop", False):
        crop_h = image_options["crop_h"]
        crop_d = image_options["crop_d"]
        offset_d = random.sample(range(depth-crop_d),1)[0]
        crop_points = range(0,width,64)
        offset_h = random.sample(crop_points,1)[0]
        translation_h = random.sample(range(-10,10),1)[0]
        if (not(offset_h==0 and translation_h<0) and not(offset_h==crop_points[-1] and translation_h>0)):
            offset_h = offset_h + translation_h
        offset_h = tf.to_int32(offset_h); offset_d = tf.to_int32(offset_d)  
        image = tf.slice(image, begin=tf.stack([0, offset_h, offset_d, 0]), size=tf.stack([-1, crop_h, crop_d, -1]))
        annotation = tf.slice(annotation, begin=tf.stack([0, offset_h, offset_d, 0]), size=tf.stack([-1, crop_h, crop_d, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, offset_h, offset_d, 0]), size=tf.stack([-1, crop_h, crop_d, -1]))
        #flip all randomly
        distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
        distort_left_right_random = distortions[0]
        distort_depth_random = distortions[1]
        try:
            mirror = tf.less(tf.stack([1.0, distort_left_right_random, distort_depth_random, 1.0]), 0.5)
            image = tf.reverse(image, mirror)        
            annotation = tf.reverse(annotation, mirror)        
            weight_annotation_edge = tf.reverse(weight_annotation_edge, mirror)        
        except Exception as e:
            mirror_cond = tf.less(distort_left_right_random, 0.5)
            mirror_depth_cond = tf.less(distort_depth_random, 0.5)
            image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
            annotation = tf.cond(mirror_cond, lambda: tf.reverse(annotation, [1]), lambda: annotation)
            weight_annotation_edge = tf.cond(mirror_cond, lambda: tf.reverse(weight_annotation_edge, [1]), lambda: weight_annotation_edge)
            image = tf.cond(mirror_depth_cond, lambda: tf.reverse(image, [2]), lambda: image)
            annotation = tf.cond(mirror_depth_cond, lambda: tf.reverse(annotation, [2]), lambda: annotation)
            weight_annotation_edge = tf.cond(mirror_depth_cond, lambda: tf.reverse(weight_annotation_edge, [2]), lambda: weight_annotation_edge)
    if image_options.get("combination", False):
        middle = crop_d/2;
        annotation = tf.slice(annotation, begin=tf.stack([0, 0, middle, 0]), size=tf.stack([-1, -1, 1, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, 0, middle, 0]), size=tf.stack([-1, -1,1, -1]))

    return image, annotation, weight_annotation_edge  
    
def ImageProducer_test(filename_queue,image_options): 
    height = 128; width = 128; depth = 8
    image_bytes = height * width * depth
    annotation_bytes = image_bytes
    weight_annotation_edge_bytes = image_bytes
    record_bytes = image_bytes + annotation_bytes+weight_annotation_edge_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    image = tf.slice(record_bytes, [0], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    depth_major = tf.reshape(image,[depth,height,width,1])
    image = tf.transpose(depth_major, [1, 2, 0, 3])  

    annotation = tf.slice(record_bytes, [image_bytes], [annotation_bytes]);
    annotation = tf.cast(annotation, tf.int32)   
    annotation = tf.reshape(annotation,[1,image_bytes])      
    depth_major = tf.reshape(annotation,[depth,height,width, 1])
    annotation = tf.transpose(depth_major, [1, 2, 0, 3]) 
    
    weight_annotation_edge = tf.slice(record_bytes, [image_bytes+annotation_bytes], [weight_annotation_edge_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    weight_annotation_edge = tf.cast(weight_annotation_edge, tf.float32)   
    weight_annotation_edge = tf.reshape(weight_annotation_edge,[1,image_bytes])  
    depth_major = tf.reshape(weight_annotation_edge,[depth,height,width,1])
    weight_annotation_edge = tf.transpose(depth_major, [1, 2, 0, 3])  

    if image_options.get("combination", False):
        middle = depth/2;
        annotation = tf.slice(annotation, begin=tf.stack([0, 0, middle, 0]), size=tf.stack([-1, -1, 1, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, 0, middle, 0]), size=tf.stack([-1, -1,1, -1]))

    return image, annotation, weight_annotation_edge

def ImageProducer(filename_queue,image_options): 
    height = 128; width = 128; depth = 1
    image_bytes = height * width * depth
    annotation_bytes = image_bytes
    weight_annotation_edge_bytes = image_bytes
    record_bytes = image_bytes + annotation_bytes + weight_annotation_edge_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    image = tf.slice(record_bytes, [0], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    depth_major = tf.reshape(image,[depth,height,width])
    image = tf.transpose(depth_major, [1, 2, 0])  
    annotation = tf.slice(record_bytes, [image_bytes], [annotation_bytes]);
    annotation = tf.cast(annotation, tf.int32)   
    annotation = tf.reshape(annotation,[1,image_bytes])  
    
    depth_major = tf.reshape(annotation,[depth,height,width])
    annotation = tf.transpose(depth_major, [1, 2, 0])
    weight_annotation_edge = tf.slice(record_bytes, [image_bytes+annotation_bytes], [weight_annotation_edge_bytes]);
    weight_annotation_edge = tf.cast(weight_annotation_edge, tf.float32)   
    weight_annotation_edge = tf.reshape(weight_annotation_edge,[1,image_bytes])  
    depth_major = tf.reshape(weight_annotation_edge,[depth,height,width])
    weight_annotation_edge = tf.transpose(depth_major, [1, 2, 0])   
    if image_options.get("resize", False) and image_options["resize"]:
        new_shape = tf.stack([image_options["resize_size1"], image_options["resize_size2"]])
        image = tf.image.resize_images(image, new_shape)
        annotation = tf.image.resize_images(annotation, new_shape)
        weight_annotation_edge = tf.image.resize_images(weight_annotation_edge, new_shape)
    if image_options.get("random_crop", False):
        crop_h = image_options["crop_h"]
        #offset = random.sample(range(width-crop_h),1)[0]
        crop_points = range(0,width,64)
        offset = random.sample(crop_points,1)[0]
        offset = tf.to_int32(offset) 
        image = tf.slice(image, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        annotation = tf.slice(annotation, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        #flip all randomly
        distortions = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)
        distort_left_right_random = distortions[0]
        try:
            mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
            image = tf.reverse(image, mirror)        
            annotation = tf.reverse(annotation, mirror)        
            weight_annotation_edge = tf.reverse(weight_annotation_edge, mirror)        
        except Exception as e:
            mirror_cond = tf.less(distort_left_right_random, 0.5)
            image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
            annotation = tf.cond(mirror_cond, lambda: tf.reverse(annotation, [1]), lambda: annotation)
            weight_annotation_edge = tf.cond(mirror_cond, lambda: tf.reverse(weight_annotation_edge, [1]), lambda: weight_annotation_edge)
    if image_options.get("channels", False):  # make sure images are of shape(h,w,3)
            try:
                image = tf.concat((image,image,image),axis=2)   
            except Exception as e:
                image = tf.concat(2,(image,image,image))           
    return image, annotation, weight_annotation_edge
    
def ImageProducer_old(filename_queue,image_options): 
    height = 326; width = 466; depth = 1
    image_bytes = height * width * depth
    annotation_bytes = image_bytes
    weight_annotation_edge_bytes = image_bytes
    weight_input_edge_bytes = image_bytes
    record_bytes = image_bytes + annotation_bytes + weight_annotation_edge_bytes + weight_input_edge_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    image = tf.slice(record_bytes, [0], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    depth_major = tf.reshape(image,[depth,height,width])
    image = tf.transpose(depth_major, [1, 2, 0])  
    annotation = tf.slice(record_bytes, [image_bytes], [annotation_bytes]);
    annotation = tf.cast(annotation, tf.int32)   
    annotation = tf.reshape(annotation,[1,image_bytes])  
    
    depth_major = tf.reshape(annotation,[depth,height,width])
    annotation = tf.transpose(depth_major, [1, 2, 0])
    weight_annotation_edge = tf.slice(record_bytes, [image_bytes+annotation_bytes], [weight_annotation_edge_bytes]);
    weight_annotation_edge = tf.cast(weight_annotation_edge, tf.float32)   
    weight_annotation_edge = tf.reshape(weight_annotation_edge,[1,image_bytes])  
    depth_major = tf.reshape(weight_annotation_edge,[depth,height,width])
    weight_annotation_edge = tf.transpose(depth_major, [1, 2, 0])   
    weight_input_edge = tf.slice(record_bytes, [image_bytes+annotation_bytes+weight_annotation_edge_bytes], [weight_input_edge_bytes]);
    weight_input_edge = tf.cast(weight_input_edge, tf.float32)   
    weight_input_edge = tf.reshape(weight_input_edge,[1,image_bytes])  
    depth_major = tf.reshape(weight_input_edge,[depth,height,width])
    weight_input_edge = tf.transpose(depth_major, [1, 2, 0])    

    if image_options.get("resize", False) and image_options["resize"]:
        new_shape = tf.stack([image_options["resize_size1"], image_options["resize_size2"]])
        image = tf.image.resize_images(image, new_shape)
        annotation = tf.image.resize_images(annotation, new_shape)
        weight_annotation_edge = tf.image.resize_images(weight_annotation_edge, new_shape)
        weight_input_edge = tf.image.resize_images(weight_input_edge, new_shape)
    if image_options.get("random_crop", False):
        crop_h = image_options["crop_h"]
        #offset = random.sample(range(width-crop_h),1)[0]
        offset = random.sample([0,64,128,192,256,320,384,402],1)[0]
        offset = tf.to_int32(offset) 
        image = tf.slice(image, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        annotation = tf.slice(annotation, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        weight_annotation_edge = tf.slice(weight_annotation_edge, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        weight_input_edge = tf.slice(weight_input_edge, begin=tf.stack([0, offset, 0]), size=tf.stack([-1, crop_h, -1]))
        #flip all randomly
        distortions = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)
        distort_left_right_random = distortions[0]
        try:
            mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
            image = tf.reverse(image, mirror)        
            annotation = tf.reverse(annotation, mirror)        
            weight_annotation_edge = tf.reverse(weight_annotation_edge, mirror)        
            weight_input_edge = tf.reverse(weight_input_edge, mirror)   
        except Exception as e:
            mirror_cond = tf.less(distort_left_right_random, 0.5)
            image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
            annotation = tf.cond(mirror_cond, lambda: tf.reverse(annotation, [1]), lambda: annotation)
            weight_annotation_edge = tf.cond(mirror_cond, lambda: tf.reverse(weight_annotation_edge, [1]), lambda: weight_annotation_edge)
            weight_input_edge = tf.cond(mirror_cond, lambda: tf.reverse(weight_input_edge, [1]), lambda: weight_input_edge)       
        
    return image, annotation, weight_annotation_edge, weight_input_edge   
def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv3d_strided(x, W, b):
    conv = tf.nn.conv3d(x, W, strides=[1, 2, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv3d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] *= 2
        output_shape[4] = W.get_shape().as_list()[3]
    # print output_shape
    conv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel

#########################################################################################################
def conv2d_strided_batch_normalization1(inpt, filter_, training, stride):
    out_channels = filter_.get_shape().as_list()[3]
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)
    return out

def conv2d_transpose_strided_batch_normalization1(x, W, training, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    out_channels = W.get_shape().as_list()[2]
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")        
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)    
    return out


def conv3d_strided_batch_normalization1(inpt, filter_, training, stride):
    out_channels = filter_.get_shape().as_list()[4]
    conv = tf.nn.conv3d(inpt, filter=filter_, strides=[1, stride, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)
    return out

def conv3d_transpose_strided_batch_normalization1(x, W, training, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    out_channels = W.get_shape().as_list()[3]
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] *= 2
        output_shape[4] = W.get_shape().as_list()[3]
    # print output_shape
    conv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")        
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)    
    return out
#########################################################################################################
def conv2d_strided_batch_normalization(inpt, filter_, training, stride):
    out_channels = filter_.get_shape().as_list()[3]
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
#    mean, var = tf.nn.moments(conv, axes=[0,1,2])
#    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
#    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")    
#    batch_norm = tf.nn.batch_norm_with_global_normalization(
#        conv, mean, var, beta, gamma, 0.001,
#        scale_after_normalization=True)
    batch_norm = tf.contrib.layers.batch_norm(
        conv,
        data_format='NHWC',  # Matching the "cnn" tensor which has shape (?, 9, 120, 160, 96).
        center=True,
        scale=True,
        is_training=training,
        #scope='cnn3d-batch_norm'
        )
    out = tf.nn.relu(batch_norm)
    return out

def conv2d_transpose_strided_batch_normalization(x, W, training, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    out_channels = W.get_shape().as_list()[2]
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
#    mean, var = tf.nn.moments(conv, axes=[0,1,2])
#    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
#    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")        
#    batch_norm = tf.nn.batch_norm_with_global_normalization(
#        conv, mean, var, beta, gamma, 0.001,
#        scale_after_normalization=True)
    batch_norm = tf.contrib.layers.batch_norm(
        conv,
        data_format='NHWC',  # Matching the "cnn" tensor which has shape (?, 9, 120, 160, 96).
        center=True,
        scale=True,
        is_training=training,
        #scope='cnn3d-batch_norm'
        )
    out = tf.nn.relu(batch_norm)    
    return out


def conv3d_strided_batch_normalization(inpt, filter_, training, stride):
    out_channels = filter_.get_shape().as_list()[4]
    conv = tf.nn.conv3d(inpt, filter=filter_, strides=[1, stride, stride, stride, 1], padding="SAME")
#    mean, var = tf.nn.moments(conv, axes=[0,1,2])
#    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
#    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")    
#    batch_norm = tf.nn.batch_norm_with_global_normalization(
#        conv, mean, var, beta, gamma, 0.001,
#        scale_after_normalization=True)
    batch_norm = tf.contrib.layers.batch_norm(
        conv,
        data_format='NHWC',  # Matching the "cnn" tensor which has shape (?, 9, 120, 160, 96).
        center=True,
        scale=True,
        is_training=training,
        #scope='cnn3d-batch_norm'
        )
    out = tf.nn.relu(batch_norm)
    return out

def conv3d_transpose_strided_batch_normalization(x, W, training, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    out_channels = W.get_shape().as_list()[3]
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] *= 2
        output_shape[4] = W.get_shape().as_list()[3]
    # print output_shape
    conv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding="SAME")
#    mean, var = tf.nn.moments(conv, axes=[0,1,2])
#    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
#    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name="gamma")        
#    batch_norm = tf.nn.batch_norm_with_global_normalization(
#        conv, mean, var, beta, gamma, 0.001,
#        scale_after_normalization=True)
    batch_norm = tf.contrib.layers.batch_norm(
        conv,
        data_format='NHWC',  # Matching the "cnn" tensor which has shape (?, 9, 120, 160, 96).
        center=True,
        scale=True,
        is_training=training,
        #scope='cnn3d-batch_norm'
        )
    out = tf.nn.relu(batch_norm)    
    return out

def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)

def unravel_argmax_x_y(argmax, shape):
    return ((argmax // (shape[2]*shape[3])) % shape[1]) , (argmax % (shape[2]*shape[3]) // shape[3])
        
def unravel_argmax(argmax, shape):
    output_list = [(argmax // (shape[2]*shape[3])) % shape[1],
                   argmax % (shape[2]*shape[3]) // shape[3]]
    return tf.stack(output_list)

def unpool_layer2x2_batch(bottom, argmax, pool_sz,bottom_shape=None):
    #bottom_shape = tf.shape(bottom)
    #top_shape = [bottom_shape[0], bottom_shape[1]*pool_sz, bottom_shape[2]*pool_sz, bottom_shape[3]]
    if not(bottom_shape):
        bottom_shape = bottom.get_shape()
    top_shape = [bottom_shape[0].value, bottom_shape[1].value*pool_sz, bottom_shape[2].value*pool_sz, bottom_shape[3].value]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = [batch_size, height, width, channels]#tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//pool_sz)*(height//pool_sz)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//pool_sz, width//pool_sz, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//pool_sz)*(height//pool_sz)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//pool_sz, width//pool_sz, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])
    #try:
    t = tf.concat(4, [t2, t3, t1])
    #except Exception as e:
    #t = tf.concat([t2, t3, t1],4)

    indices = tf.reshape(t, [(height//pool_sz)*(width//pool_sz)*channels*batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    print('values shape is:', bottom.get_shape())
    print('values shape is:', bottom_shape)
    print('indices shape is:', t.get_shape())
    values = tf.reshape(x1, [-1])

#    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    delta = tf.SparseTensor(indices, values, top_shape)

    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


def max_pool_3d_with_argmax(x,pool_sz):
    bottom_shape = x.get_shape()
    pool_sz_y = pool_sz[0]; pool_sz_x = pool_sz[1]; pool_sz_z = pool_sz[2];
    batch_size = bottom_shape[0].value
    height = bottom_shape[1].value
    width = bottom_shape[2].value
    depth = bottom_shape[3].value
    channels = bottom_shape[4].value
    
    argmax_shape = [batch_size, height, width, channels] 
    part = tf.squeeze(tf.slice(x,begin=[0,0,0,0,0],size=[-1,-1,-1,1,-1]),axis=3)
    pool , argmax = tf.nn.max_pool_with_argmax(part, ksize=[1, pool_sz_y, pool_sz_x, 1], strides=[1, pool_sz_y, pool_sz_y, 1], padding='SAME')
    pool_record = tf.expand_dims(pool,axis=3)
    argmax_y , argmax_x = unravel_argmax_x_y(argmax,argmax_shape)
    argmax_record_x = tf.expand_dims(argmax_x,axis=3)
    argmax_record_y = tf.expand_dims(argmax_y,axis=3)
    for kk in range(1,depth):
        part = tf.squeeze(tf.slice(x,begin=[0,0,0,kk,0],size=[-1,-1,-1,1,-1]),axis=3)
        pool , argmax = tf.nn.max_pool_with_argmax(part, ksize=[1, pool_sz_y, pool_sz_x, 1], strides=[1, pool_sz_y, pool_sz_y, 1], padding='SAME')
        #print('pool shape is',sess.run(tf.shape(pool)))
        pool = tf.expand_dims(pool,axis=3)
        try:
            pool_record = tf.concat((pool_record,pool),3)
        except Exception as e:
            pool_record = tf.concat(3,(pool_record,pool))
        argmax_y , argmax_x = unravel_argmax_x_y(argmax,argmax_shape)
        try:
            argmax_record_y = tf.concat((argmax_record_y,tf.expand_dims(argmax_y,axis=3)),3)
            argmax_record_x = tf.concat((argmax_record_x,tf.expand_dims(argmax_x,axis=3)),3) 
        except Exception as e:
            argmax_record_y = tf.concat(3,(argmax_record_y,tf.expand_dims(argmax_y,axis=3)))
            argmax_record_x = tf.concat(3,(argmax_record_x,tf.expand_dims(argmax_x,axis=3))) 
            
    #print('pool_record shape is',sess.run(tf.shape(pool_record)))
    argmax_shape = tf.to_int64([batch_size, height/pool_sz_y, depth, channels])  
    part = tf.squeeze(tf.slice(pool_record,begin=[0,0,0,0,0],size=[-1,-1,1,-1,-1]),axis=2)
    #print('here part shape is',sess.run(tf.shape(part)))
    pool , argmax = tf.nn.max_pool_with_argmax(part, ksize=[1, 1, pool_sz_z, 1], strides=[1, 1, pool_sz_z, 1], padding='SAME')
    _ , argmax = unravel_argmax_x_y(argmax,argmax_shape)
    #print('here pool shape is',sess.run(tf.shape(pool)))
    pool_record2= tf.expand_dims(pool,axis=2)
    argmax_record_z = tf.expand_dims(argmax,axis=2)
    for kk in range(1,int(width/2)):    
        part = tf.squeeze(tf.slice(pool_record,begin=[0,0,kk,0,0],size=[-1,-1,1,-1,-1]),axis=2)
        #print('part shape is',sess.run(tf.shape(part)))
        pool , argmax = tf.nn.max_pool_with_argmax(part, ksize=[1, 1, pool_sz_z, 1], strides=[1, 1, pool_sz_z, 1], padding='SAME')
        #print('pool shape is',sess.run(tf.shape(pool)))
        pool= tf.expand_dims(pool,axis=2)
        try:
            pool_record2 = tf.concat((pool_record2,pool),2)
        except Exception as e:
            pool_record2 = tf.concat(2,(pool_record2,pool))
        _ , argmax_z = unravel_argmax_x_y(argmax,argmax_shape)
        argmax_z = tf.expand_dims(argmax_z,axis=2)
        try:
            argmax_record_z = tf.concat((argmax_record_z,argmax_z),2)
        except Exception as e:
            argmax_record_z = tf.concat(2,(argmax_record_z,argmax_z))
    
    #print(sess.run(pool_record2))
    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//pool_sz_x)*(height//pool_sz_y)*(depth//pool_sz_z)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//pool_sz_y, width//pool_sz_x, depth//pool_sz_z])
    t1 = tf.transpose(t1, perm=[1, 2, 3, 4, 0])
    
    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//pool_sz_x)*(height//pool_sz_y)*(depth//pool_sz_z)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, height//pool_sz_y, width//pool_sz_x, depth//pool_sz_z, channels])
    
    t3 = tf.to_int64(tf.range(width//pool_sz_x))
    t3 = tf.tile(t3, [batch_size*channels*(height//pool_sz_y)*(depth//pool_sz_z)])
    t3 = tf.reshape(t3, [-1, width//pool_sz_x])
    t3 = tf.transpose(t3, perm=[1, 0])
    t3 = tf.reshape(t3, [width//pool_sz_x, batch_size, height//pool_sz_y, depth//pool_sz_z, channels])
    t3 = tf.transpose(t3, perm=[1, 2, 0, 3, 4])
    
    t4 = tf.to_int64(tf.range(height//pool_sz_y))
    t4 = tf.tile(t4, [batch_size*channels*(width//pool_sz_x)*(depth//pool_sz_z)])
    t4 = tf.reshape(t4, [-1, height//pool_sz_y])
    t4 = tf.transpose(t4, perm=[1, 0])
    t4 = tf.reshape(t4, [height//pool_sz_y, batch_size, width//pool_sz_x, depth//pool_sz_z, channels])
    t4 = tf.transpose(t4, perm=[1, 0, 2, 3, 4])
    
    
    # get list of ((row1, col1), (row2, col2), ..)
    coords = tf.stack([t2, t4, t3, argmax_record_z, t1])
    coords = tf.transpose(coords,perm=[1,2,3,4,5,0])
    coords = tf.reshape(coords, [(height//pool_sz_y)*(width//pool_sz_x)*(depth//pool_sz_z)*channels*batch_size, 5])
    argmax_record_x_final = tf.gather_nd(argmax_record_x, coords)
    #print(sess.run(tf.shape(argmax_record_x_final)))
    argmax_record_y_final = tf.gather_nd(argmax_record_y, coords)
    dim = (height//pool_sz_y)*(width//pool_sz_x)*(depth//pool_sz_z)*channels*batch_size 
    indices = [tf.reshape(t2,[dim,]),argmax_record_y_final,argmax_record_x_final,tf.reshape(argmax_record_z,[dim,]),tf.reshape(t1,[dim,])] 
    indices1 = tf.transpose(tf.stack(indices))
#    print(sess.run(tf.shape(indices1)))
#    #indices = tf.transpose(indices,perm=[1,2,3,4,5,0])
#    #indices1 = tf.reshape(indices, [(height//pool_sz)*(width//pool_sz)*(depth//pool_sz)*channels*batch_size, 5])
#    ind1 = sess.run(indices1[1])
#    print(ind1)
#    ind2 = sess.run(indices1[6])
#    print(ind2)
#    y = ss[ind1[0],ind1[1],ind1[2],ind1[3],ind1[4]]
#    print(y)
#    z = ss[ind2[0],ind2[1],ind2[2],ind2[3],ind2[4]]
#    print(z)
    return indices1

def median_pool_3d_with_argmedian(bottom,pool_sz):
    bottom_shape = bottom.get_shape()
    pool_sz_y = pool_sz[0]; pool_sz_x = pool_sz[1]; pool_sz_z = pool_sz[2];
    batch_size = bottom_shape[0].value
    height = bottom_shape[1].value
    width = bottom_shape[2].value
    depth = bottom_shape[3].value
    channels = bottom_shape[4].value
    
    part = tf.squeeze(tf.slice(bottom,begin=[0,0,0,0,0],size=[-1,-1,-1,1,-1]),axis=3)
    pool , _, argmax_y, argmax_x = median_pool_2d_with_argmax(part,[pool_sz_y,pool_sz_x])
    pool_record = tf.expand_dims(pool,axis=3)
    argmax_record_x = tf.expand_dims(argmax_x,axis=3)
    argmax_record_y = tf.expand_dims(argmax_y,axis=3)
    for kk in range(1,depth):
        part = tf.squeeze(tf.slice(bottom,begin=[0,0,0,kk,0],size=[-1,-1,-1,1,-1]),axis=3)
        pool , _, argmax_y, argmax_x = median_pool_2d_with_argmax(part,[pool_sz_y,pool_sz_x])
        #print('part shape is',sess.run(tf.shape(part)))
        #print('pool shape is',sess.run(tf.shape(pool)))
        #print('argmaxy shape is',sess.run(tf.shape(argmax_y)))
        pool = tf.expand_dims(pool,axis=3)
        try:
            pool_record = tf.concat((pool_record,pool),3)
        except Exception as e:
            pool_record = tf.concat(3,(pool_record,pool))
        try:
            argmax_record_y = tf.concat((argmax_record_y,tf.expand_dims(argmax_y,axis=3)),3)
            argmax_record_x = tf.concat((argmax_record_x,tf.expand_dims(argmax_x,axis=3)),3) 
        except Exception as e:
            argmax_record_y = tf.concat(3,(argmax_record_y,tf.expand_dims(argmax_y,axis=3)))
            argmax_record_x = tf.concat(3,(argmax_record_x,tf.expand_dims(argmax_x,axis=3))) 
            
    #print('pool_record shape is',sess.run(tf.shape(pool_record)))
    #print('argmaxy_record shape is',sess.run(tf.shape(argmax_record_y)))    
    part = tf.squeeze(tf.slice(pool_record,begin=[0,0,0,0,0],size=[-1,-1,1,-1,-1]),axis=2)
    #print('here part shape is',sess.run(tf.shape(part)))
    pool , _, _, argmax_z = median_pool_2d_with_argmax(part,[1,pool_sz_z])
    #print('here pool shape is',sess.run(tf.shape(pool)))
    pool_record2= tf.expand_dims(pool,axis=2)
    argmax_record_z = tf.expand_dims(argmax_z,axis=2)
    for kk in range(1,int(width/2)):    
        part = tf.squeeze(tf.slice(pool_record,begin=[0,0,kk,0,0],size=[-1,-1,1,-1,-1]),axis=2)
        #print('part shape is',sess.run(tf.shape(part)))
        pool , _, _, argmax_z =  median_pool_2d_with_argmax(part,[1,pool_sz_z])
        #print('pool shape is',sess.run(tf.shape(pool)))
        pool= tf.expand_dims(pool,axis=2)
        try:
            pool_record2 = tf.concat((pool_record2,pool),2)
        except Exception as e:
            pool_record2 = tf.concat(2,(pool_record2,pool))
        argmax_z = tf.expand_dims(argmax_z,axis=2)
        try:
            argmax_record_z = tf.concat((argmax_record_z,argmax_z),2)
        except Exception as e:
            argmax_record_z = tf.concat(2,(argmax_record_z,argmax_z))
    
    #print('pool shape is',sess.run(tf.shape(pool_record2)))
    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//pool_sz_x)*(height//pool_sz_y)*(depth//pool_sz_z)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//pool_sz_y, width//pool_sz_x, depth//pool_sz_z])
    t1 = tf.transpose(t1, perm=[1, 2, 3, 4, 0])
    
    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//pool_sz_x)*(height//pool_sz_y)*(depth//pool_sz_z)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, height//pool_sz_y, width//pool_sz_x, depth//pool_sz_z, channels])
    
    t3 = tf.to_int64(tf.range(width//pool_sz_x))
    t3 = tf.tile(t3, [batch_size*channels*(height//pool_sz_y)*(depth//pool_sz_z)])
    t3 = tf.reshape(t3, [-1, width//pool_sz_x])
    t3 = tf.transpose(t3, perm=[1, 0])
    t3 = tf.reshape(t3, [width//pool_sz_x, batch_size, height//pool_sz_y, depth//pool_sz_z, channels])
    t3 = tf.transpose(t3, perm=[1, 2, 0, 3, 4])
    
    t4 = tf.to_int64(tf.range(height//pool_sz_y))
    t4 = tf.tile(t4, [batch_size*channels*(width//pool_sz_x)*(depth//pool_sz_z)])
    t4 = tf.reshape(t4, [-1, height//pool_sz_y])
    t4 = tf.transpose(t4, perm=[1, 0])
    t4 = tf.reshape(t4, [height//pool_sz_y, batch_size, width//pool_sz_x, depth//pool_sz_z, channels])
    t4 = tf.transpose(t4, perm=[1, 0, 2, 3, 4])
    
    
    
    dim = (height//pool_sz_y)*(width//pool_sz_x)*(depth//pool_sz_z)*channels*batch_size  
    coords = [tf.reshape(t2,[dim,]),tf.reshape(t4,[dim,]),tf.reshape(t3,[dim,]),tf.to_int64(tf.reshape(argmax_record_z,[dim,])),tf.reshape(t1,[dim,])] 
    coords = tf.transpose(tf.stack(coords))
    argmax_record_x_final = tf.gather_nd(argmax_record_x, coords)
    #print(sess.run(tf.shape(argmax_record_x_final)))
    argmax_record_y_final = tf.gather_nd(argmax_record_y, coords)
    
    indices = [tf.reshape(t2,[dim,]),tf.to_int64(argmax_record_y_final),tf.to_int64(argmax_record_x_final),tf.to_int64(tf.reshape(argmax_record_z,[dim,])),tf.reshape(t1,[dim,])] 
    indices1 = tf.transpose(tf.stack(indices))
    return pool_record2,indices1


def median_pool_2d_with_argmedian(bottom,pool_sz):
    bottom_shape = bottom.get_shape()
    batch_size = bottom_shape[0].value
    height = bottom_shape[1].value
    width = bottom_shape[2].value
    channels = bottom_shape[3].value
    pool_sz_y = pool_sz[0]; pool_sz_x = pool_sz[1]
    patches = tf.extract_image_patches(bottom, ksizes=[1, pool_sz_y, pool_sz_x, 1], strides=[1, pool_sz_y, pool_sz_x, 1],
                                       rates=[1, 1, 1, 1], padding='SAME')
    patches_shape = patches.get_shape()
    patches_with_channel_shape = [-1, patches_shape[1].value, patches_shape[2].value,
                                  patches_shape[3].value // bottom_shape[3].value, bottom_shape[3].value]
    patches = tf.reshape(patches, patches_with_channel_shape)
    patches = tf.transpose(patches, [0, 1, 2, 4, 3])
    m_idx = int(pool_sz_x * pool_sz_y / 2 + 1)
    top, indices = tf.nn.top_k(patches, m_idx, sorted=True)
    temp = tf.slice(top, begin=[0, 0, 0, 0, m_idx - 1], size=[-1, -1, -1, -1, 1])
    median = tf.squeeze(temp, axis=4)
    temp= tf.slice(indices, begin=[0, 0, 0, 0, m_idx - 1], size=[-1, -1, -1, -1, 1])
    median_args = tf.squeeze(temp, axis=4)
    
    median_args_y = median_args // pool_sz_y
    median_args_x = median_args % pool_sz_x

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width // pool_sz_x) * (height // pool_sz_y)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height // pool_sz_y, width // pool_sz_x])
    t1 = tf.transpose(t1, perm=[1, 2, 3, 0])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width // pool_sz_x) * (height // pool_sz_y)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, height // pool_sz_y, width // pool_sz_x, channels])

    offset_y = tf.range(0,height,pool_sz_y);
    offset_y = tf.tile(offset_y, [batch_size * (width // pool_sz_x) * channels])
    offset_y = tf.reshape(offset_y, [-1, height// pool_sz_y])
    offset_y = tf.transpose(offset_y, perm=[1, 0])
    offset_y = tf.reshape(offset_y, [height// pool_sz_y, batch_size, width // pool_sz_x, channels])
    offset_y = tf.transpose(offset_y, perm=[1, 0, 2, 3])
    
    offset_x = tf.range(0,width,pool_sz_x);
    offset_x = tf.tile(offset_x, [batch_size * (height // pool_sz_y) * channels])
    offset_x = tf.reshape(offset_x, [-1, width// pool_sz_x])
    offset_x = tf.transpose(offset_x, perm=[1, 0])
    offset_x = tf.reshape(offset_x, [width// pool_sz_x, batch_size, height // pool_sz_y, channels])
    offset_x = tf.transpose(offset_x, perm=[1, 2, 0, 3])
    
    median_args_y = tf.add(median_args_y,offset_y)
    median_args_x = tf.add(median_args_x,offset_x)

    dim = (height // pool_sz_y) * (width // pool_sz_x) * channels * batch_size
    indices = [tf.reshape(t2, [dim, ]), tf.to_int64(tf.reshape(median_args_y, [dim, ])), tf.to_int64(tf.reshape(median_args_x, [dim, ])), tf.reshape(t1, [dim, ])]
    indices1 = tf.transpose(tf.stack(indices))
    return median, indices1, median_args_y, median_args_x




def unpool_layer_batch_unraveled_indices(bottom, indices, pool_sz):
    bottom_shape = tf.shape(bottom)
    try: #for 3d unpooling
        pool_sz_y = pool_sz[0]; pool_sz_x = pool_sz[1]; pool_sz_z = pool_sz[2];
        top_shape = [bottom_shape[0], bottom_shape[1]*pool_sz_y, bottom_shape[2]*pool_sz_x, bottom_shape[3]*pool_sz_z, bottom_shape[4]]
    except Exception as e: #for 2d unpooling
        pool_sz_y = pool_sz[0]; pool_sz_x = pool_sz[1];
        top_shape = [bottom_shape[0], bottom_shape[1]*pool_sz_y, bottom_shape[2]*pool_sz_x, bottom_shape[3]]

    #x1 = tf.transpose(bottom, perm=[0, 4, 1, 2, 3])
    values = tf.reshape(bottom, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
