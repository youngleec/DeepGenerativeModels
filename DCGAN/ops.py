import numpy as np 
import tensorflow as tf

class batch_norm():
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, 
                decay=self.momentum, 
                updates_collections=None, 
                epsilon=self.epsilon, 
                scale=True, 
                is_training=train, 
                scope=self.name)

def linear(input_, output_size, scope='Linear', stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        W = tf.get_variable('W', [shape[-1], output_size], tf.float32, 
                tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], tf.float32, 
                tf.constant_initializer(bias_start))
        
        if with_w:
            return tf.matmul(input_, W) + b, W, b
        else:
            return tf.matmul(input_, W) + b

def conv_cond_concat(x, y):
    x_shapes = x.get_shape().as_list()
    y_shapes = y.get_shape().as_list()
    
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    input_shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        W = tf.get_variable('W', [k_h, k_w, output_shape[-1], input_shape[-1]], 
                initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, W, output_shape, [1, d_h, d_w, 1])

        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, b)

        if with_w:
            return deconv, W, b
        else:
            return deconv

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    input_shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable('W', [k_h, k_w, input_shape[-1], output_dim], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, W, [1, d_h, d_w, 1], padding='SAME')

        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.bias_add(conv, b)

        return conv





