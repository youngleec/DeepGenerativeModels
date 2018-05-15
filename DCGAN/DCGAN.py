#Conditional DCGAN on MNIST
#Refer to https://github.com/carpedm20/DCGAN-tensorflow
import time
import os
import numpy as np
import tensorflow as tf

from ops import *
from utils import *

learning_rate = 0.0002
beta1 = 0.5

epoch = 25
batch_size = 64
sample_num = 64
sample_dir = "samples"
checkpoint_dir = "checkpoint"

input_height = 28
input_width = 28
output_height = 28
output_width = 28

y_dim = 10   #Dimension of y
z_dim = 100   #Dimension of z

gf_dim = 64   #Dimension of gen filters in first conv layer
df_dim = 64   #Dimension of discrim filters in first conv layer

gfc_dim = 1024   #Dimension of gen units for fully connected layer
dfc_dim = 1024   #Dimension of discrim units for fully connected layer

data_X, data_y = load_mnist()
c_dim = data_X[0].shape[-1]

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

#Build the Model
def generator(z, y):
    with tf.variable_scope("generator") as scope:
        batch_size_ = y.get_shape().as_list()[0]
        s_h, s_w = output_height, output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        #Batch Normalization
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')

        yb = tf.reshape(y, [-1, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        h0 = tf.nn.relu(g_bn0(linear(z, gfc_dim, 'g_h0_lin')))
        h0 = tf.concat([h0, y], 1)

        h1 = tf.nn.relu(g_bn1(linear(h0, gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [-1, s_h4, s_w4, gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(g_bn2(deconv2d(h1, [batch_size_, s_h2, s_w2, gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [batch_size_, s_h, s_w, c_dim], name='g_h3'))

def sampler(z, y):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        batch_size_ = y.get_shape().as_list()[0]
        s_h, s_w = output_height, output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        #Batch Normalization
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')

        yb = tf.reshape(y, [-1, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        h0 = tf.nn.relu(g_bn0(linear(z, gfc_dim, 'g_h0_lin'), train=False))
        h0 = tf.concat([h0, y], 1)

        h1 = tf.nn.relu(g_bn1(linear(h0, gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [-1, s_h4, s_w4, gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(g_bn2(deconv2d(h1, [batch_size_, s_h2, s_w2, gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [batch_size_, s_h, s_w, c_dim], name='g_h3'))

def discriminator(image, y, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        
        batch_size_ = image.get_shape().as_list()[0]

        #Batch Normalization
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')

        yb = tf.reshape(y, [-1, 1, 1, y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, c_dim + y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(d_bn1(conv2d(h0, df_dim + y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [batch_size_, -1])
        h1 = tf.concat([h1, y], 1)

        h2 = lrelu(d_bn2(linear(h1, dfc_dim, 'd_h2_lin')))
        h2 = tf.concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

y = tf.placeholder(tf.float32, [batch_size, y_dim], name='y')
inputs = tf.placeholder(tf.float32, [batch_size, input_height, input_width, c_dim], name='real_images')
z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')

G = generator(z, y)
D, D_logits = discriminator(inputs, y, reuse=False)
D_, D_logits_ = discriminator(G, y, reuse=True)

sample_z = tf.placeholder(tf.float32, [sample_num, z_dim], name='sample_z')
sample_y = tf.placeholder(tf.float32, [sample_num, y_dim], name='sample_y')
sampler = sampler(sample_z, sample_y)

#Loss
d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))

d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))

g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

d_loss = d_loss_real + d_loss_fake

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

saver = tf.train.Saver()
model_name = "DCGAN.model"
model_dir = "mnist"
checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
            
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

#Build a Session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

#Training
counter = 1
start_time = time.time()
for ep in range(epoch):
    batch_idxs = len(data_X) // batch_size

    for idx in range(batch_idxs):
        batch_images = data_X[idx * batch_size: (idx + 1) * batch_size]
        batch_labels = data_y[idx * batch_size: (idx + 1) * batch_size]

        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
        
        D_LOSS, _ = sess.run([d_loss, d_optim], feed_dict={inputs: batch_images, 
                                                                z: batch_z, 
                                                                y: batch_labels})
        G_LOSS, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z, y: batch_labels})
        
        #Update generator twice and update discriminator once
        _ = sess.run([g_optim], feed_dict={z: batch_z, y: batch_labels})

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (ep + 1, epoch, idx + 1, batch_idxs,
              time.time() - start_time, D_LOSS, G_LOSS))

        if counter % 100 == 1:

            sample_z_ = np.random.uniform(-1, 1, [sample_num, z_dim])
            sample_id = np.random.randint(0, len(data_X) - sample_num)
            sample_inputs = data_X[sample_id: sample_id + sample_num]
            sample_labels = data_y[sample_id: sample_id + sample_num]
            
            if sample_num == batch_size:
                samples, D_LOSS, G_LOSS = sess.run([sampler, d_loss, g_loss], 
                            feed_dict={sample_z: sample_z_,
                                       z: sample_z_,
                                       inputs: sample_inputs, 
                                       sample_y: sample_labels,
                                       y: sample_labels})
            
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (D_LOSS, G_LOSS))
            else:
                samples = sess.run([sampler], feed_dict={sample_z: sample_z_, 
                                                         sample_y: sample_labels})
            save_images(samples, image_manifold_size(samples.shape[0]), 
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, ep, idx))

        if counter % 500 == 1:
            saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step = counter)

        counter += 1






