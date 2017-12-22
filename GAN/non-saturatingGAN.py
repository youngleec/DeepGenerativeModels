import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
import numpy as np


REUSE = None

# Define a fully connected layer
def fc_op(input_op, output_dim, name, p, activation = tf.nn.relu, reuse = REUSE):

    input_dim = input_op.get_shape()[-1].value
    
    with tf.variable_scope(name, reuse = reuse):
        W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [output_dim], initializer=tf.zeros_initializer)
        logits = tf.matmul(input_op, W) + b
        f = activation(logits)
        p += [W, b]
        
    return logits, f


# Discriminator Network
def Discriminator(x):
    
    # Discriminator Parameters
    DParameter = []

    # Discriminator Network Layer 1
    _ , DNlayer1 = fc_op(x, 128, 'DNlayer1', DParameter, reuse = REUSE)
    
    # Discriminator Network Layer 2
    _ , DNlayer2 = fc_op(DNlayer1, 1, 'DNlayer2', DParameter, tf.nn.sigmoid, reuse = REUSE)

    return DNlayer2, DParameter


# Generator Network
def Generator(z, data_dim):

    # Generator Parameters
    GParameter = []

    # Generator Network Layer 1
    _ , GNlayer1 = fc_op(z, 128, 'GNlayer1', GParameter, reuse = REUSE)

    # Generator Network Layer 2
    _ , GNlayer2 = fc_op(GNlayer1, data_dim, 'GNlayer2', GParameter, tf.nn.sigmoid, reuse = REUSE)

    return GNlayer2, GParameter


# Sample from prior
def SampleZ(batch_size, latent_dim):
    return np.random.uniform(-1.0, 1.0, (batch_size, latent_dim))
    #return np.random.normal(0.0, 1.0, (batch_size, latent_dim))


# Plot
def plot(sess, j, latent_dim):

    noise = SampleZ(10, latent_dim)
    sampleG = sess.run(fake_data, {z: noise})

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(sampleG[2*i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Sample from generator")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(sampleG[2*i+1].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Sample from generator")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('fig'+str(j)+'.png', format='png')
    plt.close()


# Main
data_dim = 784
latent_dim = 100
batch_size = 128
minibatch_number = 1000000

# Build a Computational Graph
real_data = tf.placeholder(tf.float32, [None, data_dim])
z = tf.placeholder(tf.float32, [None, latent_dim])

fake_data, G_Parameter = Generator(z, data_dim)
D_real, D_Parameter = Discriminator(real_data)

REUSE = True
D_fake, _ = Discriminator(fake_data)

# Non-saturating Loss
D_Loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real, 1e-8, 1.0)) + tf.log(tf.clip_by_value(1.0 - D_fake, 1e-8, 1.0)))
G_Loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_fake, 1e-8, 1.0)))

# Training
D_Optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
D_Train = D_Optimizer.minimize(D_Loss, var_list=D_Parameter)

G_Optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
G_Train = G_Optimizer.minimize(G_Loss, var_list=G_Parameter)

# Build a Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for i in range(minibatch_number):
    batch, _ = mnist.train.next_batch(batch_size)
    noise = SampleZ(batch_size, latent_dim)
    _ , D_Loss_Num = sess.run([D_Train, D_Loss], {real_data: batch, z: noise})

    noise = SampleZ(batch_size, latent_dim)
    _ , G_Loss_Num = sess.run([G_Train, G_Loss], {z: noise})
    
    if (i+1) % 1000 == 0 or i == 0:
        print("Minibatch Number: ", i+1, "   Generator Loss: ", G_Loss_Num)
        print("Minibatch Number: ", i+1, "   Discriminator Loss: ", D_Loss_Num)
    
    if (i+1) % 10000 == 0 or i == 0:
        plot(sess, i+1, latent_dim)
        

