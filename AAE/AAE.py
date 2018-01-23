import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


data_dim = 784
z_dim = 32
hidden_dim = 256
batch_size = 128
minibatch_number = 10000000
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


# Q Network
def QNetwork(data):

    QParameter = []

    # Q Network layer 1
    _ , QNet_layer1 = fc_op(data, hidden_dim, 'QNet_layer1', QParameter)

    # Q Network layer 2
    QNet_layer2, _ = fc_op(QNet_layer1, z_dim, 'QNet_layer2', QParameter)

    return QNet_layer2, QParameter


# P Network
def PNetwork(z):

    PParameter = []

    # P Network Layer 1
    _ , PNet_layer1 = fc_op(z, hidden_dim, 'PNet_layer1', PParameter)

    # P Network Layer 2
    PNet_logit, PNet_layer = fc_op(PNet_layer1, data_dim, 'PNet_layer2', PParameter, tf.nn.sigmoid)

    return PNet_logit, PNet_layer, PParameter


# Discriminator Network
def DNetwork(z):
    
    DParameter = []

    # Discriminator Network Layer 1
    _ , DNet_layer1 = fc_op(z, hidden_dim, 'DNet_layer1', DParameter, reuse = REUSE)
    
    # Discriminator Network Layer 2
    Dlogit, DNet_layer2 = fc_op(DNet_layer1, 1, 'DNet_layer2', DParameter, tf.nn.sigmoid, reuse = REUSE)

    return Dlogit, DNet_layer2, DParameter


# Sample from the prior
def SampleZ(batch_size, z_dim = z_dim):
    
    #return np.random.uniform(-1.0, 1.0, (batch_size, z_dim))
    return np.random.normal(0.0, 1.0, (batch_size, z_dim))


# Reconstruction Plot
def ReconsPlot(sess, j, num = 5):

    batch, _ = mnist.train.next_batch(num)
    sampleX = sess.run(PNet_x, {data: batch})

    plt.figure(figsize=(8, 12))
    for i in range(num):
        plt.subplot(num, 2, 2*i + 1)
        plt.imshow(batch[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(num, 2, 2*i + 2)
        plt.imshow(sampleX[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('ReconsFig'+str(j)+'.png', format='png')
    plt.close()


# Latent Space Plot
def LatentPlot(sess, j, num = 5):

    z_prior = SampleZ(2 * num, z_dim)
    sampleX = sess.run(PNet_x, {QNet_z: z_prior})

    plt.figure(figsize=(8, 12))
    for i in range(num):
        plt.subplot(num, 2, 2*i + 1)
        plt.imshow(sampleX[2*i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Latent Space Plot")
        plt.colorbar()
        plt.subplot(num, 2, 2*i + 2)
        plt.imshow(sampleX[2*i+1].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Latent Space Plot")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('LatentFig'+str(j)+'.png', format='png')
    plt.close()


# Main
# Build a Computational Graph
data = tf.placeholder(tf.float32, [None, data_dim])
z = tf.placeholder(tf.float32, [None, z_dim])

QNet_z, QParameter = QNetwork(data)

PNet_logit, PNet_x, PParameter = PNetwork(QNet_z)

D_prior_logit, D_prior, DParameter = DNetwork(z)

REUSE = True
D_q_logit, D_q, _ = DNetwork(QNet_z)

# Loss
D_loss_prior = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_prior_logit, labels = tf.ones_like(D_prior_logit)))
D_loss_q = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_q_logit, labels = tf.zeros_like(D_q_logit)))
D_loss = D_loss_prior + D_loss_q

reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = PNet_logit, labels = data), axis = 1))
AE_loss = reconstruction_loss - D_loss

# Training
D_optimizer = tf.train.AdamOptimizer()
D_train = D_optimizer.minimize(D_loss, var_list = DParameter)

AE_optimizer = tf.train.AdamOptimizer()
AE_train = AE_optimizer.minimize(AE_loss, var_list = QParameter + PParameter)

# Build a Session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for i in range(minibatch_number):
    batch, _ = mnist.train.next_batch(batch_size)

    z_sample = SampleZ(batch_size, z_dim)
    
    _ , AE_loss_num = sess.run([AE_train, AE_loss], {data: batch, z: z_sample})

    _ , D_loss_num = sess.run([D_train, D_loss], {data: batch, z: z_sample})
    
    if (i+1) % 1000 == 0 or i == 0:
        print("Minibatch Number: ", i+1, "   AE Loss: ", AE_loss_num)
        print("Minibatch Number: ", i+1, "   Discriminator Loss: ", D_loss_num)
    
    if (i+1) % 10000 == 0 or i == 0:
        ReconsPlot(sess, i+1)
        LatentPlot(sess, i+1)



