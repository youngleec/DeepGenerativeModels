import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


data_dim = 784
z_dim = 32
hidden_dim = 256
noise_dim = 32
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
def QNetwork(data, eps):

    QParameter = []
    Q_inputs = tf.concat([data, eps], axis=1)

    # Q Network layer 1
    _ , QNet_layer1 = fc_op(Q_inputs, hidden_dim, 'QNet_layer1', QParameter)

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
def TNetwork(x, z):
    
    TParameter = []
    T_inputs = tf.concat([x, z], axis=1)

    # Discriminator Network Layer 1
    _ , TNet_layer1 = fc_op(T_inputs, hidden_dim, 'TNet_layer1', TParameter, reuse = REUSE)

    # Discriminator Network Layer 2
    _ , TNet_layer2 = fc_op(TNet_layer1, hidden_dim, 'TNet_layer2', TParameter, reuse = REUSE)

    # Discriminator Network Layer 3
    _ , TNet_layer3 = fc_op(TNet_layer2, hidden_dim, 'TNet_layer3', TParameter, reuse = REUSE)

    # Discriminator Network Layer 4
    _ , TNet_layer4 = fc_op(TNet_layer3, hidden_dim, 'TNet_layer4', TParameter, reuse = REUSE)
    
    # Discriminator Network Layer 5
    Tlogit, TNet_layer5 = fc_op(TNet_layer4, 1, 'TNet_layer5', TParameter, tf.nn.sigmoid, reuse = REUSE)

    return Tlogit, TNet_layer5, TParameter


# Sample from the prior
def SampleZ(batch_size, z_dim = z_dim):
    
    #return np.random.uniform(-1.0, 1.0, (batch_size, z_dim))
    return np.random.normal(0.0, 1.0, (batch_size, z_dim))


# Sample from the noise distribution
def SampleN(batch_size, n_dim = noise_dim):
    
    return np.random.normal(0.0, 1.0, (batch_size, n_dim))


# Reconstruction Plot
def ReconsPlot(sess, j, num = 5):

    batch, _ = mnist.train.next_batch(num)
    noise = SampleN(num, noise_dim)
    sampleX = sess.run(PNet_x, {data: batch, eps: noise})

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
eps = tf.placeholder(tf.float32, [None, noise_dim])
z = tf.placeholder(tf.float32, [None, z_dim])

QNet_z, QParameter = QNetwork(data, eps)

PNet_logit, PNet_x, PParameter = PNetwork(QNet_z)

T_prior_logit, T_prior, TParameter = TNetwork(data, z)

REUSE = True
T_q_logit, T_q, _ = TNetwork(data, QNet_z)

# Loss
T_loss_prior = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_prior_logit, labels=tf.zeros_like(T_prior_logit)))
T_loss_q = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_q_logit, labels=tf.ones_like(T_q_logit)))
T_loss = T_loss_prior + T_loss_q

log_likelihood = -tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=PNet_logit, labels=data), axis=1))
VAE_loss = - tf.reduce_mean(- T_q) - log_likelihood

# Training
T_optimizer = tf.train.AdamOptimizer()
T_train = T_optimizer.minimize(T_loss, var_list=TParameter)

VAE_optimizer = tf.train.AdamOptimizer()
VAE_train = VAE_optimizer.minimize(VAE_loss, var_list=[QParameter, PParameter])

# Build a Session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for i in range(minibatch_number):
    batch, _ = mnist.train.next_batch(batch_size)

    noise = SampleN(batch_size, noise_dim)

    z_sample = SampleZ(batch_size, z_dim)
    
    _ , VAE_loss_num = sess.run([VAE_train, VAE_loss], {data: batch, eps: noise})

    _ , T_loss_num = sess.run([T_train, T_loss], {data: batch, eps: noise, z: z_sample})
    
    if (i+1) % 1000 == 0 or i == 0:
        print("Minibatch Number: ", i+1, "   VAE Loss: ", VAE_loss_num)
        print("Minibatch Number: ", i+1, "   Discriminator Loss: ", T_loss_num)
    
    if (i+1) % 10000 == 0 or i == 0:
        ReconsPlot(sess, i+1)
        LatentPlot(sess, i+1)



