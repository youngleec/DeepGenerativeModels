import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
import numpy as np


# Define a fully connected layer
def fc_op(input_op, output_dim, name, activation = tf.nn.relu):

    input_dim = input_op.get_shape()[-1].value
    
    with tf.variable_scope(name):
        W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [output_dim], initializer=tf.zeros_initializer)
        logits = tf.matmul(input_op, W) + b
        f = activation(logits)      
    
    return logits, f


# Q Network
def QNetwork(data):

    # Q Network layer 1
    _ , QNet_layer1 = fc_op(data, 200, 'QNet_layer1')

    # Q Network layer 2
    QNet_z_mean, _ = fc_op(QNet_layer1, 200, 'QNet_layerZM')
    QNet_z_log_sigma_sq, _ = fc_op(QNet_layer1, 200, 'QNet_layerZS')

    return QNet_z_mean, QNet_z_log_sigma_sq


# Sample from Q distribution
def SampleQ(z_mean, z_log_sigma_sq):

    eps = tf.random_normal(tf.stack([tf.shape(z_log_sigma_sq)[0], 200]), 0.0, 1.0, dtype = tf.float32)

    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

    return z


# P Network
def PNetwork(z, data_dim):
    
    # P Network layer 1
    _ , PNet_layer1 = fc_op(z, 200, 'PNet_layer1')

    # P Network layer 2
    PNet_x_mean, _ = fc_op(PNet_layer1, data_dim, 'PNet_layerXM')
    PNet_x_log_sigma_sq, _ = fc_op(PNet_layer1, data_dim, 'PNet_layerXS')

    return PNet_x_mean, PNet_x_log_sigma_sq


# Loss Function
def loss(z_mean, z_log_sigma_sq, x_mean, x_log_sigma_sq, data):
    
    KL = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)

    negative_log_likelihood = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(x_mean, data), 2.0))

    loss = tf.reduce_mean(negative_log_likelihood + KL)

    return loss


# Training
def training(loss, learning_rate = 0.001):
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    train_op = optimizer.minimize(loss)
    
    return train_op


# Reconstruction
def reconstruction(sess, x_mean_op, x_sample, save_name = 'fig.png'):

    num = x_sample.shape[0]
    x_reconstruct = sess.run(x_mean_op, {data: x_sample})

    plt.figure(figsize=(8, 12))
    for i in range(num):
        plt.subplot(num, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(num, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_name, format='png')


# Compute Loss
def compute_loss(sess, loss_op, x_sample):
    return sess.run(loss_op, {data: x_sample})


# Rescale (Normlization)
def min_max_scale(x_train, x_test):
    preprocessor = prep.MinMaxScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test


# Get random minibatch data
def get_random_block_from_data(x_data, batch_size):
    start_index = np.random.randint(0, len(x_data) - batch_size)
    return x_data[start_index : (start_index + batch_size)]



# Main
data_dim = 784
batch_size = 100
minibatch_number = 10000


# Build a Computational Graph
data = tf.placeholder(tf.float32, [None, data_dim])

z_mean, z_log_sigma_sq = QNetwork(data)

z = SampleQ(z_mean, z_log_sigma_sq)

x_mean, x_log_sigma_sq = PNetwork(z, data_dim)

loss = loss(z_mean, z_log_sigma_sq, x_mean, x_log_sigma_sq, data)

train_op = training(loss)


# Build a Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Preprocess Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x_train, x_test = min_max_scale(mnist.train.images, mnist.test.images)


# Train the Model
for i in range(minibatch_number):
    batch = get_random_block_from_data(x_train, batch_size)
    loss_num, _ = sess.run([loss, train_op], {data: batch})
    if (i+1) % 100 == 0 or i == 0:
        print("minibatch number: ", i+1, "   training minibatch loss: ", loss_num)


# Test the Model
test_loss = compute_loss(sess, loss, x_test)
print("Test loss: ", test_loss)


# Reconstruct Data
x_sample = get_random_block_from_data(x_test, 5)
reconstruction(sess, x_mean, x_sample)

