import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
import numpy as np


'''Sample from Gumbel(0, 1)'''
def sample_gumbel(shape, eps=1e-20):

    U = tf.random_uniform(shape, minval=0, maxval=1)  
    return -tf.log(-tf.log(U + eps) + eps)


'''Sample from Gumbel-Softmax distribution'''
def gumbel_softmax_sample(logits, temperature):
    
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


'''Sample from Gumbel-Softmax distribution and optionally discretize'''
def gumbel_softmax(logits, temperature, hard=False):
    """
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        #k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    
    return y


'''Define a fully connected layer'''
def fc_op(input_op, output_dim, name, activation = tf.nn.relu):

    input_dim = input_op.get_shape()[-1].value
    
    with tf.variable_scope(name):
        W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [output_dim], initializer=tf.zeros_initializer)
        logits = tf.matmul(input_op, W) + b
        f = activation(logits)      
    
    return logits, f


'''Reconstruction'''
def reconstruction(sess, PNet_x, x_sample, save_name = 'fig.png'):

    num = x_sample.shape[0]
    x_reconstruct = sess.run(PNet_x, {data: x_sample, tau: np_temp})

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


'''Compute Loss'''
def compute_loss(sess, loss_op, x_sample):
    return sess.run(loss_op, {data: x_sample})



'''Main'''
data_dim = 784
batch_size = 100
minibatch_number = 50000

N = 20 # number of categorical variables
K = 10 # number of classes

data = tf.placeholder(tf.float32, [None, data_dim])


'''Q Network'''
# Q Network layer 1
_ , QNet_layer1 = fc_op(data, 200, 'QNet_layer1')

# Q Network layer 2
logit_y, _ = fc_op(QNet_layer1, N*K, 'QNet_layer2', tf.nn.softmax)


'''Sample from Q distribution'''
logit_y = tf.reshape(logit_y, [-1, K])
q_y = tf.nn.softmax(logit_y)
log_q_y = tf.log(q_y + 1e-20)

# temperature
tau = tf.Variable(5.0, name="temperature", trainable=False)

y = tf.reshape(gumbel_softmax(logit_y, tau, hard=False), [-1, N*K])


'''P Network'''
# P Network layer 1
_ , PNet_layer1 = fc_op(y, 200, 'PNet_layer1')

# P Network layer 2
PNet_x, _ = fc_op(PNet_layer1, data_dim, 'PNet_X')
Bernoulli = tf.contrib.distributions.Bernoulli
p_x = Bernoulli(logits = PNet_x)


'''Loss Function'''
kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/K)), [-1,N,K])
KL = tf.reduce_sum(kl_tmp, [1,2])
elbo = tf.reduce_sum(p_x.log_prob(data), 1) - KL
loss = tf.reduce_mean(-elbo)


'''Training'''
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)


'''Build a Session'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


'''Train the Model'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tau0 = 1.0 # initial temperature
np_temp = tau0

ANNEAL_RATE = 0.00003
MIN_TEMP = 0.5

for i in range(minibatch_number):
    batch, _ = mnist.train.next_batch(batch_size)
    loss_num, _ = sess.run([loss, train_op], {data: batch, tau: np_temp})
    if (i+1) % 100 == 0 or i == 0:
        print("minibatch number: ", i+1, "   training minibatch loss: ", loss_num)

    if (i+1) % 1000 == 0:
        np_temp = np.maximum(tau0 * np.exp(-ANNEAL_RATE * (i+1)), MIN_TEMP)


'''Test the Model'''
x_test = mnist.test.images
test_loss = compute_loss(sess, loss, x_test)
print("Test loss: ", test_loss)


'''Reconstruct Data'''
x_sample, _ = mnist.test.next_batch(5)
reconstruction(sess, PNet_x, x_sample)

