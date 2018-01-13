import data_loader
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np

def test_result():
    test_data_pred = tf.cast(tf.equal(tf.argmax(y_pred,1), tf.argmax(label_y,1)), tf.float32)
    test_data_accu = tf.reduce_mean(test_data_pred)
    feed_dict = {label_x: test_dataset, label_y: test_label, keep_prob: 1.0}
    test_result = sess.run(test_data_accu, feed_dict=feed_dict)
    print('result: %f'%(test_result))
    return test_result

# ------------------------------------------------------------------------------
# top settings
# ------------------------------------------------------------------------------
n_training_data = 10000
n_test_data = 1000

n_epoch = 30
mini_batch_size = 100
learn_rate = 0.005

# ------------------------------------------------------------------------------
# step 1: generate data 
# ------------------------------------------------------------------------------
training_data           = data_loader.load_training_data(n_training_data)
test_dataset,test_label = data_loader.load_test_data    (n_test_data    )

# ------------------------------------------------------------------------------
# step 2: setup the model
# ------------------------------------------------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def add_cnn_layer(layer_name, X, in_channel, out_channel,
                  conv_size, conv_strides, pool_size, pool_strides,
                  active=tf.nn.relu):
    if tf.name_scope(layer_name):
        w_shape = [conv_size[0], conv_size[1], in_channel, out_channel]
        W = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.1))
        b_shape = [out_channel]
        b = tf.Variable(tf.constant(0.1, shape=b_shape))
        
        conv = tf.nn.conv2d(X, W, strides=conv_strides, padding='SAME')
        A = active(conv + b)
        A_pool = tf.nn.max_pool(A, ksize=pool_size, strides=pool_strides, padding='SAME')
        return A_pool

              
label_x = tf.placeholder(tf.float32, [None, 784])
label_y = tf.placeholder(tf.float32, [None,  10])
x_image = tf.reshape(label_x, [-1,28,28,1]) # 28x28 x 1

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28x28 x 32
h_pool1 = max_pool_2x2(h_conv1) # 14x14 x 32
'''
h_pool1=add_cnn_layer('conv1', x_image,
              in_channel=1, out_channel=32,
              conv_size=[5,5], conv_strides=[1,1,1,1],
              pool_size=[1,2,2,1], pool_strides=[1,2,2,1],
              active=tf.nn.relu)
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 14x14 x 64
h_pool2 = max_pool_2x2(h_conv2) # 7x7 x 64

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pred=tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#y_pred=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


# ------------------------------------------------------------------------------
# step 4: start session & training
# ------------------------------------------------------------------------------
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_y, logits=y_pred)

cross_entropy = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
    
print('Start training... %d samples'%(len(training_data)))
loss = list()
loss_epoch = list()
result = list()
for i in range(n_epoch):
    start_time = time.time()
    random.shuffle(training_data)
    mini_batches = [training_data[k : k + mini_batch_size]
                    for k in range(0, len(training_data), mini_batch_size)]
    for j in range(len(mini_batches)):
        x = [list(m[0]) for m in mini_batches[j]]
        y = [list(m[1]) for m in mini_batches[j]]
        _, l = sess.run([train_step, cross_entropy], feed_dict={label_x: x, label_y: y, keep_prob: 0.5})
        loss.append(l)
    loss_epoch.append(np.mean(loss))
    print('Epoch %d, time: %d seconds'%(i, time.time()-start_time))
    result.append(test_result())
    #print(loss_epoch)

plt.subplot(3,1,1)
plt.plot(loss)
plt.subplot(3,1,2)
plt.plot(loss_epoch)
plt.subplot(3,1,3)
plt.plot(result)
plt.show()

# ------------------------------------------------------------------------------
# step 5: validate result
# ------------------------------------------------------------------------------


# === END ===
