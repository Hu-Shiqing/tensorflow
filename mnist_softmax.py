import data_loader
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np

def add_layer(layer_name, inputs, in_size, out_size, active_function=None):
    with tf.name_scope(layer_name):
        W = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.zeros([out_size]))
        Wx_plus_b = tf.matmul(inputs, W)+b
        if active_function is None:
            return Wx_plus_b
        else:
            return active_function(Wx_plus_b)
        
def test_model(dataset, label):
    test_data_pred = tf.cast(tf.equal(tf.argmax(y_pred,1), tf.argmax(label_y,1)), tf.float32)
    test_data_accu = tf.reduce_mean(test_data_pred)
    test_result = sess.run(test_data_accu, feed_dict={label_x: dataset, label_y: label})
    print('result: %f'%(test_result))
    return test_result

# ------------------------------------------------------------------------------
# top settings
# ------------------------------------------------------------------------------
n_training_data = 10000
n_test_data = 1000

n_epoch = 10
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
label_x = tf.placeholder(tf.float32, [None, 784])
label_y = tf.placeholder(tf.float32, [None,  10])

y_pred=add_layer('output_layer', label_x, 784, 10, tf.nn.softmax)
'''
W = tf.Variable(tf.random_normal([784, 10])); # [1, 784] x [784, 10] = [1, 10]
b = tf.Variable(tf.zeros([10]))
z_pred = tf.matmul(label_x, W) + b
y_pred = tf.nn.softmax(z_pred)
'''
# ------------------------------------------------------------------------------
# step 3: define cost function
# ------------------------------------------------------------------------------
# cross_entropy for softmax is simpler than common one...



# ------------------------------------------------------------------------------
# step 4: start session & training
# ------------------------------------------------------------------------------
cross_entropy = -tf.reduce_sum(label_y*tf.log(y_pred))
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
    
print('Start training... %d samples'%(len(training_data)))
loss = list()
loss_epoch = list()
test_result = list()
a_result = list()
for i in range(n_epoch):
    start_time = time.time()
    random.shuffle(training_data)
    mini_batches = [training_data[k : k + mini_batch_size]
                    for k in range(0, len(training_data), mini_batch_size)]
    for j in range(len(mini_batches)):
        x = [list(m[0]) for m in mini_batches[j]]
        y = [list(m[1]) for m in mini_batches[j]]
        #print('x: ', x, '\ny: ', y)
        #_, l, z1, y1, w1, b1= sess.run([train_step, cross_entropy, z_pred, y_pred, W, b], feed_dict={label_x: x, label_y: y})
        #print('z: ', z1, '\ny: ', y1, '\nb:', b1)
        _, l = sess.run([train_step, cross_entropy], feed_dict={label_x: x, label_y: y})
        loss.append(l)
    #loss_epoch.append(tf.reduce_mean(loss))
    loss_epoch.append(np.mean(loss))
    print('Epoch %d, time: %d seconds'%(i, time.time()-start_time))
    test_result.append(test_model(test_dataset, test_label))

    a_dataset=[list(m[0]) for m in training_data[0:1000]]
    a_label=[list(m[1]) for m in training_data[0:1000]]
    a_result.append(test_model(a_dataset, a_label))
    #print(loss_epoch)

plt.subplot(3,1,1)
plt.plot(loss, c='red', label='loss')
plt.legend(loc='best')
plt.subplot(3,1,2)
plt.plot(loss_epoch)
plt.subplot(3,1,3)
plt.plot(test_result, 'r', label='test dataset')
plt.plot(a_result, 'b', label='training dataset')
plt.legend(loc='best')
plt.show()

# ------------------------------------------------------------------------------
# step 5: validate result
# ------------------------------------------------------------------------------


# === END ===
