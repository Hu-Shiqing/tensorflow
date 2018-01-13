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
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if active_function is None:
            return Wx_plus_b, W
        else:
            return active_function(Wx_plus_b), W

def test_model(dataset, label, verbose=''):
    prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(label_y,1))
    accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
    feed_dict = {label_x: dataset, label_y: label, keep_prob: 1.0}
    test_result = sess.run(accuracy, feed_dict=feed_dict)
    print('[%s] result: %f'%(verbose, test_result))
    return test_result

# ------------------------------------------------------------------------------
# top settings
# ------------------------------------------------------------------------------
test_relu = 1
n_training_data = 50000
n_test_data = 1000
n_epoch = 30
mini_batch_size = 100
learn_rate = 0.005
active_function=tf.nn.softmax
#active_function=tf.nn.sigmoid
keep_prob_nn = 0.5
beta = 0.5

if test_relu == 1:
    n_training_data = 1000
    n_test_data = 100
    n_epoch = 10
    mini_batch_size = 100
    learn_rate = 0.005
    active_function=tf.nn.relu
    keep_prob_nn = 1.0
    beta = 0.0

n_input =784
n_hidden=200
n_output=10
# ------------------------------------------------------------------------------
# step 1: generate data 
# ------------------------------------------------------------------------------
training_data           = data_loader.load_training_data(n_training_data)
test_dataset,test_label = data_loader.load_test_data    (n_test_data    )
train_dataset=[list(m[0]) for m in training_data[0:n_test_data]]
train_label  =[list(m[1]) for m in training_data[0:n_test_data]]

# ------------------------------------------------------------------------------
# step 2: setup the model
# ------------------------------------------------------------------------------
label_x = tf.placeholder(tf.float32, [None, 784])
label_y = tf.placeholder(tf.float32, [None,  10])
keep_prob = tf.placeholder("float")

hidden, w1=add_layer('hidden_layer', label_x, n_input, n_hidden, active_function)
y_pred, w2=add_layer('output_layer', hidden, n_hidden, n_output, active_function)

# ------------------------------------------------------------------------------
# step 3: start session & training
# ------------------------------------------------------------------------------
#l2_loss = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
#cross_entropy = -tf.reduce_sum(label_y*tf.log(y_pred)) + beta * l2_loss
#cross_entropy = tf.reduce_mean(cross_entropy + beta * (regularizer1+regularizer2))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_y, logits=y_pred)
cross_entropy = tf.reduce_mean(cross_entropy)

#optimizer  = tf.train.GradientDescentOptimizer(learn_rate)
optimizer  = tf.train.AdamOptimizer(learn_rate)
train_step = optimizer.minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
    
print('Start training... %d samples'%(len(training_data)))
loss = list()
loss_epoch = list()
test_result = list()
train_result = list()
for i in range(n_epoch):
    start_time = time.time()
    random.shuffle(training_data)
    mini_batches = [training_data[k : k + mini_batch_size]
                    for k in range(0, len(training_data), mini_batch_size)]
    for j in range(len(mini_batches)):
        x = [list(m[0]) for m in mini_batches[j]]
        y = [list(m[1]) for m in mini_batches[j]]
        feed_dict={label_x: x, label_y: y, keep_prob: keep_prob_nn}
        _, l = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
        loss.append(l)
    loss_epoch.append(np.mean(loss))
    print('Epoch %d, time: %d seconds'%(i, time.time()-start_time))
    test_result.append(test_model(test_dataset, test_label, 'test dataset'))
    train_result.append(test_model(train_dataset, train_label, 'training dataset'))
    #print(loss_epoch)



plt.subplot(2,1,1)
plt.plot(loss_epoch, label='loss')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(train_result, 'b', label='training dataset')
plt.plot(test_result , 'r', label='test dataset')
plt.legend(loc='best')
plt.show()


# === END ===
