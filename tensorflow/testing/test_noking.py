import numpy as np
import h5py
import os
import scipy.io
import tensorflow as tf
import math
import preprocess as pre

def leakyrelu(x, alpha=0., max_value = None):
    '''ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def bn(input,depth):
    batch_mean,batch_variance = tf.nn.moments(input, [0,1,2,3], name='moments')
    beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    epsilon = 1e-3
        
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_variance])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_variance)
        
    pred = tf.placeholder(tf.bool)
    pred =  tf.Variable(True,  name='pred')
    mean, var = tf.cond(pred, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_variance)))
    BN = tf.nn.batch_normalization(input,mean,var,beta,gamma,epsilon)
    return BN

# Accuracy function
def get_accuracy(predictions, labels):
  gt = tf.argmax(labels,1)
  prediction = tf.argmax(predictions,1)
  singleacc = tf.reduce_sum(tf.cast(tf.equal(prediction, gt), tf.float32))
  accuracy = 100 * tf.reduce_mean(tf.cast(tf.equal(prediction, gt), tf.float32))
  return  gt,prediction, singleacc, accuracy

num_labels = 5

graph = tf.Graph()

with graph.as_default():
    
    predict = tf.Variable(False)
    with tf.name_scope('data') as scope:
    # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 50, 50, 100, 1), name = "tf_train_dataset")
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name = "tf_train_labels")
    
    with tf.name_scope('conv1') as scope:
    # Variables.
    #conv1
        conv1_w = tf.Variable(tf.random_normal([5, 5, 5, 1, 32]), name = "conv1_w")
        conv1_bias = tf.Variable(tf.constant(1.0, shape=[32]), name = "conv1_bias")

    with tf.name_scope('conv2') as scope:
    #conv2
        conv2_w = tf.Variable(tf.random_normal([3, 3, 3, 32, 32]), name = "conv2_w")   
        conv2_bias = tf.Variable(tf.constant(1.0, shape=[32]), name = "conv2_bias")
    
    with tf.name_scope('fc1') as scope:
    #fc1        
        fc1_w = tf.Variable(tf.random_normal([11*11*23*32, 128]), name = "fc1_w")
        fc1_bias = tf.Variable(tf.constant(1.0, shape=[128]), name = "fc1_bias")    
    
    with tf.name_scope('fc2') as scope:
    #fc2
        fc2_w = tf.Variable(tf.random_normal([128, num_labels]), name = "fc2_w")
        fc2_bias = tf.Variable(tf.constant(1.0, shape=[num_labels]), name = "fc2_bias")
    
  
    #MODEL     
    def model(data):
        # Conv1
        padding = [[0,0],[1,1],[1,1],[1,1],[0,0]]
        padded_input = tf.pad(data,padding,"CONSTANT")
        conv1 = tf.nn.conv3d(padded_input, conv1_w, [1, 2, 2, 2, 1], padding='VALID')
        hidden1 = leakyrelu(tf.nn.bias_add(conv1,conv1_bias),0.1)        
        
        dropout1 = tf.nn.dropout(hidden1, 0.8)   #keep probability        
        
        # Conv2
        conv2 = tf.nn.conv3d(dropout1, conv2_w, [1, 1, 1, 1, 1],padding='VALID')
        hidden2 = leakyrelu(tf.nn.bias_add(conv2,conv2_bias),0.1)
		
        
	#Pool1
        pool1 = tf.nn.max_pool3d(hidden2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')
        dropout2 = tf.nn.dropout(pool1, 0.7)
        
        normalize3_flat = tf.reshape(dropout2, [-1, 11*11*23*32])
        #FC1
        fc1 = tf.add(tf.matmul(normalize3_flat, fc1_w) , fc1_bias)
        hidden3 = tf.nn.relu(fc1)  
        dropout3 = tf.nn.dropout(hidden3, 0.6)

        #FC2                
        res = tf.add(tf.matmul(dropout3, fc2_w) , fc2_bias )
        return res
     
    # Training computation
    local_res = model(tf_train_dataset)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = local_res))
    tf.scalar_summary("cost_function", cross_entropy)
    
    # Predictions for the training, validation, and test data
    gt, prediction, singleacc ,accuracy = get_accuracy(local_res, tf_train_labels)
    tf.scalar_summary("accuracy", accuracy)

    valid_prediction = tf.nn.softmax(model(tf_train_dataset))
    print ('Graph was built')
    
    merged_summary_op = tf.merge_all_summaries()


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    
    saver = tf.train.Saver()
    saver.restore(session, './itrichess_noking.ckpt')
   
    pc, _ = pre.preprocess()
    pc = pc[None,:,:,:,None].astype('float32')
    val_prediction= session.run([prediction], feed_dict={tf_train_dataset: pc})
     
    if val_prediction[0] == 0 :
        print("Prediction : bishop")    
    elif val_prediction[0] == 1 :
        print("Prediction : knight")
    elif val_prediction[0] == 2 :
        print("Prediction : pawn")
    elif val_prediction[0] == 3 :
        print("Prediction : queen")
    elif val_prediction[0] == 4 :
        print("Prediction : rook")

