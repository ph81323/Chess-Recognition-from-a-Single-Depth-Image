import numpy as np
import h5py
import os
import scipy.io
import tensorflow as tf
import math


# Path to 3d tensor. Tensor.shape is (50,50,100)
val_path = '../pc_real_bias_nolookat_remove/val/'
val_list = []
for dirname in os.listdir(val_path):
    print(dirname)
    for filename in os.listdir(val_path+'/'+dirname):
        val_list.append(dirname+'/'+filename)

print(len(val_list))

def read_data(root,path):
    f = h5py.File(root+path)
    data = np.array(f['answer'])
    f.close()    
    labels = np.zeros((1,6)) 
    #labels = labels - 1
    #labels[0] = int(path[0])
    labels[0][int(path[0])] = 1
    return data, labels

# Accuracy function
def get_accuracy(predictions, labels):
  gt = tf.argmax(labels,1)
  prediction = tf.argmax(predictions,1)
  singleacc = tf.reduce_sum(tf.cast(tf.equal(prediction, gt), tf.float32))
  accuracy = 100 * tf.reduce_mean(tf.cast(tf.equal(prediction, gt), tf.float32))
  #singleacc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1)), tf.float32))
  #accuracy = 100 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1)), tf.float32))
  return  gt,prediction, singleacc, accuracy

num_labels = 6

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
            
        hidden1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))
        dropout1 = tf.nn.dropout(hidden1, 0.2)
        
        # Conv2
        conv2 = tf.nn.conv3d(dropout1, conv2_w, [1, 1, 1, 1, 1],padding='VALID')
        hidden2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))
        dropout2 = tf.nn.dropout(hidden2, 0.3)
        
		#Pool1
        pool1 = tf.nn.max_pool3d(dropout2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')
        dropout3 = tf.nn.dropout(pool1, 0.3)
        
        normalize3_flat = tf.reshape(dropout3, [-1, 11*11*23*32])
        

        #FC1
        fc1 = tf.add(tf.matmul(normalize3_flat, fc1_w) , fc1_bias)
        hidden3 = tf.nn.relu(fc1)  
        dropout3 = tf.nn.dropout(hidden3, 0.4)

        #FC2                
        res = tf.add(tf.matmul(dropout3, fc2_w) , fc2_bias )
        return res
  
   
    # Training computation
    local_res = model(tf_train_dataset)

    #with tf.name_scope("cost_function") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = local_res))
    tf.scalar_summary("cost_function", cross_entropy)
    
    # Optimizer
    #train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy)
    # Predictions for the training, validation, and test data
    #with tf.name_scope("accuracy") as scope:
    gt, prediction, singleacc ,accuracy = get_accuracy(local_res, tf_train_labels)
    tf.scalar_summary("accuracy", accuracy)

    valid_prediction = tf.nn.softmax(model(tf_train_dataset))
    print ('Graph was built')
    
    merged_summary_op = tf.merge_all_summaries()
    

tf.reset_default_graph()

    #print(v_)   

new_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(graph=new_graph,config=tf.ConfigProto(gpu_options=gpu_options)) as session:   
    
    new_saver = tf.train.import_meta_graph('itrichess_new2.ckpt.meta')    
    new_saver.restore(session, './itrichess_new2.ckpt')
    saver_pretrained = tf.train.Saver([v for v in tf.all_variables()])
    session.run(tf.initialize_all_variables())
    #all_vars = tf.trainable_variables()
    #print(graph.get_operation_by_name("conv1/conv1_w"))
    #all_vars = tf.get_collection('vars')
    #print([v.op.name for v in tf.all_variables()])
    #for v in all_vars:
    #    print v.name
    #    v_ = session.run(v)
        
    val_pred = 0        
    for j in range(len(val_list)):        
        
        pc, singlelabel = read_data(val_path,val_list[j])
        pc = pc[None,:,:,:,None].astype('float32')
        
        print(model(pc))
        #tf_train_dataset = graph.get_operation_by_name("tf_train_dataset").outputs[0]
        #tf_train_labels = graph.get_operation_by_name("tf_train_labels").outputs[0]
        #local_res=graph.get_operation_by_name("local_res").outputs[0]
        
        #print(session.run([local_res], feed_dict = {tf_train_dataset: pc, tf_train_labels: singlelabel}))
        #print session.run()
        #val_prediction, val_label, val_acc= session.run([prediction,gt,singleacc], feed_dict={tf_train_dataset: pc, tf_train_labels: singlelabel})
        #print("step %d" % j )
        #print(val_label)
        #print(val_prediction)
        #print(type(val_acc))
        
        val_pred = val_pred + val_acc
    val_accuracy = 100*val_pred/len(val_list)
    print("val accuracy: %.1f%%" % val_accuracy)      
   
