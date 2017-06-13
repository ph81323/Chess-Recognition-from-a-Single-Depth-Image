import numpy as np
import h5py
import os
import scipy.io
import tensorflow as tf
import math

# Path to 3d tensor. Tensor.shape is (50,50,100)
tensor_path = '../pc_real_bias_nolookat_remove/train/'
val_path = '../pc_real_bias_nolookat_remove/val/'
#datas = np.zeros((1,50,50,100));
#labels = np.zeros((1,6));
filelist = []
val_list = []
print(os.listdir(tensor_path))

for dirname in os.listdir(val_path):
    print(dirname)
    for filename in os.listdir(val_path+'/'+dirname):
        val_list.append(dirname+'/'+filename)
        
for dirname in os.listdir(tensor_path):
    print(dirname)
    for filename in os.listdir(tensor_path+'/'+dirname):
        filelist.append(dirname+'/'+filename)
print(len(filelist))
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


# Graph
batch_size = 64
num_labels = 6

graph = tf.Graph()

with graph.as_default():

    predict = tf.Variable(False)
    # Input data.
    with tf.name_scope('data') as scope:
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 50, 50, 100, 1), name = "train_dataset")
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name = "train_labels")

    # Variables.
    with tf.name_scope('conv1') as scope:
	#conv1
        stdv = 1/math.sqrt(5*5*5*1)
        #tf.random_uniform
        layer1_weights = tf.Variable(tf.random_uniform([5, 5, 5, 1, 32], -stdv, stdv ), name = "conv1_w")
        layer1_biases = tf.Variable(tf.random_uniform([32], -stdv, stdv ), name = "conv1_bias")
    #layer1_biases = tf.Variable(tf.constant(1.0, shape=[32]))#tf.zeros([32])
    with tf.name_scope('conv2') as scope:
	#conv2
        stdv = 1/math.sqrt(3*3*3*32)
        layer2_weights = tf.Variable(tf.random_uniform([3, 3, 3, 32, 32], -stdv, stdv ), name = "conv2_w")   
        layer2_biases = tf.Variable(tf.random_uniform([32], -stdv, stdv ), name = "conv2_bias")
    #layer2_biases = tf.Variable(tf.constant(1.0, shape=[32]))
    with tf.name_scope('fc1') as scope:
    #fc1        
        stdv = 1/math.sqrt(11*11*23*32)
        layer3_weights = tf.Variable(tf.random_uniform([11*11*23*32, 128], -stdv, stdv), name = "fc1_w")
        #layer3_weights = tf.Variable(tf.random_normal([11*11*23*32, 128]), name = "fc1_w")
        layer3_biases = tf.Variable(tf.random_uniform([128]), name = "fc1_bias")    
    #layer3_biases = tf.Variable(tf.constant(1.0, shape=[128]))
    with tf.name_scope('fc2') as scope:
	#fc2
        stdv = 1/math.sqrt(128)
        layer4_weights = tf.Variable(tf.random_uniform([128, num_labels], -stdv, stdv), name = "fc2_w")
        layer4_biases = tf.Variable(tf.random_uniform([num_labels]), name = "fc2_bias")
    #layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    #MODEL     
    def model(data):
        # Conv1
        padding = [[0,0],[1,1],[1,1],[1,1],[0,0]]
        padded_input = tf.pad(data,padding,"CONSTANT")
        #print(padded_input)        
        conv1 = tf.nn.conv3d(padded_input, layer1_weights, [1, 2, 2, 2, 1], padding='VALID')
        #print(conv1)               
        
        #bias1 = conv1 + layer1_biases
            
        hidden1 = tf.nn.relu(tf.nn.bias_add(conv1,layer1_biases))
        #hidden1 = tf.maximum(0.1*bias1,bias1)
        #hidden1 = leakyrelu(bias1,0.1)
        #print(hidden1)
        #print(list(range(len(hidden1.get_shape())-1)))        
        
        #BN1
        #BN1 = bn(hidden1,32)
        
        #print(hidden1)
        dropout1 = tf.nn.dropout(hidden1, 0.2)
        #print(dropout1)
        
        
        # Conv2
        conv2 = tf.nn.conv3d(dropout1, layer2_weights, [1, 1, 1, 1, 1],padding='VALID')
        #print(conv2)
        #bias2 = conv2 + layer2_biases
        #hidden2 = tf.maximum(0.1*bias2,bias2)
        hidden2 = tf.nn.relu(tf.nn.bias_add(conv2,layer2_biases))
        #hidden2 = leakyrelu(bias2,0.1)
        #print(hidden2)
		#BN2
        #BN2 = bn(hidden2,32)
        
		#Pool1
        pool1 = tf.nn.max_pool3d(hidden2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')
        #print(pool1)
        dropout2 = tf.nn.dropout(pool1, 0.3)
        #print(dropout2)
        
        normalize3_flat = tf.reshape(dropout2, [-1, 11*11*23*32])
        

        #FC1
        fc1 = tf.add(tf.matmul(normalize3_flat, layer3_weights) , layer3_biases)
        #print(fc1)
        #fc1 = tf.add(tf.matmul(normalize3_flat, layer3_weights), layer3_biases)
        hidden3 = tf.nn.relu(fc1)  
        #print(hidden3)
        dropout3 = tf.nn.dropout(hidden3, 0.4)
        #print(dropout3)

        #FC2                
        res = tf.add(tf.matmul(dropout3, layer4_weights) , layer4_biases )
        return res

    
    # Training computation
    local_res = model(tf_train_dataset)

    with tf.name_scope("cost_function") as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = local_res))
    tf.scalar_summary("cost_function", cross_entropy)
    
    
    # Optimizer
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    with tf.name_scope("accuracy") as scope:
        gt, prediction, singleacc ,accuracy = get_accuracy(local_res, tf_train_labels)
    tf.scalar_summary("accuracy", accuracy)


    valid_prediction = tf.nn.softmax(model(tf_train_dataset))
    print ('Graph was built')
    
    merged_summary_op = tf.merge_all_summaries()

# Session
epochs = 100
#each epochs will see all images
steps_per_epoch = int(len(filelist)/batch_size) + 1
if (len(filelist) % batch_size) == 0:
    steps_per_epoch -= 1
print ('STEPS %d' % steps_per_epoch)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as session:    
    train_writer = tf.train.SummaryWriter('./', session.graph)
    session.run(tf.initialize_all_variables())
    
    for epch in xrange(0, epochs):
        np.random.shuffle(filelist)    #shuffle filelist
        print ('EPOCH %d' % epch)
        total_prediction = 0
        for step in range(steps_per_epoch):
            offset = (step * batch_size) % len(filelist)
            
            if (len(filelist) % batch_size) == 0:
                offset = (step * batch_size) % (len(filelist) +1)
                
            # Generate a minibatch.            
            datas = np.zeros((1,50,50,100))
            labels = np.zeros((1,6));
            i = 0;
            while (offset+i) < len(filelist) and (offset+i) < offset + batch_size:
                pc, singlelabel = read_data(tensor_path,filelist[offset+i])
                datas  = np.concatenate((datas,pc[np.newaxis,...]),axis = 0) 
                labels = np.concatenate((labels,singlelabel),axis = 0)                  
                i += 1
            if (offset+i) < offset + batch_size:
                left = batch_size - i
                offset = 0
                for i in range(left):
                    pc, singlelabel = read_data(tensor_path,filelist[offset+i])
                    datas  = np.concatenate((datas,pc[np.newaxis,...]),axis = 0)
                    labels = np.concatenate((labels,singlelabel),axis = 0)  
                
            datas = datas[1:datas.shape[0],:,:,:]
            batch_labels = labels[1:labels.shape[0],:]     #delete first labels
            batch_data = datas.astype('float32')
            batch_data = batch_data[:,:,:,:,None]
            
            rawoutput,_,train_labels, train_accuracy, train_prediction,_ ,singleaccuracy, summary_str = session.run([local_res, cross_entropy,gt, accuracy, prediction, train_step, singleacc, merged_summary_op], feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
                      
            total_prediction = int(total_prediction) + int(singleaccuracy)
            train_writer.add_summary(summary_str, step)
            
        total_accuracy = 100*total_prediction/ (batch_size * (step + 1))
        print("Epoch %d accuracy: %.1f%%" % (epch, total_accuracy))
        
        #val
        val_pred = 0        
        for j in range(len(val_list)):
            #print(val_list[j])
            pc, singlelabel = read_data(val_path,val_list[j])
            pc = pc[None,:,:,:,None].astype('float32')
            val_prediction, val_label, val_acc,_ = session.run([prediction,gt,singleacc,merged_summary_op],                                                 feed_dict={tf_train_dataset: pc, tf_train_labels: singlelabel})

            
            val_pred = val_pred + val_acc
        val_accuracy = 100*val_pred/len(val_list)
        print("Epoch %d val accuracy: %.1f%%" % (epch, val_accuracy))
           


